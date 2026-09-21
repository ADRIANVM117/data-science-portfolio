"""Execute frozen D008 only after separate real-execution authorization."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import xgboost


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS, split_discovery_fold  # noqa: E402
from discovery.d007_raw_path_masks_ternary_xgboost import (  # noqa: E402
    XGBOOST_PARAMS,
    assert_ternary_target,
    encode_ternary_target,
    evaluate_d007_probabilities,
    evaluate_majority_baseline,
    make_d007_classifier,
    ternary_probabilities,
)
from discovery.d008_raw_path_beyond_availability_intensity import (  # noqa: E402
    C_COLUMNS,
    P_COLUMNS,
    assert_identical_d008_rows,
    build_d008_representations,
    d008_primary_screen,
    fit_intensity_transformer,
    validate_expected_fold_count,
)
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    RESULTS_DIR,
    load_physical_discovery_data,
)
from src.exp008_recent_movement_intensity import build_recent_window_structure  # noqa: E402


ARTIFACT_NAMES = (
    "D008_fold_manifest.csv",
    "D008_preprocessing_parameters.csv",
    "D008_model_spec.json",
    "D008_metrics.csv",
    "D008_incremental_log_loss.csv",
    "D008_summary.json",
    "D008_confusion_matrices.json",
    "D008_majority_baseline.csv",
    "D008_metadata.json",
)


def assert_fresh_artifacts() -> None:
    """Prevent an unapproved overwrite of frozen D008 result artifacts."""
    existing = [name for name in ARTIFACT_NAMES if (RESULTS_DIR / name).exists()]
    if existing:
        raise FileExistsError(f"D008 refuses to overwrite existing result artifacts: {existing}")


def manifest_row(fold: int, subset: str, features: pd.DataFrame, target: pd.Series) -> dict[str, int | str]:
    """Persist frozen population membership and class counts."""
    validate_expected_fold_count(fold, subset, len(target))
    counts = target.value_counts().reindex([-1, 0, 1], fill_value=0)
    return {
        "fold": fold,
        "subset": subset,
        "n_rows": len(target),
        "class_minus": int(counts.loc[-1]),
        "class_zero": int(counts.loc[0]),
        "class_plus": int(counts.loc[1]),
        "n_unique_days": int(features["day"].nunique()),
        "n_unique_equities": int(features["equity"].nunique()),
    }


def main() -> None:
    assert_fresh_artifacts()
    features, reod = load_physical_discovery_data()
    manifests: list[dict[str, int | str]] = []
    parameter_rows: list[dict[str, int | float]] = []
    metric_rows: list[dict[str, int | float | str]] = []
    delta_rows: list[dict[str, int | float]] = []
    baseline_rows: list[dict[str, int | float]] = []
    confusion: dict[str, object] = {"labels": [-1, 0, 1], "matrices": {}}

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        assert_ternary_target(subsets.fit_reod, expected_index=subsets.fit_features.index, context=f"D008 fit {fold.fold}")
        assert_ternary_target(subsets.validation_reod, expected_index=subsets.validation_features.index, context=f"D008 validation {fold.fold}")
        manifests.extend(
            [
                manifest_row(fold.fold, "fit", subsets.fit_features, subsets.fit_reod),
                manifest_row(fold.fold, "validation", subsets.validation_features, subsets.validation_reod),
            ]
        )

        fit_structure, transformer = fit_intensity_transformer(subsets.fit_features)
        validation_structure = build_recent_window_structure(subsets.validation_features)
        parameter_rows.append(
            {
                "fold": fold.fold,
                "fit_intensity_median": float(transformer.fit_intensity_median_),
                "scaler_mean": float(transformer.scaler_.mean_[0]),
                "scaler_scale": float(transformer.scaler_.scale_[0]),
                "n_fit_defined_intensity": int(fit_structure["N_obs_w"].gt(0).sum()),
                "n_fit_undefined_intensity": int(fit_structure["N_obs_w"].eq(0).sum()),
            }
        )
        fit_intensity_z = transformer.transform(fit_structure["raw_I_recent"])
        validation_intensity_z = transformer.transform(validation_structure["raw_I_recent"])
        fit_matrices = build_d008_representations(subsets.fit_features, fit_intensity_z)
        validation_matrices = build_d008_representations(subsets.validation_features, validation_intensity_z)
        assert_identical_d008_rows(fit_matrices, subsets.fit_features, subsets.fit_reod)
        assert_identical_d008_rows(validation_matrices, subsets.validation_features, subsets.validation_reod)
        if not fit_matrices["C"].loc[:, ["I_recent_z"]].equals(fit_matrices["P"].loc[:, ["I_recent_z"]]):
            raise AssertionError("D008 Fit I_recent_z differs between C and P.")
        if not validation_matrices["C"].loc[:, ["I_recent_z"]].equals(validation_matrices["P"].loc[:, ["I_recent_z"]]):
            raise AssertionError("D008 Validation I_recent_z differs between C and P.")
        if tuple(fit_matrices["C"].columns) != C_COLUMNS or tuple(fit_matrices["P"].columns) != P_COLUMNS:
            raise AssertionError("D008 Fit predictor schema differs from the frozen contract.")

        losses: dict[str, float] = {}
        for representation in ("C", "P"):
            model = make_d007_classifier().fit(
                fit_matrices[representation], encode_ternary_target(subsets.fit_reod)
            )
            probabilities = ternary_probabilities(model, validation_matrices[representation])
            metrics, matrix = evaluate_d007_probabilities(subsets.validation_reod, probabilities)
            metric_rows.append(
                {
                    "fold": fold.fold,
                    "representation": representation,
                    "n_rows": len(subsets.validation_reod),
                    **metrics,
                }
            )
            confusion["matrices"][f"{representation}_fold_{fold.fold}_validation"] = matrix
            losses[representation] = metrics["multiclass_log_loss"]

        delta_rows.append(
            {
                "fold": fold.fold,
                "C_log_loss": losses["C"],
                "P_log_loss": losses["P"],
                "delta_log_loss": losses["C"] - losses["P"],
            }
        )
        selected, metrics, matrix = evaluate_majority_baseline(subsets.fit_reod, subsets.validation_reod)
        baseline_rows.append({"fold": fold.fold, "fit_majority_class": selected, "n_rows": len(subsets.validation_reod), **metrics})
        confusion["matrices"][f"majority_fold_{fold.fold}_validation"] = matrix

    delta_frame = pd.DataFrame(delta_rows).sort_values("fold")
    summary = d008_primary_screen(delta_frame["delta_log_loss"].tolist())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifests).sort_values(["fold", "subset"]).to_csv(RESULTS_DIR / "D008_fold_manifest.csv", index=False)
    pd.DataFrame(parameter_rows).sort_values("fold").to_csv(RESULTS_DIR / "D008_preprocessing_parameters.csv", index=False)
    (RESULTS_DIR / "D008_model_spec.json").write_text(json.dumps(dict(XGBOOST_PARAMS), indent=2), encoding="utf-8")
    pd.DataFrame(metric_rows).sort_values(["representation", "fold"]).to_csv(RESULTS_DIR / "D008_metrics.csv", index=False)
    delta_frame.to_csv(RESULTS_DIR / "D008_incremental_log_loss.csv", index=False)
    (RESULTS_DIR / "D008_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "D008_confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    pd.DataFrame(baseline_rows).sort_values("fold").to_csv(RESULTS_DIR / "D008_majority_baseline.csv", index=False)
    (RESULTS_DIR / "D008_metadata.json").write_text(
        json.dumps(
            {
                "experiment": "D008",
                "scope": "physical Discovery days 0-352 x E_dev",
                "return_window": [f"r{i}" for i in range(41, 53)],
                "xgboost": xgboost.__version__,
                "python": platform.python_version(),
                "competition_test_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("D008 completed only after separate real-execution authorization.")


if __name__ == "__main__":
    main()

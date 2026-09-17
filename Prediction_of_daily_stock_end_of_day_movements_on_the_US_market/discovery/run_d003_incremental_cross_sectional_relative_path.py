"""Execute frozen D003 only after separate Discovery execution approval."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import (  # noqa: E402
    DISCOVERY_FOLDS,
    build_d003_matrices,
    build_relative_returns,
    descriptive_candidate_rule,
    evaluate_d003_model,
    make_positional_logistic_regression,
    prepare_directional_subset,
    split_discovery_fold,
    assert_e_dev_only,
    validate_discovery_features,
)
from src.exp003_conditional_directional_sign import assert_both_sign_classes  # noqa: E402
from src.exp005_positional_path_signal import PositionalPathTransformer  # noqa: E402


DISCOVERY_DATA_DIR = PROJECT_ROOT / "data" / "discovery"
INPUT_PATH = DISCOVERY_DATA_DIR / "discovery_input_training.csv"
OUTPUT_PATH = DISCOVERY_DATA_DIR / "discovery_output_training.csv"
MANIFEST_PATH = DISCOVERY_DATA_DIR / "discovery_partition_manifest.json"
PARTITION_PATH = PROJECT_ROOT / "experiments" / "EXP_000_equity_partition.csv"
RESULTS_DIR = PROJECT_ROOT / "discovery" / "results"


def load_physical_discovery_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the authorized physical Discovery input and aligned labels only."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    features = pd.read_csv(INPUT_PATH)
    labels = pd.read_csv(OUTPUT_PATH)
    validate_discovery_features(features)
    assert_e_dev_only(features, pd.read_csv(PARTITION_PATH))
    if len(features) != int(manifest["row_count"]) or features["ID"].nunique() != len(features):
        raise AssertionError("D003 physical Discovery input disagrees with its manifest.")
    if set(labels.columns) != {"ID", "reod"} or labels["ID"].duplicated().any():
        raise AssertionError("D003 physical Discovery labels have an unexpected schema.")
    aligned = labels.set_index("ID").reindex(features["ID"])
    if aligned["reod"].isna().any():
        raise AssertionError("D003 labels do not align to physical Discovery IDs.")
    reod = pd.Series(aligned["reod"].to_numpy(), index=features.index, name="reod")
    if not set(reod.unique()).issubset({-1, 0, 1}):
        raise AssertionError("D003 labels must be ternary before directional conditioning.")
    return features, reod


def main() -> None:
    features, reod = load_physical_discovery_data()
    # This target-free transformation precedes all directional conditioning.
    relative_all = build_relative_returns(features)

    manifest_rows: list[dict[str, int | str]] = []
    parameter_rows: list[dict[str, int | float | str | bool]] = []
    metric_rows: list[dict[str, int | float | str]] = []
    delta_rows: list[dict[str, int | float]] = []
    confusion: dict[str, object] = {"labels": [0, 1], "matrices": {}}

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        fit_own, fit_relative, fit_target, fit_manifest = prepare_directional_subset(
            subsets.fit_features, subsets.fit_reod, relative_all
        )
        validation_own, validation_relative, validation_target, validation_manifest = prepare_directional_subset(
            subsets.validation_features, subsets.validation_reod, relative_all
        )
        assert_both_sign_classes(fit_target, context=f"D003 fit fold {fold.fold}")
        assert_both_sign_classes(validation_target, context=f"D003 validation fold {fold.fold}")
        manifest_rows.extend([
            {"fold": fold.fold, "subset": "fit", **fit_manifest},
            {"fold": fold.fold, "subset": "validation", **validation_manifest},
        ])

        own_transformer = PositionalPathTransformer().fit(fit_own)
        relative_transformer = PositionalPathTransformer().fit(fit_relative)
        for family, transformer in (("own", own_transformer), ("relative", relative_transformer)):
            for row in transformer.parameter_rows(fold=fold.fold):
                parameter_rows.append({"family": family, **row})

        fit_matrices = build_d003_matrices(own_transformer, relative_transformer, fit_own, fit_relative, fit_target)
        validation_matrices = build_d003_matrices(
            own_transformer, relative_transformer, validation_own, validation_relative, validation_target
        )
        fold_metrics: dict[str, dict[str, float]] = {}
        for condition in ("A", "B"):
            model = make_positional_logistic_regression().fit(fit_matrices[condition], fit_target)
            if tuple(model.classes_) != (0, 1):
                raise AssertionError("D003 Logistic Regression must retain sign classes [0, 1].")
            metrics, matrix = evaluate_d003_model(validation_target, validation_matrices[condition], model)
            fold_metrics[condition] = metrics
            metric_rows.append({
                "condition": condition, "fold": fold.fold, "subset": "validation",
                "n_rows": int(len(validation_target)), **metrics,
            })
            confusion["matrices"][f"{condition}_fold_{fold.fold}_validation"] = matrix
        delta_rows.append({
            "fold": fold.fold, "auc_a": fold_metrics["A"]["roc_auc"],
            "auc_b": fold_metrics["B"]["roc_auc"],
            "delta_auc_b_minus_a": float(fold_metrics["B"]["roc_auc"] - fold_metrics["A"]["roc_auc"]),
        })

    metric_frame = pd.DataFrame(metric_rows).sort_values(["condition", "fold"])
    delta_frame = pd.DataFrame(delta_rows).sort_values("fold")
    summary = metric_frame.groupby("condition")[["roc_auc", "balanced_accuracy", "accuracy", "recall_negative", "recall_positive"]].agg(["mean", lambda x: x.std(ddof=1)])
    summary.columns = [f"{metric}_{stat}".replace("<lambda_0>", "std_ddof_1") for metric, stat in summary.columns]
    decision = descriptive_candidate_rule(delta_frame["delta_auc_b_minus_a"].tolist())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(RESULTS_DIR / "D003_fold_manifest.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(RESULTS_DIR / "D003_preprocessing_parameters.csv", index=False)
    metric_frame.to_csv(RESULTS_DIR / "D003_metrics.csv", index=False)
    delta_frame.to_csv(RESULTS_DIR / "D003_incremental_auc.csv", index=False)
    summary.reset_index().to_csv(RESULTS_DIR / "D003_summary.csv", index=False)
    pd.DataFrame([decision]).to_csv(RESULTS_DIR / "D003_candidate_rule.csv", index=False)
    with (RESULTS_DIR / "D003_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump(confusion, handle, indent=2)
    metadata = {
        "discovery": "D003", "physical_input_only": True, "days": "0-352", "equity_partition": "E_dev",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "competition_test_accessed": False,
    }
    with (RESULTS_DIR / "D003_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("D003 completed using only the physical Discovery partition; competition test was not accessed.")


if __name__ == "__main__":
    main()

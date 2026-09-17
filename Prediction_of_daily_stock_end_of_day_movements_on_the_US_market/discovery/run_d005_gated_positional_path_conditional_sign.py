"""Execute frozen D005 only after separate real-execution authorization."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import split_discovery_fold  # noqa: E402
from discovery.d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    build_d004_matrix,
    directional_probability,
    fit_directional_priors,
    fit_intensity_transformer,
)
from discovery.d005_gated_positional_path_conditional_sign import (  # noqa: E402
    DISCOVERY_FOLDS,
    PositionalPathTransformer,
    build_d005_path_matrix,
    descriptive_candidate_screen,
    evaluate_d005_sign_model,
    make_positional_logistic_regression,
    prepare_gated_directional_rows,
    sign_majority_class,
)
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    RESULTS_DIR,
    load_physical_discovery_data,
)
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402
from src.exp003_conditional_directional_sign import assert_both_sign_classes, evaluate_sign_majority_baseline  # noqa: E402


def _manifest_record(fold: int, subset: str, rows, source_features: pd.DataFrame, priors) -> dict[str, int | float | str]:
    target = rows.sign_target
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    retained_source = source_features.loc[rows.raw_returns.index]
    if not retained_source.index.equals(rows.raw_returns.index):
        raise AssertionError("D005 retained identifiers changed row order.")
    return {
        "fold": fold, "subset": subset, **rows.manifest,
        "n_negative_retained": int(counts.loc[0]), "n_positive_retained": int(counts.loc[1]),
        "fraction_negative_retained": float(counts.loc[0] / len(target)) if len(target) else np.nan,
        "fraction_positive_retained": float(counts.loc[1] / len(target)) if len(target) else np.nan,
        "n_unique_days_retained": int(retained_source["day"].nunique()),
        "n_unique_equities_retained": int(retained_source["equity"].nunique()),
        "pi_minus": priors.pi_minus, "pi_plus": priors.pi_plus,
        "pi_max": priors.pi_max, "p_d_star": priors.threshold,
    }


def main() -> None:
    """Run one complete frozen D005 procedure on physical Discovery only."""
    features, reod = load_physical_discovery_data()
    manifest_rows: list[dict[str, int | float | str]] = []
    metric_rows: list[dict[str, int | float | str]] = []
    parameter_rows: list[dict[str, int | float | str | bool]] = []
    confusion: dict[str, object] = {"labels": [0, 1], "matrices": {}}
    validation_aucs: list[float] = []

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        _, gate_transformer = fit_intensity_transformer(subsets.fit_features)
        fit_gate_matrix, _ = build_d004_matrix(subsets.fit_features, subsets.fit_reod, gate_transformer)
        validation_gate_matrix, _ = build_d004_matrix(subsets.validation_features, subsets.validation_reod, gate_transformer)
        gate_model = make_binary_logistic_regression().fit(fit_gate_matrix, build_binary_target(subsets.fit_reod))
        priors = fit_directional_priors(subsets.fit_reod)
        fit_rows = prepare_gated_directional_rows(
            subsets.fit_features, subsets.fit_reod, directional_probability(gate_model, fit_gate_matrix), priors
        )
        validation_rows = prepare_gated_directional_rows(
            subsets.validation_features, subsets.validation_reod,
            directional_probability(gate_model, validation_gate_matrix), priors,
        )
        assert_both_sign_classes(fit_rows.sign_target, context="D005 Fit")
        assert_both_sign_classes(validation_rows.sign_target, context="D005 Validation")
        transformer = PositionalPathTransformer().fit(fit_rows.raw_returns)
        fit_matrix = build_d005_path_matrix(transformer, fit_rows)
        validation_matrix = build_d005_path_matrix(transformer, validation_rows)
        classifier = make_positional_logistic_regression().fit(fit_matrix, fit_rows.sign_target)
        metrics, matrix = evaluate_d005_sign_model(validation_rows.sign_target, validation_matrix, classifier)
        baseline_class = sign_majority_class(fit_rows.sign_target)
        baseline_metrics, baseline_matrix = evaluate_sign_majority_baseline(validation_rows.sign_target, baseline_class)
        validation_aucs.append(metrics["roc_auc"])
        manifest_rows.extend((
            _manifest_record(fold.fold, "fit", fit_rows, subsets.fit_features, priors),
            _manifest_record(fold.fold, "validation", validation_rows, subsets.validation_features, priors),
        ))
        metric_rows.extend((
            {"fold": fold.fold, "condition": "D005_path_mask", "n_rows": len(validation_rows.sign_target), **metrics},
            {"fold": fold.fold, "condition": "fit_majority_sign_baseline", "n_rows": len(validation_rows.sign_target), "baseline_class_from_fit": baseline_class, **baseline_metrics},
        ))
        parameter_rows.extend(transformer.parameter_rows(fold=fold.fold))
        confusion["matrices"][f"D005_path_mask_fold_{fold.fold}_validation"] = matrix
        confusion["matrices"][f"fit_majority_sign_baseline_fold_{fold.fold}_validation"] = baseline_matrix

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(RESULTS_DIR / "D005_fold_manifest.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(RESULTS_DIR / "D005_preprocessing_parameters.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(RESULTS_DIR / "D005_metrics.csv", index=False)
    pd.DataFrame([descriptive_candidate_screen(validation_aucs)]).to_csv(RESULTS_DIR / "D005_summary.csv", index=False)
    (RESULTS_DIR / "D005_confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    metadata = {
        "experiment": "D005", "scope": "physical Discovery days 0-352 x E_dev",
        "gate": "exact D004, p_D > p_D_star", "competition_test_accessed": False,
        "protected_days_353_502_accessed": False, "e_holdout_accessed": False,
    }
    (RESULTS_DIR / "D005_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("D005 completed using only physical Discovery data after separate execution authorization.")


if __name__ == "__main__":
    main()

"""Execute I007A only after separate authorization; never runs on import."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS, split_discovery_fold  # noqa: E402
from discovery.d007_raw_path_masks_ternary_xgboost import build_d007_representations, encode_ternary_target, make_d007_classifier  # noqa: E402
from discovery.i007_frozen_d007_model_interpretation import (  # noqa: E402
    GLOBAL_SHAP_SAMPLE_MAX,
    GLOBAL_SHAP_SEED,
    RECONSTRUCTION_LOG_LOSS_ATOL,
    SHAP_ADDITIVITY_ATOL,
    SHAP_ADDITIVITY_RTOL,
    additivity_record,
    allocation_by_fold,
    assert_reconstruction_matches,
    deterministic_validation_positions,
    explain_and_validate_additivity,
    expected_d007_r_plus_m_log_loss,
    orientation_diagnostics,
    rank_stability,
    reconstruction_log_loss,
    sampled_validation_matrix,
    stable_raw_features,
)
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import RESULTS_DIR, load_physical_discovery_data  # noqa: E402


METRICS_PATH = RESULTS_DIR / "D007_metrics.csv"
ARTIFACT_NAMES = (
    "I007A_reproducibility_gate.csv", "I007A_sample_manifest.csv", "I007A_additivity.csv",
    "I007A_allocation.csv", "I007A_feature_stability.csv", "I007A_rank_correlations.csv",
    "I007A_top10.csv", "I007A_orientation_spearman.csv", "I007A_orientation_bins.csv",
    "I007A_metadata.json",
)


def assert_fresh_artifacts() -> None:
    existing = [name for name in ARTIFACT_NAMES if (RESULTS_DIR / name).exists()]
    if existing:
        raise FileExistsError(f"I007A refuses to overwrite result artifacts: {existing}")


def main() -> None:
    """Reconstruct every D007 R+M model, pass gates, then produce SHAP summaries."""
    started = time.perf_counter()
    assert_fresh_artifacts()
    if not METRICS_PATH.is_file():
        raise FileNotFoundError("I007A requires persisted D007 metrics for its reconstruction gate.")
    references = expected_d007_r_plus_m_log_loss(pd.read_csv(METRICS_PATH))
    features, reod = load_physical_discovery_data()
    reconstructed: dict[int, tuple[object, pd.DataFrame, pd.DataFrame]] = {}
    gate_records: list[dict[str, float | int | bool]] = []

    # All reconstruction checks complete before the first SHAP call.
    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        fit_matrix = build_d007_representations(subsets.fit_features, subsets.fit_reod)["R+M"]
        validation_matrix = build_d007_representations(subsets.validation_features, subsets.validation_reod)["R+M"]
        model = make_d007_classifier().fit(fit_matrix, encode_ternary_target(subsets.fit_reod))
        reconstructed_loss = reconstruction_log_loss(subsets.validation_reod, model, validation_matrix)
        gate_records.append(assert_reconstruction_matches(fold=fold.fold, reconstructed=reconstructed_loss, expected=references[fold.fold]))
        reconstructed[fold.fold] = (model, validation_matrix, subsets.validation_features)

    allocation_tables: list[pd.DataFrame] = []
    sample_tables: list[pd.DataFrame] = []
    additivity_tables: list[dict[str, float | int | bool]] = []
    orientation_correlations: list[pd.DataFrame] = []
    orientation_bins: list[pd.DataFrame] = []
    sampled_by_fold: dict[int, tuple[pd.DataFrame, np.ndarray, pd.DataFrame]] = {}
    for fold in DISCOVERY_FOLDS:
        model, validation_matrix, validation_features = reconstructed[fold.fold]
        sample_matrix, manifest = sampled_validation_matrix(validation_matrix, validation_features, fold=fold.fold)
        values, base_values, margins = explain_and_validate_additivity(model, sample_matrix)
        sample_tables.append(manifest)
        allocation_tables.append(allocation_by_fold(values, fold=fold.fold))
        additivity_tables.append(additivity_record(fold=fold.fold, values=values, base_values=base_values, margins=margins))
        sampled_by_fold[fold.fold] = (sample_matrix, values, manifest)

    allocation = pd.concat(allocation_tables, ignore_index=True)
    ranked, rank_correlations, top10 = rank_stability(allocation)
    summary = ranked.groupby(["output_index", "output_label", "feature"], as_index=False).agg(
        mean_abs_shap_macro=("mean_abs_shap", "mean"), median_rank=("rank_within_fold", "median"),
        min_rank=("rank_within_fold", "min"), max_rank=("rank_within_fold", "max"),
        top10_membership_count=("top10_membership_count", "max"), stable_top10=("stable_top10", "max"),
    )
    for fold in DISCOVERY_FOLDS:
        sample_matrix, values, manifest = sampled_by_fold[fold.fold]
        for output_label in (-1, 1):
            selected = stable_raw_features(summary, output_label=output_label)
            correlations, bins = orientation_diagnostics(sample_matrix, values, manifest, fold=fold.fold, output_label=output_label, features=selected)
            orientation_correlations.append(correlations)
            orientation_bins.append(bins)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gate_records).to_csv(RESULTS_DIR / "I007A_reproducibility_gate.csv", index=False)
    pd.concat(sample_tables, ignore_index=True).to_csv(RESULTS_DIR / "I007A_sample_manifest.csv", index=False)
    pd.DataFrame(additivity_tables).to_csv(RESULTS_DIR / "I007A_additivity.csv", index=False)
    allocation.to_csv(RESULTS_DIR / "I007A_allocation.csv", index=False)
    ranked.merge(summary, on=["output_index", "output_label", "feature", "median_rank", "min_rank", "max_rank", "top10_membership_count", "stable_top10"], how="left").to_csv(RESULTS_DIR / "I007A_feature_stability.csv", index=False)
    rank_correlations.to_csv(RESULTS_DIR / "I007A_rank_correlations.csv", index=False)
    top10.to_csv(RESULTS_DIR / "I007A_top10.csv", index=False)
    correlation_table = pd.concat(orientation_correlations, ignore_index=True) if orientation_correlations else pd.DataFrame(
        columns=["fold", "output_label", "feature", "n_observed", "spearman_value_vs_same_output_shap", "status"]
    )
    bin_table = pd.concat(orientation_bins, ignore_index=True) if orientation_bins else pd.DataFrame(
        columns=["fold", "output_label", "feature", "bin", "n_bins", "count", "raw_min", "raw_max", "raw_mean", "mean_same_output_shap"]
    )
    correlation_table.to_csv(RESULTS_DIR / "I007A_orientation_spearman.csv", index=False)
    bin_table.to_csv(RESULTS_DIR / "I007A_orientation_bins.csv", index=False)
    metadata = {
        "interpretation": "I007A frozen D007 main-effect TreeSHAP only",
        "scope": "physical Discovery days 0-352 x E_dev", "model_reconstructed": "D007 R+M only",
        "reconstruction_log_loss_atol": RECONSTRUCTION_LOG_LOSS_ATOL,
        "sampling": {"maximum_rows_per_fold": GLOBAL_SHAP_SAMPLE_MAX, "seed": GLOBAL_SHAP_SEED, "restore_source_order": True},
        "shap": {"version": shap.__version__, "feature_perturbation": "tree_path_dependent", "model_output": "raw", "additivity_rtol": SHAP_ADDITIVITY_RTOL, "additivity_atol": SHAP_ADDITIVITY_ATOL, "interactions_computed": False},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "internal_confirmation_accessed": False, "e_holdout_accessed": False, "competition_test_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS_DIR / "I007A_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("I007A completed only after exact D007 reconstruction and raw-margin additivity gates.")


if __name__ == "__main__":
    main()

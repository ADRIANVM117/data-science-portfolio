"""Execute I008A only after a separate real-reconstruction authorization."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS, split_discovery_fold  # noqa: E402
from discovery.d007_raw_path_masks_ternary_xgboost import encode_ternary_target, make_d007_classifier  # noqa: E402
from discovery.d008_raw_path_beyond_availability_intensity import (  # noqa: E402
    assert_identical_d008_rows, build_d008_representations, fit_intensity_transformer,
)
from discovery.i008a_frozen_d008_probability_allocation_diagnostic import (  # noqa: E402
    ARTIFACT_NAMES, assert_fresh_artifacts, assert_gain_conservation, gate_record,
    level1_summary, level2_summary, persisted_d008_log_losses, prediction_transitions,
    probabilities_with_metadata, probability_changes, reconstruction_log_loss, realized_gain,
    validate_fit_preprocessing,
)
from discovery.run_d004_hierarchical_recent_intensity_ternary_decision import RESULTS_DIR, load_physical_discovery_data  # noqa: E402
from src.exp008_recent_movement_intensity import build_recent_window_structure  # noqa: E402


D008_METRICS_PATH = RESULTS_DIR / "D008_metrics.csv"
D008_PREPROCESSING_PATH = RESULTS_DIR / "D008_preprocessing_parameters.csv"


def main() -> None:
    """Reconstruct D008, pass all gates, then write only frozen I008A summaries."""
    started = time.perf_counter()
    assert_fresh_artifacts(RESULTS_DIR)
    if not D008_METRICS_PATH.is_file() or not D008_PREPROCESSING_PATH.is_file():
        raise FileNotFoundError("I008A requires persisted D008 metrics and preprocessing artifacts.")
    references = persisted_d008_log_losses(pd.read_csv(D008_METRICS_PATH))
    persisted_preprocessing = pd.read_csv(D008_PREPROCESSING_PATH)
    features, reod = load_physical_discovery_data()

    # Everything stays in memory until every reconstructed C/P arm passes its gate.
    gate_rows: list[dict[str, object]] = []
    level1_rows: list[dict[str, object]] = []
    level2_tables: list[pd.DataFrame] = []
    transition_tables: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []

    for fold in DISCOVERY_FOLDS:
        subsets = split_discovery_fold(features, reod, fold)
        fit_structure, transformer = fit_intensity_transformer(subsets.fit_features)
        validate_fit_preprocessing(fold.fold, fit_structure, transformer, persisted_preprocessing)
        validation_structure = build_recent_window_structure(subsets.validation_features)
        fit_matrices = build_d008_representations(subsets.fit_features, transformer.transform(fit_structure["raw_I_recent"]))
        validation_matrices = build_d008_representations(subsets.validation_features, transformer.transform(validation_structure["raw_I_recent"]))
        assert_identical_d008_rows(fit_matrices, subsets.fit_features, subsets.fit_reod)
        assert_identical_d008_rows(validation_matrices, subsets.validation_features, subsets.validation_reod)

        probabilities: dict[str, pd.DataFrame] = {}
        losses: dict[str, float] = {}
        for arm in ("C", "P"):
            model = make_d007_classifier().fit(fit_matrices[arm], encode_ternary_target(subsets.fit_reod))
            probability, dtype, row_sum_tolerance = probabilities_with_metadata(model, validation_matrices[arm])
            loss = reconstruction_log_loss(subsets.validation_reod, probability)
            gate_rows.append(gate_record(
                fold=fold.fold, arm=arm, target=subsets.validation_reod,
                features=subsets.validation_features, matrices=validation_matrices,
                probabilities=probability, raw_probability_dtype=dtype,
                row_sum_tolerance=row_sum_tolerance, reconstructed_loss=loss,
                persisted_loss=references[(fold.fold, arm)],
            ))
            probabilities[arm], losses[arm] = probability, loss

        changes = probability_changes(probabilities["C"], probabilities["P"])
        gain = realized_gain(subsets.validation_reod, probabilities["C"], probabilities["P"])
        assert_gain_conservation(
            gain,
            persisted_delta=references[(fold.fold, "C")] - references[(fold.fold, "P")],
            reconstructed_delta=losses["C"] - losses["P"],
        )
        level1_rows.append(level1_summary(fold.fold, changes))
        level2_tables.append(level2_summary(fold.fold, subsets.validation_reod, changes, gain))
        transition_tables.append(prediction_transitions(fold.fold, subsets.validation_reod, probabilities["C"], probabilities["P"]))
        fold_metadata.append({"fold": fold.fold, "reconstructed_delta_log_loss": losses["C"] - losses["P"], "gain_conservation_atol": 1e-10})

    # No result artifact is created before all folds, arms, gates, and identities pass.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(gate_rows).sort_values(["fold", "arm"]).to_csv(RESULTS_DIR / "I008A_reproduction_gate.csv", index=False)
    pd.DataFrame(level1_rows).sort_values("fold").to_csv(RESULTS_DIR / "I008A_probability_movement.csv", index=False)
    pd.concat(level2_tables, ignore_index=True).sort_values(["fold", "true_class"]).to_csv(RESULTS_DIR / "I008A_realized_class_gain.csv", index=False)
    pd.concat(transition_tables, ignore_index=True).sort_values(["fold", "transition_type", "true_class", "C_prediction", "P_prediction"], na_position="first").to_csv(RESULTS_DIR / "I008A_prediction_transitions.csv", index=False)
    metadata = {
        "interpretation": "I008A frozen D008 probability-allocation diagnostic",
        "scope": "physical Discovery days 0-352 x E_dev",
        "model_reconstructed": "D008 C and P, three folds each",
        "artifact_names": list(ARTIFACT_NAMES),
        "reproduction_gate": {"log_loss_atol": 1e-8, "rtol": 0.0, "all_six_required": True},
        "gain_conservation": {"atol": 1e-10, "rtol": 0.0},
        "folds": fold_metadata,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "internal_confirmation_accessed": False, "e_holdout_accessed": False,
        "competition_test_accessed": False, "runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS_DIR / "I008A_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("I008A completed only after all six exact D008 reproduction gates passed.")


if __name__ == "__main__":
    main()

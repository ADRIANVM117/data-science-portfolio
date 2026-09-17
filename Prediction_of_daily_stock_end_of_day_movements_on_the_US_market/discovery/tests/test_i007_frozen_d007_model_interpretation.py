"""Synthetic I007A tests; never load physical Discovery or real D007 rows."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d007_raw_path_masks_ternary_xgboost import (  # noqa: E402
    RM_COLUMNS, build_d007_representations, encode_ternary_target, make_d007_classifier,
)
from discovery.i007_frozen_d007_model_interpretation import (  # noqa: E402
    GLOBAL_SHAP_SAMPLE_MAX, OUTPUT_LABELS, RECONSTRUCTION_LOG_LOSS_ATOL,
    SHAP_ADDITIVITY_ATOL, SHAP_ADDITIVITY_RTOL, TOP_K, additivity_record,
    allocation_by_fold, assert_reconstruction_matches, deterministic_validation_positions,
    explain_and_validate_additivity, expected_d007_r_plus_m_log_loss,
    orientation_diagnostics, rank_stability, sampled_validation_matrix, stable_raw_features,
)
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert phrase in str(error)
        return
    raise AssertionError("Expected an integrity AssertionError.")


def synthetic_features(n: int = 45) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(18)
    frame = pd.DataFrame(rng.normal(size=(n, 53)), columns=RETURN_COLUMNS)
    frame.loc[::7, "r3"] = np.nan
    frame.loc[::9, "r47"] = np.nan
    frame.insert(0, "equity", np.arange(n) % 6)
    frame.insert(0, "day", np.arange(n) % 8)
    frame.insert(0, "ID", np.arange(1000, 1000 + n))
    target = pd.Series(np.resize(np.array([-1, 0, 1]), n), index=frame.index, name="reod")
    return frame, target


def fitted_synthetic_model() -> tuple[object, pd.DataFrame, pd.DataFrame]:
    features, target = synthetic_features()
    matrix = build_d007_representations(features, target)["R+M"]
    model = make_d007_classifier().fit(matrix, encode_ternary_target(target))
    return model, matrix, features


def test_reference_gate_and_deterministic_target_blind_sample() -> None:
    metrics = pd.DataFrame({"fold": [1, 1, 2, 2, 3, 3], "representation": ["M", "R+M"] * 3, "multiclass_log_loss": [1.1, 1.01, 1.2, 1.02, 1.3, 1.03]})
    assert expected_d007_r_plus_m_log_loss(metrics) == {1: 1.01, 2: 1.02, 3: 1.03}
    record = assert_reconstruction_matches(fold=1, reconstructed=1.01 + RECONSTRUCTION_LOG_LOSS_ATOL, expected=1.01)
    assert record["passed"]
    assert_raises(lambda: assert_reconstruction_matches(fold=1, reconstructed=1.01 + 1.1 * RECONSTRUCTION_LOG_LOSS_ATOL, expected=1.01), "mismatch")
    positions_a = deterministic_validation_positions(2_300, fold=2)
    positions_b = deterministic_validation_positions(2_300, fold=2)
    assert len(positions_a) == GLOBAL_SHAP_SAMPLE_MAX and np.array_equal(positions_a, positions_b)
    assert np.all(np.diff(positions_a) > 0) and deterministic_validation_positions(10, fold=1).tolist() == list(range(10))


def test_sample_manifest_and_schema_are_target_free() -> None:
    _, matrix, features = fitted_synthetic_model()
    sample, manifest = sampled_validation_matrix(matrix, features, fold=1)
    assert tuple(sample.columns) == RM_COLUMNS and len(sample) == len(matrix)
    assert manifest.columns.tolist() == ["fold", "source_row_position", "source_index", "ID"]
    assert "reod" not in manifest and sample.index.tolist() == manifest["source_index"].tolist()
    assert np.array_equal(manifest["source_row_position"].to_numpy(), np.arange(len(matrix)))


def test_synthetic_multiclass_shap_shape_and_raw_margin_additivity() -> None:
    model, matrix, _ = fitted_synthetic_model()
    values, base_values, margins = explain_and_validate_additivity(model, matrix.iloc[:12])
    assert values.shape == (12, 106, 3) and base_values.shape == margins.shape == (12, 3)
    assert np.allclose(base_values + values.sum(axis=1), margins, rtol=SHAP_ADDITIVITY_RTOL, atol=SHAP_ADDITIVITY_ATOL)
    record = additivity_record(fold=1, values=values, base_values=base_values, margins=margins)
    assert record["passed"] and record["n_features"] == 106 and record["n_outputs"] == 3
    assert_raises(lambda: explain_and_validate_additivity(model, matrix.loc[:, tuple(reversed(RM_COLUMNS))]), "schema")


def test_allocation_rank_stability_and_top10_membership() -> None:
    rng = np.random.default_rng(4)
    allocations = []
    for fold in (1, 2, 3):
        values = rng.normal(size=(14, 106, 3))
        values[:, 0, 0] += 9.0; values[:, 1, 2] += 8.0
        allocations.append(allocation_by_fold(values, fold=fold))
    ranked, correlations, top = rank_stability(pd.concat(allocations, ignore_index=True))
    assert len(ranked) == 3 * 3 * 106 and len(correlations) == 3 * 3
    assert set(ranked["rank_within_fold"]) >= {1.0, float(TOP_K)}
    assert (ranked.groupby(["fold", "output_index"])["rank_within_fold"].max() == 106).all()
    assert (ranked.loc[ranked.rank_within_fold.le(TOP_K)].groupby(["fold", "output_index"]).size() == TOP_K).all()
    assert {"pairwise_overlap", "top10_membership"}.issubset(set(top["record_type"]))
    summary = ranked.groupby(["output_index", "output_label", "feature"], as_index=False).agg(
        stable_top10=("stable_top10", "max"), top10_membership_count=("top10_membership_count", "max")
    )
    selected = stable_raw_features(summary, output_label=-1)
    assert all(feature.startswith("r") for feature in selected)


def test_observed_only_orientation_uses_same_output_and_equal_count_bins() -> None:
    _, matrix, features = fitted_synthetic_model()
    sample, manifest = sampled_validation_matrix(matrix, features, fold=1)
    values = np.zeros((len(sample), 106, 3), dtype=float)
    values[:, 0, 0] = np.nan_to_num(sample["r0"].to_numpy(), nan=0.0)
    correlations, bins = orientation_diagnostics(sample, values, manifest, fold=1, output_label=-1, features=("r0",))
    assert correlations.iloc[0]["n_observed"] == int(sample["r0"].notna().sum())
    assert correlations.iloc[0]["status"] == "defined"
    assert bins["n_bins"].nunique() == 1 and bins["n_bins"].iloc[0] == 10
    assert bins["count"].sum() == correlations.iloc[0]["n_observed"]
    assert_raises(lambda: orientation_diagnostics(sample, values, manifest, fold=1, output_label=-1, features=("m0",)), "raw returns only")


def test_runner_is_inert_and_excludes_interactions_and_protected_sources() -> None:
    source = (PROJECT_ROOT / "discovery" / "run_i007_frozen_d007_model_interpretation.py").read_text(encoding="utf-8")
    assert "shap_interaction" not in source and "input_test" not in source and "output_test" not in source
    assert "if __name__ == \"__main__\":" in source
    assert not any((PROJECT_ROOT / "discovery" / "results").glob("I007A_*"))


if __name__ == "__main__":
    tests = (
        test_reference_gate_and_deterministic_target_blind_sample,
        test_sample_manifest_and_schema_are_target_free,
        test_synthetic_multiclass_shap_shape_and_raw_margin_additivity,
        test_allocation_rank_stability_and_top10_membership,
        test_observed_only_orientation_uses_same_output_and_equal_count_bins,
        test_runner_is_inert_and_excludes_interactions_and_protected_sources,
    )
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests))
    raise SystemExit(not result.wasSuccessful())

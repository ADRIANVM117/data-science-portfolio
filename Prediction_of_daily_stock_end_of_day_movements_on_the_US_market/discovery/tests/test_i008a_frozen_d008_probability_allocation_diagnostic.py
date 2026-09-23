"""Synthetic I008A integrity tests; never load physical Discovery or reconstruct D008."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import discovery.i008a_frozen_d008_probability_allocation_diagnostic as i008a  # noqa: E402
from discovery.d008_raw_path_beyond_availability_intensity import build_d008_representations, fit_intensity_transformer  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402


def features(n: int = 9) -> pd.DataFrame:
    values = np.arange(n * 53, dtype=float).reshape(n, 53) / 100.0
    values[1, 4] = np.nan
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, n + 1)); frame.insert(0, "day", np.arange(n)); frame.insert(0, "ID", np.arange(100, 100 + n))
    return frame


def target(index: pd.Index) -> pd.Series:
    return pd.Series(np.resize(np.array([-1, 0, 1], dtype=int), len(index)), index=index, name="reod")


def probabilities(index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    control = pd.DataFrame([[0.2, 0.5, 0.3], [0.0, 1.0, 0.0], [0.7, 0.1, 0.2]] * 3, index=index, columns=i008a.PROBABILITY_COLUMNS)
    candidate = pd.DataFrame([[0.1, 0.5, 0.4], [0.0, 1.0, 0.0], [0.6, 0.1, 0.3]] * 3, index=index, columns=i008a.PROBABILITY_COLUMNS)
    return control, candidate


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except (AssertionError, FileExistsError) as error:
        assert phrase in str(error)
        return
    raise AssertionError("Expected a fail-closed error.")


def synthetic_matrices():
    frame = features(); y = target(frame.index); structure, transformer = fit_intensity_transformer(frame)
    return frame, y, build_d008_representations(frame, transformer.transform(structure["raw_I_recent"]))


def test_constants_artifacts_fingerprint_and_lifecycle() -> None:
    assert i008a.RECONSTRUCTION_LOG_LOSS_ATOL == 1e-8 and i008a.CONSERVATION_ATOL == 1e-10
    assert i008a.ARTIFACT_NAMES == (
        "I008A_reproduction_gate.csv", "I008A_probability_movement.csv", "I008A_realized_class_gain.csv", "I008A_prediction_transitions.csv", "I008A_metadata.json",
    )
    ids = pd.Series([10, 11, 12])
    assert i008a.ordered_id_fingerprint(ids) == i008a.ordered_id_fingerprint(ids)
    assert i008a.ordered_id_fingerprint(ids) != i008a.ordered_id_fingerprint(ids.iloc[::-1])
    assert_raises(lambda: i008a.ordered_id_fingerprint(pd.Series([1, 1])), "unique")
    class Candidate:
        def __init__(self, exists: bool): self._exists = exists
        def exists(self) -> bool: return self._exists
    class ResultsDirectoryFixture:
        def __init__(self, count: int): self._count = count
        def __truediv__(self, name: str) -> Candidate:
            return Candidate(i008a.ARTIFACT_NAMES.index(name) < self._count)
    fresh, partial, complete = ResultsDirectoryFixture(0), ResultsDirectoryFixture(1), ResultsDirectoryFixture(len(i008a.ARTIFACT_NAMES))
    assert i008a.artifact_state(fresh)[0] == "fresh"; i008a.assert_fresh_artifacts(fresh)
    assert i008a.artifact_state(partial)[0] == "incomplete"
    assert_raises(lambda: i008a.assert_fresh_artifacts(partial), "incomplete")
    assert i008a.artifact_state(complete)[0] == "complete"
    assert_raises(lambda: i008a.assert_fresh_artifacts(complete), "overwrite")


def test_persisted_references_and_exact_gate_tolerance() -> None:
    metrics = pd.DataFrame({"fold": [1, 1, 2, 2, 3, 3], "representation": ["C", "P"] * 3, "multiclass_log_loss": [1.0, 0.9, 1.1, 1.0, 1.2, 1.1]})
    refs = i008a.persisted_d008_log_losses(metrics)
    assert refs[(1, "C")] == 1.0 and len(refs) == 6
    assert_raises(lambda: i008a.persisted_d008_log_losses(metrics.iloc[:-1]), "exactly")
    frame, y, matrices = synthetic_matrices(); control, _ = probabilities(frame.index)
    original = i008a.validate_expected_fold_count
    try:
        i008a.validate_expected_fold_count = lambda fold, subset, count: None
        record = i008a.gate_record(fold=1, arm="C", target=y, features=frame, matrices=matrices, probabilities=control, raw_probability_dtype="float64", row_sum_tolerance=1e-12, reconstructed_loss=1.0 + 1e-8, persisted_loss=1.0)
        assert record["passed"] and record["ordered_id_fingerprint"] == i008a.ordered_id_fingerprint(frame["ID"])
        assert_raises(lambda: i008a.gate_record(fold=1, arm="C", target=y, features=frame, matrices=matrices, probabilities=control, raw_probability_dtype="float64", row_sum_tolerance=1e-12, reconstructed_loss=1.0 + 1.1e-8, persisted_loss=1.0), "loss mismatch")
        wrong = matrices.copy(); wrong["P"] = wrong["P"].iloc[::-1]
        assert_raises(lambda: i008a.gate_record(fold=1, arm="C", target=y, features=frame, matrices=wrong, probabilities=control, raw_probability_dtype="float64", row_sum_tolerance=1e-12, reconstructed_loss=1.0, persisted_loss=1.0), "row identity or order")
    finally:
        i008a.validate_expected_fold_count = original


def test_probability_output_classes_shapes_and_float32_tolerance() -> None:
    matrix = pd.DataFrame({"x": [1, 2]})
    class Stub:
        classes_ = np.array([0, 1, 2])
        def predict_proba(self, _): return np.array([[0.109866201877594, 0.8272767663002014, 0.06285698711872101], [0.184199258685112, 0.6397972106933594, 0.17600347101688385]], dtype=np.float32)
    output, dtype, tolerance = i008a.probabilities_with_metadata(Stub(), matrix)
    assert dtype == "float32" and tolerance == 2 * np.finfo(np.float32).eps and output.shape == (2, 3)
    class BadOrder(Stub): classes_ = np.array([1, 0, 2])
    assert_raises(lambda: i008a.probabilities_with_metadata(BadOrder(), matrix), "class order")
    class BadShape(Stub):
        def predict_proba(self, _): return np.ones((2, 2), dtype=np.float32) / 2
    assert_raises(lambda: i008a.probabilities_with_metadata(BadShape(), matrix), "finite and coherent")
    class BadSum(Stub):
        def predict_proba(self, _): return np.array([[0.1, 0.2, 0.8], [0.2, 0.3, 0.5]], dtype=np.float32)
    assert_raises(lambda: i008a.probabilities_with_metadata(BadSum(), matrix), "finite and coherent")
    for invalid in (np.nan, np.inf, -np.inf, -0.1):
        class Invalid(Stub):
            def predict_proba(self, _, value=invalid): return np.array([[value, 0.4, 0.6], [0.2, 0.3, 0.5]])
        assert_raises(lambda Invalid=Invalid: i008a.probabilities_with_metadata(Invalid(), matrix), "finite and coherent")


def test_fit_only_preprocessing_and_shared_c_block() -> None:
    frame, y, matrices = synthetic_matrices(); structure, transformer = fit_intensity_transformer(frame)
    persisted = pd.DataFrame([{"fold": 1, "fit_intensity_median": transformer.fit_intensity_median_, "scaler_mean": transformer.scaler_.mean_[0], "scaler_scale": transformer.scaler_.scale_[0], "n_fit_defined_intensity": int(structure.N_obs_w.gt(0).sum()), "n_fit_undefined_intensity": int(structure.N_obs_w.eq(0).sum())}])
    i008a.validate_fit_preprocessing(1, structure, transformer, persisted)
    bad = persisted.copy(); bad.loc[0, "scaler_mean"] += 1e-8
    assert_raises(lambda: i008a.validate_fit_preprocessing(1, structure, transformer, bad), "preprocessing mismatch")
    assert matrices["C"].equals(matrices["P"].loc[:, matrices["C"].columns]) and y.index.equals(matrices["C"].index)


def test_probability_decomposition_and_undefined_q_behavior() -> None:
    frame = features(); control, candidate = probabilities(frame.index); changes = i008a.probability_changes(control, candidate)
    assert np.isclose(changes.loc[0, "delta_p_D"], 0.0) and np.isclose(changes.loc[0, "delta_q_plus"], 0.8 - 0.6)
    assert not changes.loc[1, "delta_q_defined"] and np.isnan(changes.loc[1, "delta_q_plus"])
    assert np.isclose(changes.loc[2, "delta_p_D"], 0.0) and np.isclose(changes.loc[2, "delta_q_plus"], (1 / 3) - (2 / 9))
    assert_raises(lambda: i008a.probability_changes(control.iloc[::-1], candidate), "identical")


def test_realized_gain_clipping_no_renormalization_and_conservation() -> None:
    frame = features(); y = target(frame.index); control, candidate = probabilities(frame.index)
    gain = i008a.realized_gain(y, control, candidate)
    c_loss = i008a.reconstruction_log_loss(y, control); p_loss = i008a.reconstruction_log_loss(y, candidate)
    assert np.isclose(gain.mean(), c_loss - p_loss) and gain.loc[2] > 0.0
    i008a.assert_gain_conservation(gain, persisted_delta=c_loss - p_loss, reconstructed_delta=c_loss - p_loss)
    assert_raises(lambda: i008a.assert_gain_conservation(gain, persisted_delta=(c_loss - p_loss) + 1e-5, reconstructed_delta=c_loss - p_loss), "conservation")
    zeros = control.copy(); zeros.iloc[0] = [0.0, 1.0, 0.0]
    loss = i008a.per_row_log_loss(y, zeros)
    assert np.isfinite(loss).all() and np.isclose(loss.mean(), log_loss(y, zeros, labels=[-1, 0, 1]))


def test_exact_level_schemas_and_transition_conservation() -> None:
    frame = features(); y = target(frame.index); control, candidate = probabilities(frame.index)
    changes = i008a.probability_changes(control, candidate); gain = i008a.realized_gain(y, control, candidate)
    level1 = i008a.level1_summary(1, changes)
    assert set(level1) == {"fold", "delta_p_D_mean", "delta_p_D_median", "delta_p_D_mean_abs", "delta_p_D_median_abs", "delta_q_plus_n_defined", "delta_q_plus_mean", "delta_q_plus_median", "delta_q_plus_mean_abs", "delta_q_plus_median_abs"}
    level2 = i008a.level2_summary(1, y, changes, gain)
    assert level2.shape[0] == 3 and set(level2.true_class) == {-1, 0, 1}
    transitions = i008a.prediction_transitions(1, y, control, candidate)
    unconditional = transitions.loc[transitions.transition_type.eq("unconditional")]
    assert len(unconditional) == 9 and unconditional["count"].sum() == len(y)
    conditional = transitions.loc[transitions.transition_type.eq("by_true_class")]
    assert len(conditional) == 27 and conditional["count"].sum() == len(y)
    # NumPy argmax resolves an exact -1/+1 tie to the earlier class -1.
    tie = pd.DataFrame([[0.5, 0.0, 0.5]], index=[99], columns=i008a.PROBABILITY_COLUMNS)
    tie_transitions = i008a.prediction_transitions(1, pd.Series([-1], index=[99]), tie, tie)
    assert tie_transitions.loc[(tie_transitions.transition_type.eq("unconditional")) & (tie_transitions.C_prediction.eq(-1)) & (tie_transitions.P_prediction.eq(-1)), "count"].iloc[0] == 1


def test_runner_source_governance_and_no_real_artifacts() -> None:
    runner = (PROJECT_ROOT / "discovery" / "run_i008a_frozen_d008_probability_allocation_diagnostic.py").read_text(encoding="utf-8")
    module = (PROJECT_ROOT / "discovery" / "i008a_frozen_d008_probability_allocation_diagnostic.py").read_text(encoding="utf-8")
    assert "input_test" not in runner and "output_test" not in runner and "D009" not in runner
    assert "shap" not in runner.lower() and "interaction" not in runner.lower()
    assert "all six" in runner and "before all folds" in runner
    assert not any((PROJECT_ROOT / "discovery" / "results" / name).exists() for name in i008a.ARTIFACT_NAMES)


if __name__ == "__main__":
    tests = (
        test_constants_artifacts_fingerprint_and_lifecycle,
        test_persisted_references_and_exact_gate_tolerance,
        test_probability_output_classes_shapes_and_float32_tolerance,
        test_fit_only_preprocessing_and_shared_c_block,
        test_probability_decomposition_and_undefined_q_behavior,
        test_realized_gain_clipping_no_renormalization_and_conservation,
        test_exact_level_schemas_and_transition_conservation,
        test_runner_source_governance_and_no_real_artifacts,
    )
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests))
    raise SystemExit(not result.wasSuccessful())

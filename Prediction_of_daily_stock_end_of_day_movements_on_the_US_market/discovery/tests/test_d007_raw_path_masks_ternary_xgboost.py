"""Synthetic integrity tests for frozen D007; never loads physical Discovery data."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import assert_e_dev_only, validate_discovery_features  # noqa: E402
from discovery.d007_raw_path_masks_ternary_xgboost import (  # noqa: E402
    DISCOVERY_FOLDS, ENCODED_CLASS_ORDER, M_COLUMNS, RM_COLUMNS, XGBOOST_PARAMS,
    build_d007_representations, encode_ternary_target, evaluate_d007_probabilities,
    evaluate_majority_baseline, make_d007_classifier, primary_screen, ternary_probabilities,
)
import discovery.run_d007_raw_path_masks_ternary_xgboost as d007_runner  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402


def features(n: int = 18) -> pd.DataFrame:
    values = np.arange(n * 53, dtype=float).reshape(n, 53) / 100.0
    values[1, 4] = np.nan; values[2, 41:53] = np.nan; values[3, 0] = 0.0
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, n + 1)); frame.insert(0, "day", np.arange(n) % 4); frame.insert(0, "ID", np.arange(100, 100 + n))
    return frame


def labels(index: pd.Index) -> pd.Series:
    return pd.Series(np.resize(np.array([-1, 0, 1], dtype=int), len(index)), index=index, name="reod")


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert phrase in str(error); return
    raise AssertionError("Expected an integrity AssertionError.")


def test_frozen_schemas_and_nan_zero_semantics() -> None:
    frame, target = features(), labels(pd.RangeIndex(18)); matrices = build_d007_representations(frame, target)
    assert tuple(matrices["M"].columns) == M_COLUMNS and matrices["M"].shape == (18, 53)
    assert tuple(matrices["R+M"].columns) == RM_COLUMNS and matrices["R+M"].shape == (18, 106)
    assert matrices["M"].equals(matrices["R+M"].loc[:, M_COLUMNS])
    assert matrices["M"].loc[1, "m4"] == 1 and matrices["M"].loc[3, "m0"] == 0
    assert np.isnan(matrices["R+M"].loc[1, "r4"]) and matrices["R+M"].loc[3, "r0"] == 0.0


def test_nonfinite_and_alignment_fail_closed() -> None:
    frame, target = features(), labels(pd.RangeIndex(18)); bad = frame.copy(); bad.loc[0, "r0"] = np.inf
    assert_raises(lambda: build_d007_representations(bad, target), "observed returns")
    shifted = target.copy(); shifted.index = pd.RangeIndex(1, 19)
    assert_raises(lambda: build_d007_representations(frame, shifted), "aligned")


def test_xgboost_encoding_probability_and_metrics() -> None:
    frame, target = features(), labels(pd.RangeIndex(18)); matrix = build_d007_representations(frame, target)["M"]
    model = make_d007_classifier(); assert {key: model.get_params()[key] for key in XGBOOST_PARAMS} == XGBOOST_PARAMS
    model.fit(matrix, encode_ternary_target(target)); probabilities = ternary_probabilities(model, matrix)
    assert tuple(model.classes_) == ENCODED_CLASS_ORDER and np.allclose(probabilities.sum(axis=1), 1.0)
    metrics, confusion = evaluate_d007_probabilities(target, probabilities)
    assert "multiclass_log_loss" in metrics and len(confusion) == len(confusion[0]) == 3


def test_probability_coherence_is_dtype_aware_but_still_fails_closed() -> None:
    matrix = pd.DataFrame({"m0": [0, 1]})

    class Stub:
        classes_ = np.array([0, 1, 2])
        def __init__(self, values): self.values = values
        def predict_proba(self, _): return self.values

    valid_float32 = np.array([[0.109866201877594, 0.8272767663002014, 0.06285698711872101], [0.184199258685112, 0.6397972106933594, 0.17600347101688385]], dtype=np.float32)
    assert np.abs(valid_float32.sum(axis=1) - 1.0).max() > 1e-12
    assert np.allclose(ternary_probabilities(Stub(valid_float32), matrix).sum(axis=1), 1.0)
    bad_shape = np.ones((2, 2), dtype=np.float32) / 2
    assert_raises(lambda: ternary_probabilities(Stub(bad_shape), matrix), "finite and coherent")
    bad_sum = np.array([[0.1, 0.2, 0.8], [0.2, 0.3, 0.5]], dtype=np.float32)
    assert_raises(lambda: ternary_probabilities(Stub(bad_sum), matrix), "finite and coherent")
    for invalid in (np.array([[np.nan, 0.4, 0.6], [0.2, 0.3, 0.5]]), np.array([[np.inf, 0.4, 0.6], [0.2, 0.3, 0.5]]), np.array([[-np.inf, 0.4, 0.6], [0.2, 0.3, 0.5]]), np.array([[-0.01, 0.41, 0.60], [0.2, 0.3, 0.5]])):
        assert_raises(lambda invalid=invalid: ternary_probabilities(Stub(invalid), matrix), "finite and coherent")
    class WrongOrder(Stub):
        classes_ = np.array([1, 0, 2])
    assert_raises(lambda: ternary_probabilities(WrongOrder(valid_float32), matrix), "encoded")


def test_screen_majority_and_boundary_reuse() -> None:
    assert primary_screen([0.01, 0.02, 0.03])["screen_met"] and not primary_screen([0.01, -0.02, 0.03])["screen_met"]
    frame, target = features(), labels(pd.RangeIndex(18)); selected, _, _ = evaluate_majority_baseline(target, target)
    assert selected == -1
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in DISCOVERY_FOLDS] == [(1, 0, 202, 203, 252), (2, 0, 252, 253, 302), (3, 0, 302, 303, 352)]
    validate_discovery_features(frame)
    partition = pd.DataFrame({"equity": list(range(1, 19)) + [99], "partition": ["E_dev"] * 18 + ["E_holdout"]})
    assert_e_dev_only(frame, partition); bad = frame.copy(); bad.loc[0, "day"] = 353
    assert_raises(lambda: validate_discovery_features(bad), "outside")


def test_runner_has_no_protected_references_and_preexecution_artifact_guard() -> None:
    """D007 artifacts are expected after closure; overwrite protection is pre-execution only."""
    source = (PROJECT_ROOT / "discovery" / "run_d007_raw_path_masks_ternary_xgboost.py").read_text(encoding="utf-8")
    assert "input_test" not in source and "output_test" not in source and "full-training" not in source
    # The former assertion that the repository lacks D007_* artifacts was a
    # pre-execution lifecycle guard. D007 is closed, so valid artifacts must
    # persist. Verify the guard itself in isolation without touching them.
    class Candidate:
        def __init__(self, exists: bool): self._exists = exists
        def exists(self) -> bool: return self._exists

    class ResultsDirectoryFixture:
        def __init__(self, exists: bool): self._exists = exists
        def __truediv__(self, _name: str) -> Candidate: return Candidate(self._exists)

    original_results_dir = d007_runner.RESULTS_DIR
    try:
        d007_runner.RESULTS_DIR = ResultsDirectoryFixture(False)
        d007_runner.assert_fresh_artifacts()
        d007_runner.RESULTS_DIR = ResultsDirectoryFixture(True)
        try:
            d007_runner.assert_fresh_artifacts()
        except FileExistsError:
            pass
        else:
            raise AssertionError("D007 pre-execution artifact overwrite guard must fail closed.")
    finally:
        d007_runner.RESULTS_DIR = original_results_dir


if __name__ == "__main__":
    tests = (
        test_frozen_schemas_and_nan_zero_semantics,
        test_nonfinite_and_alignment_fail_closed,
        test_xgboost_encoding_probability_and_metrics,
        test_probability_coherence_is_dtype_aware_but_still_fails_closed,
        test_screen_majority_and_boundary_reuse,
        test_runner_has_no_protected_references_and_preexecution_artifact_guard,
    )
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests))
    raise SystemExit(not result.wasSuccessful())

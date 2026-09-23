"""Synthetic D008 integrity tests; never load physical Discovery data."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS, validate_discovery_features  # noqa: E402
from discovery.d007_raw_path_masks_ternary_xgboost import (  # noqa: E402
    ENCODED_CLASS_ORDER,
    XGBOOST_PARAMS,
    encode_ternary_target,
    make_d007_classifier,
    ternary_probabilities,
)
from discovery.d008_raw_path_beyond_availability_intensity import (  # noqa: E402
    C_COLUMNS,
    EXPECTED_ROW_COUNTS,
    P_COLUMNS,
    RECENT_WINDOW_COLUMNS,
    RECENT_WINDOW_SIZE,
    assert_identical_d008_rows,
    build_d008_representations,
    d008_primary_screen,
    fit_intensity_transformer,
    validate_expected_fold_count,
)
import discovery.run_d008_raw_path_beyond_availability_intensity as d008_runner  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp008_recent_movement_intensity import build_recent_window_structure  # noqa: E402


def synthetic_features(n: int = 18) -> pd.DataFrame:
    values = np.arange(n * 53, dtype=float).reshape(n, 53) / 100.0
    values[1, 4] = np.nan
    values[2, 41:53] = np.nan
    values[3, 41:53] = 0.0
    values[4, 41:53] = np.nan
    values[4, 41:44] = [0.0, -3.0, 4.0]
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, n + 1))
    frame.insert(0, "day", np.arange(n) % 4)
    frame.insert(0, "ID", np.arange(100, 100 + n))
    return frame


def labels(index: pd.Index) -> pd.Series:
    return pd.Series(np.resize(np.array([-1, 0, 1], dtype=int), len(index)), index=index, name="reod")


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert phrase in str(error)
        return
    raise AssertionError("Expected an integrity AssertionError.")


def test_exact_exp008_window_rms_zero_nan_and_nonfinite_semantics() -> None:
    frame = synthetic_features()
    assert RECENT_WINDOW_COLUMNS == tuple(f"r{i}" for i in range(41, 53))
    assert RECENT_WINDOW_SIZE == len(RECENT_WINDOW_COLUMNS) == 12
    structure = build_recent_window_structure(frame)
    assert structure.loc[2, "N_obs_w"] == 0 and np.isnan(structure.loc[2, "raw_I_recent"])
    assert structure.loc[3, "N_obs_w"] == 12 and structure.loc[3, "raw_I_recent"] == 0.0
    assert structure.loc[4, "N_obs_w"] == 3
    assert np.isclose(structure.loc[4, "raw_I_recent"], np.sqrt((0.0 + 9.0 + 16.0) / 3.0))
    bad = frame.copy()
    bad.loc[0, "r0"] = np.inf
    assert_raises(lambda: build_d008_representations(bad, pd.DataFrame({"I_recent_z": np.zeros(len(bad))}, index=bad.index)), "observed returns")
    bad_w = frame.copy()
    bad_w.loc[0, "r41"] = -np.inf
    assert_raises(lambda: build_recent_window_structure(bad_w), "Observed non-NaN")


def test_fit_only_intensity_and_oos_perturbation_boundary() -> None:
    fit = synthetic_features(9)
    oos = synthetic_features(6)
    fit_structure, transformer = fit_intensity_transformer(fit)
    before = (transformer.fit_intensity_median_, transformer.scaler_.mean_.copy(), transformer.scaler_.scale_.copy())
    oos_structure = build_recent_window_structure(oos)
    first = transformer.transform(oos_structure["raw_I_recent"])
    changed = oos_structure["raw_I_recent"].copy()
    changed.iloc[:] = 1_000_000.0
    second = transformer.transform(changed)
    assert transformer.fit_intensity_median_ == before[0]
    assert np.array_equal(transformer.scaler_.mean_, before[1])
    assert np.array_equal(transformer.scaler_.scale_, before[2])
    assert not first.equals(second)
    assert np.isfinite(transformer.transform(fit_structure["raw_I_recent"]).to_numpy()).all()


def test_c_p_schema_masks_native_nans_and_shared_intensity_identity() -> None:
    frame = synthetic_features()
    target = labels(frame.index)
    structure, transformer = fit_intensity_transformer(frame)
    intensity_z = transformer.transform(structure["raw_I_recent"])
    representations = build_d008_representations(frame, intensity_z)
    assert tuple(representations["C"].columns) == C_COLUMNS and representations["C"].shape == (len(frame), 54)
    assert tuple(representations["P"].columns) == P_COLUMNS and representations["P"].shape == (len(frame), 107)
    assert representations["C"].dtypes.iloc[:53].eq(np.dtype("int8")).all()
    assert representations["C"].loc[:, ["I_recent_z"]].equals(representations["P"].loc[:, ["I_recent_z"]])
    assert representations["C"].loc[1, "m4"] == 1 and representations["C"].loc[3, "m41"] == 0
    assert np.isnan(representations["P"].loc[1, "r4"]) and representations["P"].loc[3, "r41"] == 0.0
    assert_identical_d008_rows(representations, frame, target)
    assert_raises(lambda: assert_identical_d008_rows({**representations, "P": representations["P"].iloc[::-1]}, frame, target), "row identity or order")


def test_frozen_learner_probability_order_folds_counts_and_screen() -> None:
    frame = synthetic_features()
    target = labels(frame.index)
    structure, transformer = fit_intensity_transformer(frame)
    matrix = build_d008_representations(frame, transformer.transform(structure["raw_I_recent"]))["C"]
    model = make_d007_classifier()
    assert {name: model.get_params()[name] for name in XGBOOST_PARAMS} == XGBOOST_PARAMS
    model.fit(matrix, encode_ternary_target(target))
    probabilities = ternary_probabilities(model, matrix)
    assert tuple(model.classes_) == ENCODED_CLASS_ORDER and list(probabilities.columns) == ["p_minus", "p_zero", "p_plus"]
    assert [(fold.fold, fold.train_start, fold.train_end, fold.oos_start, fold.oos_end) for fold in DISCOVERY_FOLDS] == [
        (1, 0, 202, 203, 252), (2, 0, 252, 253, 302), (3, 0, 302, 303, 352)
    ]
    assert EXPECTED_ROW_COUNTS == {1: {"fit": 271897, "validation": 67031}, 2: {"fit": 338928, "validation": 66989}, 3: {"fit": 405917, "validation": 66899}}
    validate_expected_fold_count(1, "fit", 271897)
    assert_raises(lambda: validate_expected_fold_count(1, "validation", 1), "row count")
    assert d008_primary_screen([0.01, 0.02, 0.03])["screen_met"]
    assert not d008_primary_screen([0.01, 0.0, 0.03])["screen_met"]


def test_runner_is_protected_and_artifact_guard_is_preexecution_only() -> None:
    source = (PROJECT_ROOT / "discovery" / "run_d008_raw_path_beyond_availability_intensity.py").read_text(encoding="utf-8")
    assert "input_test" not in source and "output_test" not in source and "full-training" not in source
    # Artifact absence is a pre-execution lifecycle guard, not a permanent
    # repository invariant: D008 is closed and its valid artifacts must persist.
    # Exercise the runner guard against isolated synthetic directory states.

    class Candidate:
        def __init__(self, exists: bool): self._exists = exists
        def exists(self) -> bool: return self._exists

    class ResultsDirectoryFixture:
        def __init__(self, existing_names: set[str]): self._existing_names = existing_names
        def __truediv__(self, name: str) -> Candidate: return Candidate(name in self._existing_names)

    original = d008_runner.RESULTS_DIR
    try:
        d008_runner.RESULTS_DIR = ResultsDirectoryFixture(set())
        d008_runner.assert_fresh_artifacts()
        d008_runner.RESULTS_DIR = ResultsDirectoryFixture({d008_runner.ARTIFACT_NAMES[0]})
        try:
            d008_runner.assert_fresh_artifacts()
        except FileExistsError:
            pass
        else:
            raise AssertionError("D008 artifact overwrite guard must fail closed.")
    finally:
        d008_runner.RESULTS_DIR = original


if __name__ == "__main__":
    tests = (
        test_exact_exp008_window_rms_zero_nan_and_nonfinite_semantics,
        test_fit_only_intensity_and_oos_perturbation_boundary,
        test_c_p_schema_masks_native_nans_and_shared_intensity_identity,
        test_frozen_learner_probability_order_folds_counts_and_screen,
        test_runner_is_protected_and_artifact_guard_is_preexecution_only,
    )
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    )
    raise SystemExit(not result.wasSuccessful())

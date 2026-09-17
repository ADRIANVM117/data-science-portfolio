"""Synthetic integrity tests for frozen CONF_001 only."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.conf001_recent_intensity_confirmation import (  # noqa: E402
    CANDIDATE_COLUMNS,
    CONFIRMATION_BLOCKS,
    CONTROL_COLUMNS,
    assert_confirmation_row_identity,
    confirmation_primary_decision,
    fit_confirmation_transformer,
    prepare_confirmation_matrices,
)
from src.exp000_validation import FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402
from src.exp008_recent_movement_intensity import RECENT_WINDOW_COLUMNS  # noqa: E402


def assert_raises(action, expected: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert expected in str(error)
        return
    raise AssertionError("Expected an integrity assertion.")


def feature_frame(values: np.ndarray, *, index: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS, index=index)
    frame.insert(0, "equity", np.arange(1, len(frame) + 1))
    frame.insert(0, "day", 0)
    frame.insert(0, "ID", np.arange(1000, 1000 + len(frame)))
    return frame


def test_confirmation_blocks_and_frozen_partition_split() -> None:
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in CONFIRMATION_BLOCKS] == [
        (1, 0, 352, 353, 402), (2, 0, 402, 403, 452), (3, 0, 452, 453, 502)
    ]
    rows, labels, identifier = [], [], 0
    for day in range(4):
        for equity in range(5):
            row = {"ID": identifier, "day": day, "equity": equity}
            row.update({column: 0.0 for column in RETURN_COLUMNS})
            rows.append(row)
            labels.append(identifier % 2)
            identifier += 1
    features = pd.DataFrame(rows)
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, pd.Series(labels, index=features.index), partition, FoldSpec(1, 0, 1, 2, 3))
    holdout = set(partition.loc[partition["partition"].eq("E_holdout"), "equity"])
    assert holdout.isdisjoint(set(subsets.fit_features["equity"]))
    assert set(subsets.joint_features["equity"]).issubset(holdout)


def test_feature_schema_w_q_rms_and_fit_only_preprocessing() -> None:
    assert RECENT_WINDOW_COLUMNS == tuple(f"r{i}" for i in range(41, 53))
    values = np.zeros((4, 53), dtype=float)
    values[0, 41:53] = np.nan
    values[1, 41:53] = [0.0, 3.0, 4.0] + [np.nan] * 9
    values[2, 41:53] = 2.0
    values[3, 41:53] = 5.0
    features = feature_frame(values, index=[40, 30, 20, 10])
    target = build_binary_target(pd.Series([0, -1, 1, 0], index=features.index))
    fit_structure, transformer = fit_confirmation_transformer(features)
    assert fit_structure.loc[40, "q"] == 1
    assert np.isnan(fit_structure.loc[40, "raw_I_recent"])
    assert fit_structure.loc[30, "N_obs_w"] == 3
    assert np.isclose(fit_structure.loc[30, "raw_I_recent"], np.sqrt(25.0 / 3.0))
    assert fit_structure.loc[20, "raw_I_recent"] == 2.0
    median_before = transformer.fit_intensity_median_
    mean_before = transformer.scaler_.mean_.copy()
    scale_before = transformer.scaler_.scale_.copy()
    structure, matrices = prepare_confirmation_matrices(features, target, transformer)
    assert structure.equals(fit_structure)
    assert tuple(matrices["C"].columns) == CONTROL_COLUMNS
    assert tuple(matrices["B"].columns) == CANDIDATE_COLUMNS
    assert matrices["C"].index.equals(features.index)
    assert matrices["B"].index.equals(features.index)
    assert matrices["C"].equals(matrices["B"].loc[:, CONTROL_COLUMNS])
    assert transformer.fit_intensity_median_ == median_before
    assert np.array_equal(transformer.scaler_.mean_, mean_before)
    assert np.array_equal(transformer.scaler_.scale_, scale_before)
    assert_confirmation_row_identity(matrices, features, target)


def test_oos_perturbation_cannot_change_fit_parameters_and_invalid_values_fail() -> None:
    fit_values = np.zeros((3, 53), dtype=float)
    fit_values[:, 41:53] = [[1.0] * 12, [2.0] * 12, [3.0] * 12]
    fit = feature_frame(fit_values, index=[1, 2, 3])
    _, transformer = fit_confirmation_transformer(fit)
    before = (transformer.fit_intensity_median_, transformer.scaler_.mean_.copy(), transformer.scaler_.scale_.copy())
    oos_values = fit_values.copy()
    oos_values[:, 41:53] = 1_000_000.0
    oos = feature_frame(oos_values, index=[4, 5, 6])
    target = pd.Series([0, 1, 0], index=oos.index, name="Z", dtype="int8")
    prepare_confirmation_matrices(oos, target, transformer)
    assert transformer.fit_intensity_median_ == before[0]
    assert np.array_equal(transformer.scaler_.mean_, before[1])
    assert np.array_equal(transformer.scaler_.scale_, before[2])
    invalid = fit.copy()
    invalid.loc[1, "r41"] = np.inf
    assert_raises(lambda: fit_confirmation_transformer(invalid), "Observed non-NaN")


def test_primary_decision_model_parameters_and_runner_boundary() -> None:
    passed = confirmation_primary_decision([0.001, 0.02, 0.0001])
    assert passed["confirmation_criterion_satisfied"]
    assert passed["n_joint_blocks"] == 3
    assert not confirmation_primary_decision([0.01, 0.0, 0.02])["confirmation_criterion_satisfied"]
    assert_raises(lambda: confirmation_primary_decision([0.01] * 4), "exactly three")
    classifier = make_binary_logistic_regression()
    assert {name: classifier.get_params()[name] for name in LOGISTIC_REGRESSION_PARAMS} == LOGISTIC_REGRESSION_PARAMS
    runner_source = (PROJECT_ROOT / "scripts" / "run_conf001_recent_intensity_confirmation.py").read_text(encoding="utf-8")
    assert "input_test" not in runner_source and "output_test" not in runner_source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_confirmation_blocks_and_frozen_partition_split,
        test_feature_schema_w_q_rms_and_fit_only_preprocessing,
        test_oos_perturbation_cannot_change_fit_parameters_and_invalid_values_fail,
        test_primary_decision_model_parameters_and_runner_boundary,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

"""Synthetic contract tests for frozen EXP_009 recency-weighted intensity."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    aggregate_secondary_decision,
    assert_both_binary_classes,
    build_binary_target,
    build_missing_ratio,
    make_binary_logistic_regression,
    secondary_fold_decision,
    threshold_predictions,
)
from src.exp009_recency_weighted_movement_intensity import (  # noqa: E402
    RECENCY_WEIGHTS,
    RECENCY_WEIGHT_SUM,
    RECENT_WINDOW_COLUMNS,
    RECENT_WINDOW_SIZE,
    CompleteWindowIntensityTransformer,
    aggregate_primary_decision,
    assert_identical_representation_rows,
    build_complete_window_structure,
    build_exp009_representations,
    build_raw_intensities,
    filter_complete_window_rows,
    primary_fold_decision,
)


def assert_raises(action, message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert message in str(error)
        return
    raise AssertionError("Expected integrity assertion.")


def feature_frame(values: np.ndarray, *, index: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS, index=index)
    frame.insert(0, "equity", np.arange(1, len(frame) + 1))
    frame.insert(0, "day", 0)
    frame.insert(0, "ID", np.arange(100, 100 + len(frame)))
    return frame


def complete_values(window: list[float]) -> np.ndarray:
    values = np.zeros((1, 53), dtype=float)
    values[0, 41:53] = window
    return values


def test_window_weights_complete_semantics_zeros_and_nonfinite_rejection() -> None:
    assert RECENT_WINDOW_COLUMNS == tuple(f"r{i}" for i in range(41, 53))
    assert RECENT_WINDOW_SIZE == 12
    assert np.array_equal(RECENCY_WEIGHTS, np.arange(1, 13, dtype=float))
    assert RECENCY_WEIGHT_SUM == 78.0
    values = np.zeros((3, 53), dtype=float)
    values[0, 41:53] = np.nan
    values[1, 41:53] = 0.0
    values[1, 41] = np.nan
    values[2, 41:53] = 0.0
    frame = feature_frame(values, index=[7, 9, 11])
    target = pd.Series([0, 1, 0], index=frame.index, name="Z")
    structure = build_complete_window_structure(frame)
    assert structure.loc[7, "N_obs_w"] == 0 and not structure.loc[7, "complete_w"]
    assert structure.loc[9, "N_obs_w"] == 11 and not structure.loc[9, "complete_w"]
    assert structure.loc[11, "N_obs_w"] == 12 and structure.loc[11, "complete_w"]
    retained, retained_target = filter_complete_window_rows(frame, target, structure)
    assert retained.index.tolist() == [11] and retained_target.index.tolist() == [11]
    raw = build_raw_intensities(retained)
    assert raw.loc[11, "raw_I_recent"] == 0.0 and raw.loc[11, "raw_I_recency"] == 0.0

    invalid = frame.copy()
    invalid.loc[11, "r41"] = np.inf
    invalid_structure = build_complete_window_structure(invalid)
    assert invalid_structure.loc[11, "complete_w"]
    assert_raises(
        lambda: filter_complete_window_rows(invalid, target, invalid_structure),
        "retained observed W values must be finite",
    )


def test_exact_formulas_and_frozen_permutation_examples() -> None:
    g = [2.0] * 8 + [20.0] * 4
    j = [2.0] * 6 + [20.0] * 4 + [2.0] * 2
    h = [20.0] * 4 + [2.0] * 8
    frame = feature_frame(np.vstack([complete_values(g), complete_values(j), complete_values(h)]).reshape(3, 53))
    raw = build_raw_intensities(frame)
    expected_unweighted = np.sqrt((8 * 2.0**2 + 4 * 20.0**2) / 12.0)
    assert np.allclose(raw["raw_I_recent"], expected_unweighted)
    assert raw.loc[0, "raw_I_recency"] > raw.loc[1, "raw_I_recency"] > raw.loc[2, "raw_I_recency"]
    expected_weighted_g = np.sqrt(np.dot(RECENCY_WEIGHTS, np.square(g)) / 78.0)
    assert np.isclose(raw.loc[0, "raw_I_recency"], expected_weighted_g)


def test_independent_fit_only_scalers_and_aligned_c_d_matrices() -> None:
    fit_raw = pd.DataFrame(
        {"raw_I_recent": [1.0, 2.0, 3.0], "raw_I_recency": [4.0, 7.0, 10.0]}, index=[10, 11, 12]
    )
    oos_raw = pd.DataFrame(
        {"raw_I_recent": [999.0, 1000.0], "raw_I_recency": [-999.0, -1000.0]}, index=[20, 21]
    )
    transformer = CompleteWindowIntensityTransformer().fit(fit_raw)
    recent_mean = transformer.scaler_recent_.mean_.copy()
    recency_mean = transformer.scaler_recency_.mean_.copy()
    recent_scale = transformer.scaler_recent_.scale_.copy()
    recency_scale = transformer.scaler_recency_.scale_.copy()
    transformed = transformer.transform(oos_raw)
    assert np.array_equal(transformer.scaler_recent_.mean_, recent_mean)
    assert np.array_equal(transformer.scaler_recency_.mean_, recency_mean)
    assert np.array_equal(transformer.scaler_recent_.scale_, recent_scale)
    assert np.array_equal(transformer.scaler_recency_.scale_, recency_scale)
    assert not np.array_equal(recent_mean, recency_mean)

    values = np.zeros((2, 53), dtype=float)
    values[0, 0] = np.nan
    features = feature_frame(values, index=[20, 21])
    target = pd.Series([0, 1], index=features.index, name="Z")
    ratio = build_missing_ratio(features)
    representations = build_exp009_representations(ratio, transformed)
    assert list(representations["C"].columns) == ["missing_ratio", "I_recent_z"]
    assert list(representations["D"].columns) == ["missing_ratio", "I_recent_z", "I_recency_z"]
    assert representations["C"].shape[1] == 2 and representations["D"].shape[1] == 3
    assert "q" not in representations["C"] and "q" not in representations["D"]
    assert representations["C"].loc[20, "missing_ratio"] == 1 / 53
    assert_identical_representation_rows(representations, features, target)
    assert_raises(
        lambda: assert_identical_representation_rows(
            {**representations, "D": representations["D"].iloc[::-1]}, features, target
        ),
        "row identity or order",
    )


def test_frozen_split_model_decisions_and_training_only_runner() -> None:
    assert [(fold.fold, fold.train_start, fold.train_end, fold.oos_start, fold.oos_end) for fold in FOLDS] == [
        (1, 0, 302, 303, 352), (2, 0, 352, 353, 402),
        (3, 0, 402, 403, 452), (4, 0, 452, 453, 502),
    ]
    rows, labels, identifier = [], [], 0
    for day in range(4):
        for equity in range(5):
            row = {"ID": identifier, "day": day, "equity": equity}
            row.update({column: 0.0 for column in RETURN_COLUMNS})
            rows.append(row)
            labels.append(-1 if identifier % 2 == 0 else 0)
            identifier += 1
    features = pd.DataFrame(rows)
    target = build_binary_target(pd.Series(labels, name="reod"))
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, target, partition, FoldSpec(1, 0, 1, 2, 3))
    assert set(subsets.fit_features["equity"]).isdisjoint(
        set(partition.loc[partition["partition"].eq("E_holdout"), "equity"])
    )
    assert_both_binary_classes(subsets.fit_target, context="synthetic fit")
    classifier = make_binary_logistic_regression()
    assert classifier.get_params() == {**classifier.get_params(), **LOGISTIC_REGRESSION_PARAMS}
    assert np.array_equal(threshold_predictions(np.array([0.499, 0.5, 0.501])), np.array([0, 1, 1], dtype=np.int8))

    primary = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.02, 0.03, 0.04])
    assert primary["incremental_recency_weighted_intensity_evidence"]
    assert not aggregate_primary_decision([0.51] * 4, [0.01, 0.0, 0.01, 0.01])["incremental_recency_weighted_intensity_evidence"]
    assert not primary_fold_decision(0.5, 0.01)["d_roc_auc_above_one_half"]
    failed = {"balanced_accuracy": 0.5, "recall_neutral": 1.0, "recall_directional": 0.0}
    assert not aggregate_secondary_decision([secondary_fold_decision(failed)] * 4, [0.01] * 4)[
        "fixed_threshold_classification_evidence"
    ]
    runner_source = (PROJECT_ROOT / "scripts" / "run_exp_009_recency_weighted_movement_intensity.py").read_text(encoding="utf-8")
    assert "input_test" not in runner_source and "output_test" not in runner_source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_window_weights_complete_semantics_zeros_and_nonfinite_rejection,
        test_exact_formulas_and_frozen_permutation_examples,
        test_independent_fit_only_scalers_and_aligned_c_d_matrices,
        test_frozen_split_model_decisions_and_training_only_runner,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

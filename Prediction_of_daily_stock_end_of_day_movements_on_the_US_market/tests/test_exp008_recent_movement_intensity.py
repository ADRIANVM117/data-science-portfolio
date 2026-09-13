"""Synthetic contract tests for frozen EXP_008 recent movement intensity."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    aggregate_secondary_decision,
    build_missing_ratio,
    build_binary_target,
    secondary_fold_decision,
)
from src.exp008_recent_movement_intensity import (  # noqa: E402
    RECENT_WINDOW_COLUMNS,
    RECENT_WINDOW_SIZE,
    RecentIntensityTransformer,
    aggregate_primary_decision,
    assert_identical_representation_rows,
    build_exp008_representations,
    build_recent_window_structure,
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


def test_window_q_zeros_rms_and_nonfinite_observed_contract() -> None:
    assert RECENT_WINDOW_COLUMNS == tuple(f"r{i}" for i in range(41, 53))
    assert len(RECENT_WINDOW_COLUMNS) == RECENT_WINDOW_SIZE == 12
    values = np.zeros((3, 53), dtype=float)
    values[0, 41:53] = np.nan
    values[1, 41:53] = np.nan
    values[1, 41:44] = [0.0, -3.0, 4.0]
    values[2, 41:53] = 0.0
    frame = feature_frame(values, index=[7, 9, 11])
    structure = build_recent_window_structure(frame)
    assert structure.loc[7, "N_obs_w"] == 0 and structure.loc[7, "q"] == 1
    assert np.isnan(structure.loc[7, "raw_I_recent"])
    assert structure.loc[9, "N_obs_w"] == 3 and structure.loc[9, "q"] == 0
    assert np.isclose(structure.loc[9, "raw_I_recent"], np.sqrt((0.0 + 9.0 + 16.0) / 3.0))
    assert structure.loc[11, "N_obs_w"] == 12 and structure.loc[11, "raw_I_recent"] == 0.0

    invalid = frame.copy()
    invalid.loc[7, "r41"] = np.inf
    assert_raises(lambda: build_recent_window_structure(invalid), "Observed non-NaN")


def test_fit_only_median_scaling_and_oos_perturbation_boundary() -> None:
    fit_raw = pd.Series([1.0, np.nan, 3.0], index=[10, 11, 12], name="raw_I_recent")
    oos_raw = pd.Series([np.nan, 1000.0], index=[20, 21], name="raw_I_recent")
    transformer = RecentIntensityTransformer().fit(fit_raw)
    assert transformer.fit_intensity_median_ == 2.0
    assert np.isclose(transformer.scaler_.mean_[0], 2.0)
    median_before = transformer.fit_intensity_median_
    mean_before = transformer.scaler_.mean_.copy()
    scale_before = transformer.scaler_.scale_.copy()
    transformed = transformer.transform(oos_raw)
    assert list(transformed.columns) == ["I_recent_z"] and np.isfinite(transformed.to_numpy()).all()
    assert transformer.fit_intensity_median_ == median_before
    assert np.array_equal(transformer.scaler_.mean_, mean_before)
    assert np.array_equal(transformer.scaler_.scale_, scale_before)
    assert_raises(lambda: RecentIntensityTransformer().fit(pd.Series([np.nan, np.nan])), "no defined")
    assert_raises(lambda: transformer.transform(pd.Series([np.inf])), "non-finite non-NaN")


def test_a_c_b_are_exact_aligned_and_use_exp002_missing_ratio() -> None:
    values = np.zeros((3, 53), dtype=float)
    values[0, [0, 41]] = np.nan
    values[1, 41:53] = np.nan
    values[2, [1, 2, 3]] = np.nan
    features = feature_frame(values, index=[30, 20, 10])
    target = build_binary_target(pd.Series([-1, 0, 1], index=features.index, name="reod"))
    structure = build_recent_window_structure(features)
    ratio = build_missing_ratio(features)
    transformer = RecentIntensityTransformer().fit(structure["raw_I_recent"])
    intensity_z = transformer.transform(structure["raw_I_recent"])
    representations = build_exp008_representations(ratio, structure, intensity_z)
    assert list(representations["A"].columns) == ["missing_ratio"]
    assert list(representations["C"].columns) == ["missing_ratio", "q"]
    assert list(representations["B"].columns) == ["missing_ratio", "q", "I_recent_z"]
    assert representations["A"].loc[30, "missing_ratio"] == 2 / 53
    assert representations["C"].loc[20, "q"] == 1
    assert_identical_representation_rows(representations, features, target)
    assert_raises(
        lambda: assert_identical_representation_rows(
            {**representations, "B": representations["B"].iloc[::-1]}, features, target
        ),
        "row identity or order",
    )


def test_frozen_split_decisions_and_runner_is_training_only() -> None:
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
            labels.append(0 if identifier % 2 == 0 else 1)
            identifier += 1
    features = pd.DataFrame(rows)
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, pd.Series(labels, name="Z"), partition, FoldSpec(1, 0, 1, 2, 3))
    assert set(subsets.fit_features["equity"]).isdisjoint(
        set(partition.loc[partition["partition"].eq("E_holdout"), "equity"])
    )

    primary = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.02, 0.03, 0.04])
    assert primary["incremental_recent_intensity_evidence"]
    assert not aggregate_primary_decision([0.51] * 4, [0.01, 0.0, 0.01, 0.01])["incremental_recent_intensity_evidence"]
    assert not primary_fold_decision(0.5, 0.01)["b_roc_auc_above_one_half"]
    failed = {"balanced_accuracy": 0.5, "recall_neutral": 1.0, "recall_directional": 0.0}
    assert not aggregate_secondary_decision([secondary_fold_decision(failed)] * 4, [0.01] * 4)[
        "fixed_threshold_classification_evidence"
    ]
    runner_source = (PROJECT_ROOT / "scripts" / "run_exp_008_recent_movement_intensity.py").read_text(encoding="utf-8")
    assert "input_test" not in runner_source and "output_test" not in runner_source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_window_q_zeros_rms_and_nonfinite_observed_contract,
        test_fit_only_median_scaling_and_oos_perturbation_boundary,
        test_a_c_b_are_exact_aligned_and_use_exp002_missing_ratio,
        test_frozen_split_decisions_and_runner_is_training_only,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

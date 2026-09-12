"""Contract tests for frozen EXP_002 utilities using synthetic data only."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import (  # noqa: E402
    BINARY_CLASS_ORDER,
    PREDICTION_THRESHOLD,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_subset_matches_frozen_split,
    binary_majority_class,
    build_binary_target,
    build_missing_ratio,
    directional_probability,
    evaluate_binary_predictions,
    make_binary_logistic_regression,
    primary_fold_decision,
    secondary_fold_decision,
    threshold_predictions,
)


def assert_raises_integrity(action, expected_message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert expected_message in str(error)
        return
    raise AssertionError("Expected an integrity assertion to be raised.")


def return_frame() -> pd.DataFrame:
    values = np.zeros((2, 53), dtype=float)
    values[0, [0, 10, 52]] = np.nan
    values[1, :] = np.nan
    return pd.DataFrame(values, columns=RETURN_COLUMNS, index=pd.Index([4, 9], name="row_id"))


def test_binary_target_and_missing_ratio_are_exact() -> None:
    reod = pd.Series([-1, 0, 1], index=[3, 5, 8], name="reod")
    target = build_binary_target(reod)
    assert target.tolist() == [1, 0, 1]
    assert target.name == "Z"
    assert_raises_integrity(lambda: build_binary_target(pd.Series([2])), "Unexpected")

    matrix = build_missing_ratio(return_frame())
    assert matrix.index.tolist() == [4, 9]
    assert matrix.loc[4, "missing_ratio"] == 3 / 53
    assert matrix.loc[9, "missing_ratio"] == 1.0


def test_shared_split_prevents_holdout_leakage_and_filtering() -> None:
    rows, labels, identifier = [], [], 0
    for day in range(4):
        for equity in range(5):
            rows.append({"ID": identifier, "day": day, "equity": equity})
            labels.append(0 if identifier % 2 == 0 else 1)
            identifier += 1
    features = pd.DataFrame(rows)
    target = pd.Series(labels, name="Z")
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, target, partition, FoldSpec(1, 0, 1, 2, 3))
    assert set(subsets.fit_features["equity"]).isdisjoint(
        set(partition.loc[partition["partition"].eq("E_holdout"), "equity"])
    )
    assert_subset_matches_frozen_split(subsets.joint_features, subsets.joint_features, subset_name="joint_oos")
    filtered = subsets.joint_features.iloc[1:]
    assert_raises_integrity(
        lambda: assert_subset_matches_frozen_split(filtered, subsets.joint_features, subset_name="joint_oos"),
        "frozen shared splitter",
    )

    structural_expected = pd.DataFrame({"ID": [10, 11], "day": [112, 134], "equity": [1, 2]})
    assert_subset_matches_frozen_split(
        structural_expected, structural_expected, subset_name="joint_oos"
    )
    assert_raises_integrity(
        lambda: assert_subset_matches_frozen_split(
            structural_expected.loc[structural_expected["day"].ne(112)],
            structural_expected,
            subset_name="joint_oos",
        ),
        "frozen shared splitter",
    )


def test_frozen_fold_definitions_are_reused() -> None:
    assert [(fold.fold, fold.train_start, fold.train_end, fold.oos_start, fold.oos_end) for fold in FOLDS] == [
        (1, 0, 302, 303, 352),
        (2, 0, 352, 353, 402),
        (3, 0, 402, 403, 452),
        (4, 0, 452, 453, 502),
    ]


def test_baseline_tie_probability_column_threshold_and_metrics() -> None:
    assert binary_majority_class(pd.Series([1, 0], name="Z")) == 0
    assert binary_majority_class(pd.Series([1, 1, 0], name="Z")) == 1
    assert BINARY_CLASS_ORDER == (0, 1)
    assert PREDICTION_THRESHOLD == 0.5

    classifier = make_binary_logistic_regression()
    matrix = pd.DataFrame({"missing_ratio": [0.0, 1.0, 0.0, 1.0]})
    classifier.fit(matrix, pd.Series([0, 1, 0, 1], name="Z"))
    probability = directional_probability(classifier, matrix)
    prediction = threshold_predictions(np.array([0.49, 0.50, 0.51]))
    assert prediction.tolist() == [0, 1, 1]
    assert_raises_integrity(lambda: threshold_predictions(probability, threshold=0.4), "exactly 0.5")

    target = pd.Series([0, 0, 1, 1], name="Z")
    metrics, matrix_confusion = evaluate_binary_predictions(
        target, np.array([0, 0, 1, 1]), directional_probability_values=np.array([0.1, 0.4, 0.6, 0.8])
    )
    assert matrix_confusion == [[2, 0], [0, 2]]
    assert metrics["roc_auc"] == 1.0
    assert_raises_integrity(
        lambda: evaluate_binary_predictions(pd.Series([0, 0]), np.array([0, 0])), "Both binary classes"
    )


def test_primary_and_secondary_decisions_are_separate_and_strict() -> None:
    primary = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54])
    assert primary["promising_oos_discrimination"]
    assert not primary_fold_decision(0.5)["roc_auc_above_one_half"]

    threshold_fail_metrics = {
        "balanced_accuracy": 0.5,
        "recall_neutral": 1.0,
        "recall_directional": 0.0,
    }
    secondary_folds = [secondary_fold_decision(threshold_fail_metrics) for _ in range(4)]
    secondary = aggregate_secondary_decision(secondary_folds, [0.01, 0.01, 0.01, 0.01])
    assert not secondary["fixed_threshold_classification_evidence"]
    assert primary["promising_oos_discrimination"] and not secondary["fixed_threshold_classification_evidence"]


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_binary_target_and_missing_ratio_are_exact,
            test_shared_split_prevents_holdout_leakage_and_filtering,
            test_frozen_fold_definitions_are_reused,
            test_baseline_tie_probability_column_threshold_and_metrics,
            test_primary_and_secondary_decisions_are_separate_and_strict,
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

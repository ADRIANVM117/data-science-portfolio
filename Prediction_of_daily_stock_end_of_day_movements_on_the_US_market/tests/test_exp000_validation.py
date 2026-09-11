"""Protocol tests for EXP_000 validation boundaries and baseline behavior."""

from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import (  # noqa: E402
    FoldSpec,
    assert_fold_integrity,
    build_equity_partition,
    evaluate_constant_baseline,
    majority_class,
    split_fold,
    validate_equity_partition,
    with_injected_fit_row,
)


def synthetic_data() -> tuple[pd.DataFrame, pd.Series, FoldSpec]:
    rows = []
    labels = []
    identifier = 0
    for day in range(4):
        for equity in range(5):
            rows.append({"ID": identifier, "day": day, "equity": equity, "r0": float(identifier)})
            labels.append([-1, 0, 1][identifier % 3])
            identifier += 1
    features = pd.DataFrame(rows)
    target = pd.Series(labels, name="reod")
    return features, target, FoldSpec(1, 0, 1, 2, 3)


def assert_raises_integrity(action, expected_message: str) -> None:
    """Dependency-free assertion helper for protocol-violation tests."""
    try:
        action()
    except AssertionError as error:
        assert expected_message in str(error)
        return
    raise AssertionError("Expected an integrity assertion to be raised.")


def test_partition_is_deterministic_and_complete() -> None:
    first = build_equity_partition(range(10), seed=20260908, n_holdout=2)
    second = build_equity_partition(range(10), seed=20260908, n_holdout=2)
    pd.testing.assert_frame_equal(first, second)
    validate_equity_partition(first, range(10), n_holdout=2)


def test_fold_subsets_are_disjoint_and_follow_boundaries() -> None:
    features, target, fold = synthetic_data()
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, target, partition, fold)

    assert set(subsets.fit_features["day"]) == {0, 1}
    assert set(subsets.temporal_features["day"]) == {2, 3}
    assert set(subsets.joint_features["day"]) == {2, 3}
    assert set(subsets.fit_features["ID"]).isdisjoint(subsets.temporal_features["ID"])
    assert set(subsets.fit_features["ID"]).isdisjoint(subsets.joint_features["ID"])


def test_holdout_equity_or_oos_day_in_fit_fails_integrity() -> None:
    features, target, fold = synthetic_data()
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, target, partition, fold)
    dev = set(partition.loc[partition["partition"] == "E_dev", "equity"])
    holdout = set(partition.loc[partition["partition"] == "E_holdout", "equity"])

    holdout_row = subsets.joint_features.iloc[[0]]
    contaminated = with_injected_fit_row(subsets, holdout_row, subsets.joint_target.iloc[[0]])
    assert_raises_integrity(
        lambda: assert_fold_integrity(contaminated, dev, holdout),
        "Holdout equity",
    )

    future_dev_row = subsets.temporal_features.iloc[[0]]
    contaminated = with_injected_fit_row(subsets, future_dev_row, subsets.temporal_target.iloc[[0]])
    assert_raises_integrity(
        lambda: assert_fold_integrity(contaminated, dev, holdout),
        "Fit contains an OOS day",
    )


def test_majority_tie_uses_contract_order_and_metrics_keep_all_classes() -> None:
    tied_target = pd.Series([0, -1], name="reod")
    assert majority_class(tied_target) == -1

    metrics, matrix = evaluate_constant_baseline(pd.Series([-1, 0], name="reod"), predicted_class=0)
    assert set(metrics) == {"accuracy", "macro_f1", "balanced_accuracy"}
    assert matrix == [[0, 1, 0], [0, 1, 0], [0, 0, 0]]


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(test)
        for test in (
            test_partition_is_deterministic_and_complete,
            test_fold_subsets_are_disjoint_and_follow_boundaries,
            test_holdout_equity_or_oos_day_in_fit_fails_integrity,
            test_majority_tie_uses_contract_order_and_metrics_keep_all_classes,
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

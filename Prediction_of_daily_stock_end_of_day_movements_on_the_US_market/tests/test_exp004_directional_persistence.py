"""Synthetic contract tests for frozen EXP_004 directional persistence."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp004_directional_persistence import (  # noqa: E402
    aggregate_primary_decision, aggregate_secondary_decision, coefficient_sign,
    coefficient_sign_consistent, evaluate_sign_predictions,
    make_persistence_logistic_regression, positive_probability,
    prepare_evaluable_persistence_subset, primary_fold_decision,
    secondary_fold_decision, sign_majority_class, threshold_predictions,
    validate_persistence_matrix,
)


def assert_raises(action, message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert message in str(error)
        return
    raise AssertionError("Expected an integrity assertion.")


def feature_frame(paths: list[list[float]]) -> pd.DataFrame:
    values = np.full((len(paths), 53), np.nan)
    for row, path in enumerate(paths):
        values[row, :len(path)] = path
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", range(1, len(paths) + 1))
    frame.insert(0, "day", 0)
    frame.insert(0, "ID", range(100, 100 + len(paths)))
    return frame


def test_exact_persistence_counts_zero_nan_and_all_nan_exclusion() -> None:
    features = feature_frame([[1, 2], [-1, -3], [1, -1], [1, 0, -1, 0], [], [0], [1, 0]])
    reod = pd.Series([-1, 1, -1, 1, 1, 0, 1], name="reod")
    prepared = prepare_evaluable_persistence_subset(features, reod)
    assert prepared.feature_matrix.columns.tolist() == ["P"]
    assert prepared.feature_matrix["P"].tolist() == [1.0, -1.0, 0.0, 0.0, 0.5]
    assert prepared.sign_target.tolist() == [0, 1, 0, 1, 1]
    assert 4 not in prepared.feature_matrix.index
    assert 5 not in prepared.feature_matrix.index
    assert prepared.manifest == {
        "n_directional_before_evaluability": 6,
        "n_all_nan_directional_excluded": 1,
        "n_evaluable_directional_retained": 5,
        "n_negative_before_evaluability": 2,
        "n_positive_before_evaluability": 4,
        "n_negative_after_evaluability": 2,
        "n_positive_after_evaluability": 3,
    }
    # [+, 0] has N_obs=2 and P=0.5: observed zeros dilute |P|.
    assert prepared.feature_matrix.loc[6, "P"] == 0.5
    assert_raises(lambda: validate_persistence_matrix(pd.DataFrame({"P": [0.0], "N_obs": [1]})), "exactly P")


def test_frozen_split_and_no_preprocessing_boundary() -> None:
    rows, labels, identifier = [], [], 0
    for day in range(4):
        for equity in range(5):
            row = {"ID": identifier, "day": day, "equity": equity}
            row.update({column: 1.0 for column in RETURN_COLUMNS})
            rows.append(row)
            labels.append(-1 if identifier % 2 == 0 else 1)
            identifier += 1
    features = pd.DataFrame(rows)
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, pd.Series(labels, name="reod"), partition, FoldSpec(1, 0, 1, 2, 3))
    assert set(subsets.fit_features["equity"]).isdisjoint(set(partition.loc[partition["partition"].eq("E_holdout"), "equity"]))
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in FOLDS] == [(1, 0, 302, 303, 352), (2, 0, 352, 353, 402), (3, 0, 402, 403, 452), (4, 0, 452, 453, 502)]
    source = (PROJECT_ROOT / "src" / "exp004_directional_persistence.py").read_text(encoding="utf-8")
    assert "StandardScaler" not in source


def test_model_probability_threshold_metrics_baseline_and_integrity() -> None:
    assert sign_majority_class(pd.Series([0, 1], name="S")) == 0
    classifier = make_persistence_logistic_regression()
    matrix = pd.DataFrame({"P": [-1.0, 1.0, -1.0, 1.0]})
    classifier.fit(matrix, pd.Series([0, 1, 0, 1], name="S"))
    probability = positive_probability(classifier, matrix)
    assert threshold_predictions(np.array([0.49, 0.5, 0.51])).tolist() == [0, 1, 1]
    assert_raises(lambda: threshold_predictions(probability, threshold=0.4), "exactly 0.5")
    metrics, confusion = evaluate_sign_predictions(pd.Series([0, 0, 1, 1]), np.array([0, 0, 1, 1]), probability=np.array([0.1, 0.4, 0.6, 0.8]))
    assert confusion == [[2, 0], [0, 2]] and metrics["roc_auc"] == 1.0
    assert_raises(lambda: evaluate_sign_predictions(pd.Series([1, 1]), np.array([1, 1])), "Both sign classes")


def test_decisions_coefficients_and_no_competition_reference() -> None:
    assert aggregate_primary_decision([0.51, 0.52, 0.53, 0.54])["promising_oos_sign_discrimination"]
    assert not primary_fold_decision(0.5)["roc_auc_above_one_half"]
    failed = {"balanced_accuracy": 0.5, "recall_negative": 1.0, "recall_positive": 0.0}
    assert not aggregate_secondary_decision([secondary_fold_decision(failed)] * 4, [0.01] * 4)["fixed_threshold_sign_classification_evidence"]
    assert coefficient_sign(1.0) == "positive" and coefficient_sign(-1.0) == "negative" and coefficient_sign(0.0) == "zero"
    assert coefficient_sign_consistent([0.1, 0.2, 0.3, 0.4])
    assert not coefficient_sign_consistent([0.1, 0.0, 0.3, 0.4])
    runner = PROJECT_ROOT / "scripts" / "run_exp_004_directional_persistence.py"
    assert runner.exists()
    assert "input_test" not in runner.read_text(encoding="utf-8")


if __name__ == "__main__":
    import unittest
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_exact_persistence_counts_zero_nan_and_all_nan_exclusion,
        test_frozen_split_and_no_preprocessing_boundary,
        test_model_probability_threshold_metrics_baseline_and_integrity,
        test_decisions_coefficients_and_no_competition_reference,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

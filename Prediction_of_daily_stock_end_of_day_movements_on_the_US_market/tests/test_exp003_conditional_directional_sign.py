"""Synthetic contract tests for frozen EXP_003 utilities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp000_validation import FOLDS, FoldSpec, build_equity_partition, split_fold  # noqa: E402
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp003_conditional_directional_sign import (  # noqa: E402
    aggregate_primary_decision, aggregate_secondary_decision, build_sign_target,
    coefficient_sign, coefficient_sign_consistent, evaluate_sign_predictions,
    fit_scaler_on_fit, make_sign_logistic_regression, positive_probability,
    prepare_evaluable_directional_subset, primary_fold_decision,
    secondary_fold_decision, sign_majority_class, threshold_predictions,
    transform_with_fit_scaler,
)


def assert_raises(action, message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert message in str(error)
        return
    raise AssertionError("Expected an integrity assertion.")


def features_for_evaluability() -> pd.DataFrame:
    values = np.full((4, 53), np.nan)
    values[0, :] = 0.0
    values[1, 0] = 10.0
    values[1, 1] = -3.0
    values[3, 0] = 0.0
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", [1, 1, 2, 2])
    frame.insert(0, "day", [0, 0, 0, 0])
    frame.insert(0, "ID", [10, 11, 12, 13])
    return frame


def test_conditioning_evaluability_and_r_obs_are_exact() -> None:
    reod = pd.Series([-1, 1, 1, 0], index=features_for_evaluability().index, name="reod")
    prepared = prepare_evaluable_directional_subset(features_for_evaluability(), reod)
    assert prepared.sign_target.tolist() == [0, 1]
    assert prepared.feature_matrix.columns.tolist() == ["R_obs"]
    assert prepared.feature_matrix["R_obs"].tolist() == [0.0, 7.0]
    assert prepared.manifest == {
        "n_directional_before_evaluability": 3,
        "n_all_nan_directional_excluded": 1,
        "n_evaluable_directional_retained": 2,
        "n_negative_before_evaluability": 1,
        "n_positive_before_evaluability": 2,
        "n_negative_after_evaluability": 1,
        "n_positive_after_evaluability": 1,
    }
    assert 2 not in prepared.feature_matrix.index
    assert_raises(lambda: build_sign_target(pd.Series([-1, 0])), "requires only")


def test_frozen_split_and_scaler_boundaries() -> None:
    rows, labels, identifier = [], [], 0
    for day in range(4):
        for equity in range(5):
            row = {"ID": identifier, "day": day, "equity": equity}
            row.update({column: float(identifier) for column in RETURN_COLUMNS})
            rows.append(row)
            labels.append(-1 if identifier % 2 == 0 else 1)
            identifier += 1
    features = pd.DataFrame(rows)
    partition = build_equity_partition(features["equity"], n_holdout=1)
    subsets = split_fold(features, pd.Series(labels, name="reod"), partition, FoldSpec(1, 0, 1, 2, 3))
    assert set(subsets.fit_features["equity"]).isdisjoint(set(partition.loc[partition["partition"].eq("E_holdout"), "equity"]))
    assert [(f.fold, f.train_start, f.train_end, f.oos_start, f.oos_end) for f in FOLDS] == [(1, 0, 302, 303, 352), (2, 0, 352, 353, 402), (3, 0, 402, 403, 452), (4, 0, 452, 453, 502)]

    fit = pd.DataFrame({"R_obs": [1.0, 3.0]})
    scaler, scaled_fit = fit_scaler_on_fit(fit)
    transformed = transform_with_fit_scaler(scaler, pd.DataFrame({"R_obs": [101.0]}))
    assert scaled_fit["R_obs"].tolist() == [-1.0, 1.0]
    assert transformed.iloc[0, 0] == 99.0


def test_model_probability_threshold_metrics_and_baseline() -> None:
    assert sign_majority_class(pd.Series([0, 1], name="S")) == 0
    assert sign_majority_class(pd.Series([1, 1, 0], name="S")) == 1
    classifier = make_sign_logistic_regression()
    matrix = pd.DataFrame({"R_obs": [-1.0, 1.0, -1.0, 1.0]})
    classifier.fit(matrix, pd.Series([0, 1, 0, 1], name="S"))
    probability = positive_probability(classifier, matrix)
    assert threshold_predictions(np.array([0.49, 0.5, 0.51])).tolist() == [0, 1, 1]
    assert_raises(lambda: threshold_predictions(probability, threshold=0.4), "exactly 0.5")
    metrics, confusion = evaluate_sign_predictions(pd.Series([0, 0, 1, 1]), np.array([0, 0, 1, 1]), probability=np.array([0.1, 0.4, 0.6, 0.8]))
    assert confusion == [[2, 0], [0, 2]] and metrics["roc_auc"] == 1.0
    assert_raises(lambda: evaluate_sign_predictions(pd.Series([1, 1]), np.array([1, 1])), "Both sign classes")


def test_decisions_and_coefficient_diagnostic_are_independent() -> None:
    assert aggregate_primary_decision([0.51, 0.52, 0.53, 0.54])["promising_oos_sign_discrimination"]
    assert not primary_fold_decision(0.5)["roc_auc_above_one_half"]
    failed = {"balanced_accuracy": 0.5, "recall_negative": 1.0, "recall_positive": 0.0}
    secondary = aggregate_secondary_decision([secondary_fold_decision(failed)] * 4, [0.01] * 4)
    assert not secondary["fixed_threshold_sign_classification_evidence"]
    assert coefficient_sign(1.0) == "positive" and coefficient_sign(-1.0) == "negative" and coefficient_sign(0.0) == "zero"
    assert coefficient_sign_consistent([0.1, 0.2, 0.3, 0.4])
    assert not coefficient_sign_consistent([0.1, 0.2, 0.0, 0.4])


def test_runner_has_no_competition_test_reference() -> None:
    source = (PROJECT_ROOT / "scripts" / "run_exp_003_conditional_directional_sign.py").read_text(encoding="utf-8")
    assert "input_test" not in source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_conditioning_evaluability_and_r_obs_are_exact,
        test_frozen_split_and_scaler_boundaries,
        test_model_probability_threshold_metrics_and_baseline,
        test_decisions_and_coefficient_diagnostic_are_independent,
        test_runner_has_no_competition_test_reference,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

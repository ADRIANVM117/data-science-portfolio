"""Synthetic tests for frozen D004; never loads physical Discovery data."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d003_incremental_cross_sectional_relative_path import assert_e_dev_only, validate_discovery_features  # noqa: E402
from discovery.d004_hierarchical_recent_intensity_ternary_decision import (  # noqa: E402
    RELIABILITY_LABELS,
    build_d004_matrix,
    calibration_diagnostics,
    directional_probability,
    fit_directional_priors,
    fit_intensity_transformer,
    fit_majority_predictions,
    hierarchical_predictions,
    hierarchical_probabilities,
    ternary_metrics,
)
from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp002_neutral_directional import build_binary_target, make_binary_logistic_regression  # noqa: E402


def assert_raises(action, phrase: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert phrase in str(error)
        return
    raise AssertionError("Expected an AssertionError.")


def synthetic_features(n: int = 12) -> pd.DataFrame:
    values = np.arange(n * 53, dtype=float).reshape(n, 53) / 100.0
    values[1, 41:53] = np.nan
    values[3, 45] = np.nan
    frame = pd.DataFrame(values, columns=RETURN_COLUMNS)
    frame.insert(0, "equity", np.arange(1, n + 1))
    frame.insert(0, "day", np.arange(n) % 3)
    frame.insert(0, "ID", np.arange(100, 100 + n))
    return frame


def test_priors_thresholds_and_strict_ties() -> None:
    equal = fit_directional_priors(pd.Series([-1, 1, 0, 0]))
    assert equal.pi_minus + equal.pi_plus == 1.0
    assert np.isclose(equal.threshold, 2.0 / 3.0)
    assert hierarchical_predictions(np.array([equal.threshold]), equal).tolist() == [0]
    above = np.nextafter(equal.threshold, np.inf)
    assert hierarchical_predictions(np.array([above]), equal).tolist() == [-1]
    uneven = fit_directional_priors(pd.Series([-1] * 11 + [1] * 9))
    assert np.isclose(uneven.threshold, 1.0 / 1.55)
    assert hierarchical_predictions(np.array([0.8]), uneven).tolist() == [-1]
    plus = fit_directional_priors(pd.Series([-1] * 4 + [1] * 6))
    assert hierarchical_predictions(np.array([0.8]), plus).tolist() == [1]


def test_probabilities_baseline_and_reliability_boundaries() -> None:
    priors = fit_directional_priors(pd.Series([-1] * 11 + [1] * 9))
    probabilities = hierarchical_probabilities(np.array([0.0, 0.5, 1.0]), priors)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    selected, _ = fit_majority_predictions(pd.Series([-1, 0, 1]), 2)
    assert selected == -1
    target = pd.Series([0, 0, 1, 1], dtype="int8")
    _, table = calibration_diagnostics(target, np.array([0.0, 0.1, 0.9, 1.0]))
    assert tuple(table["bin"]) == RELIABILITY_LABELS
    assert table.loc[0, "n_rows"] == 1 and table.loc[1, "n_rows"] == 1 and table.loc[9, "n_rows"] == 2


def test_probability_column_is_located_by_class_value() -> None:
    class Stub:
        classes_ = np.array([1, 0])
        def predict_proba(self, matrix):
            return np.tile(np.array([0.7, 0.3]), (len(matrix), 1))
    matrix = pd.DataFrame({"x": [0.0, 1.0]})
    assert np.allclose(directional_probability(Stub(), matrix), [0.7, 0.7])


def test_fit_only_preprocessing_and_validation_labels_do_not_change_fit_quantities() -> None:
    features = synthetic_features()
    fit_features, validation_features = features.iloc[:8], features.iloc[8:]
    fit_reod = pd.Series([-1, 0, 1, 0, -1, 1, 0, -1], index=fit_features.index)
    validation_reod = pd.Series([0, 1, -1, 0], index=validation_features.index)
    _, transformer = fit_intensity_transformer(fit_features)
    median, mean, scale = transformer.fit_intensity_median_, transformer.scaler_.mean_.copy(), transformer.scaler_.scale_.copy()
    matrix_one, _ = build_d004_matrix(validation_features, validation_reod, transformer)
    changed_validation = pd.Series([-1, -1, -1, -1], index=validation_features.index)
    matrix_two, _ = build_d004_matrix(validation_features, changed_validation, transformer)
    assert matrix_one.equals(matrix_two)
    assert transformer.fit_intensity_median_ == median and np.array_equal(transformer.scaler_.mean_, mean) and np.array_equal(transformer.scaler_.scale_, scale)
    priors_one, priors_two = fit_directional_priors(fit_reod), fit_directional_priors(fit_reod)
    assert priors_one == priors_two


def test_protected_boundaries_and_end_to_end_synthetic_fold() -> None:
    features = synthetic_features()
    validate_discovery_features(features)
    partition = pd.DataFrame({"equity": list(range(1, 13)) + [99], "partition": ["E_dev"] * 12 + ["E_holdout"]})
    assert_e_dev_only(features, partition)
    bad_day = features.copy(); bad_day.loc[0, "day"] = 353
    assert_raises(lambda: validate_discovery_features(bad_day), "days outside")
    bad_equity = features.copy(); bad_equity.loc[0, "equity"] = 99
    assert_raises(lambda: assert_e_dev_only(bad_equity, partition), "non-E_dev")

    fit_features, validation_features = features.iloc[:8], features.iloc[8:]
    fit_reod = pd.Series([-1, 0, 1, 0, -1, 1, 0, -1], index=fit_features.index)
    validation_reod = pd.Series([0, 1, -1, 0], index=validation_features.index)
    _, transformer = fit_intensity_transformer(fit_features)
    fit_matrix, _ = build_d004_matrix(fit_features, fit_reod, transformer)
    validation_matrix, _ = build_d004_matrix(validation_features, validation_reod, transformer)
    model = make_binary_logistic_regression().fit(fit_matrix, build_binary_target(fit_reod))
    p_d = directional_probability(model, validation_matrix)
    priors = fit_directional_priors(fit_reod)
    candidate = hierarchical_predictions(p_d, priors)
    baseline_class, baseline = fit_majority_predictions(fit_reod, len(validation_reod))
    metrics, matrix = ternary_metrics(validation_reod, candidate)
    calibration, reliability = calibration_diagnostics(build_binary_target(validation_reod), p_d)
    assert baseline_class in (-1, 0, 1) and len(baseline) == len(candidate) == len(validation_reod)
    assert set(metrics) == {"accuracy", "balanced_accuracy", "macro_f1", "recall_negative", "recall_neutral", "recall_positive"}
    assert len(matrix) == 3 and np.isfinite(list(calibration.values())).all() and reliability["n_rows"].sum() == len(validation_reod)


if __name__ == "__main__":
    import unittest
    tests = (
        test_priors_thresholds_and_strict_ties,
        test_probabilities_baseline_and_reliability_boundaries,
        test_probability_column_is_located_by_class_value,
        test_fit_only_preprocessing_and_validation_labels_do_not_change_fit_quantities,
        test_protected_boundaries_and_end_to_end_synthetic_fold,
    )
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests))
    raise SystemExit(not result.wasSuccessful())

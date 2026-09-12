"""Synthetic contract tests for frozen EXP_006 nonlinear positional paths."""

from pathlib import Path
import os
import sys

# The Windows sandbox reports zero physical cores to joblib. This affects only
# synthetic test execution; HGB's frozen estimator parameters remain unchanged.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402
from src.exp003_conditional_directional_sign import (  # noqa: E402
    evaluate_sign_predictions,
    positive_probability,
    threshold_predictions,
)
from src.exp005_positional_path_signal import (  # noqa: E402
    PATH_MASK_COLUMNS,
    PositionalPathTransformer,
    assert_identical_representation_rows,
    build_original_mask,
    build_path_mask,
)
from src.exp006_nonlinear_positional_path_signal import (  # noqa: E402
    HGB_PARAMS,
    REQUIRED_SKLEARN_VERSION,
    aggregate_primary_decision,
    aggregate_secondary_decision,
    assert_independent_estimators,
    assert_sklearn_version,
    make_hist_gradient_boosting_classifier,
    primary_fold_decision,
    secondary_fold_decision,
)


def assert_raises(action, message: str) -> None:
    try:
        action()
    except AssertionError as error:
        assert message in str(error)
        return
    raise AssertionError("Expected integrity assertion.")


def synthetic_raw_returns() -> pd.DataFrame:
    values = np.zeros((4, len(RETURN_COLUMNS)), dtype=float)
    values[0, 0] = np.nan
    values[1, 1] = np.nan
    values[2, 2] = np.nan
    values[3, 3] = np.nan
    return pd.DataFrame(values, columns=RETURN_COLUMNS, index=[10, 20, 30, 40])


def test_exact_hgb_class_parameters_version_and_early_stopping() -> None:
    assert sklearn.__version__ == REQUIRED_SKLEARN_VERSION == "1.6.1"
    assert_sklearn_version()
    estimator = make_hist_gradient_boosting_classifier()
    assert isinstance(estimator, HistGradientBoostingClassifier)
    assert estimator.get_params() == dict(HGB_PARAMS)
    assert estimator.get_params()["early_stopping"] is False


def test_reused_exp005_representation_is_finite_and_aligned() -> None:
    raw = synthetic_raw_returns()
    target = pd.Series([0, 1, 0, 1], index=raw.index, name="S", dtype="int8")
    transformer = PositionalPathTransformer().fit(raw)
    a_mask = build_original_mask(raw)
    b_path_mask = build_path_mask(transformer, raw)
    assert a_mask.shape == (4, 53)
    assert b_path_mask.shape == (4, 106)
    assert list(b_path_mask.columns) == list(PATH_MASK_COLUMNS)
    assert np.isfinite(b_path_mask.to_numpy(dtype=float)).all()
    assert_identical_representation_rows(a_mask, b_path_mask, target)


def test_independent_models_probability_threshold_and_confusion_order() -> None:
    raw = synthetic_raw_returns()
    transformer = PositionalPathTransformer().fit(raw)
    a_matrix = build_original_mask(raw)
    b_matrix = build_path_mask(transformer, raw)
    target = pd.Series([0, 1, 0, 1], index=raw.index, name="S")
    a_estimator = make_hist_gradient_boosting_classifier()
    b_estimator = make_hist_gradient_boosting_classifier()
    assert_independent_estimators(a_estimator, b_estimator)
    assert_raises(lambda: assert_independent_estimators(a_estimator, a_estimator), "independent estimators")
    a_estimator.fit(a_matrix, target)
    b_estimator.fit(b_matrix, target)
    assert tuple(a_estimator.classes_) == (0, 1)
    assert tuple(b_estimator.classes_) == (0, 1)
    probability = positive_probability(b_estimator, b_matrix)
    assert len(probability) == len(target) and np.isfinite(probability).all()
    assert threshold_predictions(np.array([0.49, 0.5, 0.51])).tolist() == [0, 1, 1]
    assert_raises(lambda: threshold_predictions(probability, threshold=0.4), "exactly 0.5")
    metrics, confusion = evaluate_sign_predictions(
        pd.Series([0, 0, 1, 1]), np.array([0, 0, 1, 1]), probability=np.array([0.1, 0.4, 0.6, 0.8])
    )
    assert metrics["roc_auc"] == 1.0 and confusion == [[2, 0], [0, 2]]


def test_strict_primary_and_independent_secondary_decisions() -> None:
    passed = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.02, 0.03, 0.04])
    assert passed["incremental_nonlinear_positional_return_evidence"]
    failed_delta = aggregate_primary_decision([0.51, 0.52, 0.53, 0.54], [0.01, 0.0, 0.03, 0.04])
    assert not failed_delta["incremental_nonlinear_positional_return_evidence"]
    assert not primary_fold_decision(0.5, 0.01)["path_roc_auc_above_one_half"]
    failed_metrics = {"balanced_accuracy": 0.5, "recall_negative": 1.0, "recall_positive": 0.0}
    assert not aggregate_secondary_decision([secondary_fold_decision(failed_metrics)] * 4, [0.01] * 4)[
        "fixed_threshold_path_sign_classification_evidence"
    ]


def test_runner_has_no_competition_or_exp005_metric_dependency() -> None:
    runner = PROJECT_ROOT / "scripts" / "run_exp_006_nonlinear_positional_path_signal.py"
    source = runner.read_text(encoding="utf-8")
    assert "input_test" not in source
    assert "EXP_005_metrics" not in source
    assert "EXP_005_summary" not in source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_exact_hgb_class_parameters_version_and_early_stopping,
        test_reused_exp005_representation_is_finite_and_aligned,
        test_independent_models_probability_threshold_and_confusion_order,
        test_strict_primary_and_independent_secondary_decisions,
        test_runner_has_no_competition_or_exp005_metric_dependency,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

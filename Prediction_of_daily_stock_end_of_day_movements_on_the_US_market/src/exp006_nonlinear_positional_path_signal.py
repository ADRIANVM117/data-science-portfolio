"""Frozen EXP_006 HistGradientBoosting utilities."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from src.exp001_missingness import strictly_greater


REQUIRED_SKLEARN_VERSION = "1.6.1"
HGB_PARAMS: Mapping[str, object] = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "max_bins": 255,
    "categorical_features": "from_dtype",
    "monotonic_cst": None,
    "interaction_cst": None,
    "warm_start": False,
    "early_stopping": False,
    "scoring": "loss",
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "tol": 1e-7,
    "verbose": 0,
    "random_state": 20260908,
    "class_weight": None,
}


def assert_sklearn_version() -> None:
    if sklearn.__version__ != REQUIRED_SKLEARN_VERSION:
        raise AssertionError(
            f"EXP_006 requires scikit-learn {REQUIRED_SKLEARN_VERSION}, found {sklearn.__version__}."
        )


def make_hist_gradient_boosting_classifier() -> HistGradientBoostingClassifier:
    """Build and validate the exact frozen EXP_006 HGB estimator."""
    assert_sklearn_version()
    estimator = HistGradientBoostingClassifier(**HGB_PARAMS)
    if not isinstance(estimator, HistGradientBoostingClassifier):
        raise AssertionError("EXP_006 estimator must be HistGradientBoostingClassifier.")
    parameters = estimator.get_params()
    for name, expected in HGB_PARAMS.items():
        if parameters[name] != expected:
            raise AssertionError(f"EXP_006 HGB parameter {name!r} differs from the frozen contract.")
    if parameters["early_stopping"] is not False:
        raise AssertionError("EXP_006 forbids internal early stopping.")
    return estimator


def assert_independent_estimators(
    a_mask_estimator: HistGradientBoostingClassifier,
    b_path_mask_estimator: HistGradientBoostingClassifier,
) -> None:
    if a_mask_estimator is b_path_mask_estimator:
        raise AssertionError("EXP_006 A_mask and B_path_mask require independent estimators.")


def primary_fold_decision(path_auc: float, path_minus_mask_auc: float) -> dict[str, bool]:
    return {
        "path_roc_auc_above_one_half": strictly_greater(path_auc, 0.5),
        "path_minus_mask_auc_positive": strictly_greater(path_minus_mask_auc, 0.0),
    }


def aggregate_primary_decision(path_aucs: Sequence[float], path_minus_mask_aucs: Sequence[float]) -> dict[str, float | bool | int]:
    if len(path_aucs) != 4 or len(path_minus_mask_aucs) != 4:
        raise AssertionError("EXP_006 requires exactly four Joint OOS folds.")
    mean_auc = float(np.mean(path_aucs))
    mean_delta = float(np.mean(path_minus_mask_aucs))
    all_auc = all(strictly_greater(value, 0.5) for value in path_aucs)
    all_delta = all(strictly_greater(value, 0.0) for value in path_minus_mask_aucs)
    return {
        "n_joint_folds": 4,
        "all_four_path_roc_auc_above_one_half": all_auc,
        "mean_joint_path_roc_auc": mean_auc,
        "mean_joint_path_roc_auc_above_one_half": strictly_greater(mean_auc, 0.5),
        "all_four_path_minus_mask_auc_positive": all_delta,
        "mean_joint_path_minus_mask_auc": mean_delta,
        "mean_joint_path_minus_mask_auc_positive": strictly_greater(mean_delta, 0.0),
        "incremental_nonlinear_positional_return_evidence": (
            all_auc
            and strictly_greater(mean_auc, 0.5)
            and all_delta
            and strictly_greater(mean_delta, 0.0)
        ),
    }


def secondary_fold_decision(path_metrics: Mapping[str, float]) -> dict[str, bool]:
    decision = {
        "balanced_accuracy_above_one_half": strictly_greater(path_metrics["balanced_accuracy"], 0.5),
        "recall_negative_positive": path_metrics["recall_negative"] > 0.0,
        "recall_positive_positive": path_metrics["recall_positive"] > 0.0,
    }
    decision["all_fold_conditions_met"] = all(decision.values())
    return decision


def aggregate_secondary_decision(
    fold_decisions: Sequence[Mapping[str, bool]], path_ba_deltas: Sequence[float]
) -> dict[str, float | bool | int]:
    if len(fold_decisions) != 4 or len(path_ba_deltas) != 4:
        raise AssertionError("EXP_006 requires exactly four Joint OOS folds.")
    mean_delta = float(np.mean(path_ba_deltas))
    all_conditions = all(bool(decision["all_fold_conditions_met"]) for decision in fold_decisions)
    return {
        "n_joint_folds": 4,
        "all_four_fixed_threshold_conditions_met": all_conditions,
        "mean_joint_path_balanced_accuracy_delta_vs_baseline": mean_delta,
        "mean_joint_path_balanced_accuracy_delta_positive": strictly_greater(mean_delta, 0.0),
        "fixed_threshold_path_sign_classification_evidence": (
            all_conditions and strictly_greater(mean_delta, 0.0)
        ),
    }

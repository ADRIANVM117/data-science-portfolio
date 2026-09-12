"""Utilities for frozen EXP_004 conditional directional persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.exp001_missingness import COMPARISON_ATOL, LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS, strictly_greater
from src.exp003_conditional_directional_sign import (
    SIGN_CLASS_ORDER,
    assert_both_sign_classes,
    build_sign_target,
    coefficient_beta,
    coefficient_sign,
    coefficient_sign_consistent,
    evaluate_sign_majority_baseline,
    evaluate_sign_predictions,
    positive_probability,
    sign_majority_class,
    threshold_predictions,
)


DIRECTIONAL_REOD_VALUES = (-1, 1)


@dataclass(frozen=True)
class EvaluablePersistenceSubset:
    """One frozen subset after directional conditioning and N_obs evaluability."""

    feature_matrix: pd.DataFrame
    sign_target: pd.Series
    manifest: dict[str, int]


def _validate_return_columns(frame: pd.DataFrame) -> None:
    missing = set(RETURN_COLUMNS).difference(frame.columns)
    if missing:
        raise AssertionError(f"Missing EXP_004 return columns: {sorted(missing)}")


def prepare_evaluable_persistence_subset(features: pd.DataFrame, reod: pd.Series) -> EvaluablePersistenceSubset:
    """Apply D then N_obs>=1, and construct the one allowed persistence feature."""
    if not features.index.equals(reod.index):
        raise AssertionError("Features and reod must have identical indices.")
    _validate_return_columns(features)
    if reod.isna().any() or not set(reod.unique()).issubset({-1, 0, 1}):
        raise AssertionError("Unexpected reod labels before EXP_004 conditioning.")

    directional = reod.isin(DIRECTIONAL_REOD_VALUES)
    directional_features = features.loc[directional]
    directional_reod = reod.loc[directional]
    returns = directional_features.loc[:, RETURN_COLUMNS]
    n_plus = returns.gt(0).sum(axis=1).astype("int8")
    n_minus = returns.lt(0).sum(axis=1).astype("int8")
    n_zero = returns.eq(0).sum(axis=1).astype("int8")
    n_obs = (n_plus + n_minus + n_zero).astype("int8")
    evaluable = n_obs.ge(1)

    retained_reod = directional_reod.loc[evaluable]
    retained_n_plus = n_plus.loc[evaluable]
    retained_n_minus = n_minus.loc[evaluable]
    retained_n_obs = n_obs.loc[evaluable]
    # P is calculated only after excluding all-NaN directional rows.
    persistence = ((retained_n_plus - retained_n_minus) / retained_n_obs).astype("float64")
    matrix = pd.DataFrame({"P": persistence}, index=retained_reod.index)
    target = build_sign_target(retained_reod)
    validate_persistence_matrix(matrix, expected_index=target.index)

    before = directional_reod.value_counts().reindex(DIRECTIONAL_REOD_VALUES, fill_value=0)
    after = retained_reod.value_counts().reindex(DIRECTIONAL_REOD_VALUES, fill_value=0)
    manifest = {
        "n_directional_before_evaluability": int(len(directional_reod)),
        "n_all_nan_directional_excluded": int((~evaluable).sum()),
        "n_evaluable_directional_retained": int(len(retained_reod)),
        "n_negative_before_evaluability": int(before.loc[-1]),
        "n_positive_before_evaluability": int(before.loc[1]),
        "n_negative_after_evaluability": int(after.loc[-1]),
        "n_positive_after_evaluability": int(after.loc[1]),
    }
    if manifest["n_directional_before_evaluability"] != manifest["n_all_nan_directional_excluded"] + manifest["n_evaluable_directional_retained"]:
        raise AssertionError("EXP_004 evaluability counts do not reconcile.")
    if not retained_n_obs.ge(1).all() or not retained_reod.isin(DIRECTIONAL_REOD_VALUES).all():
        raise AssertionError("Non-evaluable or neutral row retained in EXP_004.")
    return EvaluablePersistenceSubset(matrix, target, manifest)


def validate_persistence_matrix(matrix: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    """Assert that P is the sole finite bounded predictive feature."""
    if list(matrix.columns) != ["P"]:
        raise AssertionError("EXP_004 predictive matrix must contain exactly P.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("P changed row identity or order.")
    values = matrix.to_numpy(dtype=float)
    if matrix.isna().any().any() or not np.isfinite(values).all() or ((values < -1.0) | (values > 1.0)).any():
        raise AssertionError("P must be finite and bounded in [-1, 1].")


def make_persistence_logistic_regression() -> LogisticRegression:
    """Return EXP_004's frozen, unscaled LogisticRegression estimator."""
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    for name, expected in LOGISTIC_REGRESSION_PARAMS.items():
        if classifier.get_params()[name] != expected:
            raise AssertionError(f"LogisticRegression parameter {name!r} differs from EXP_004.")
    return classifier


def persistence_beta(classifier: LogisticRegression) -> float:
    """Extract the sole unscaled persistence coefficient."""
    return coefficient_beta(classifier)


def delta_vs_baseline(value: float, reference: float) -> float:
    delta = value - reference
    return 0.0 if np.isclose(delta, 0.0, rtol=0.0, atol=COMPARISON_ATOL) else float(delta)


def primary_fold_decision(roc_auc: float) -> dict[str, bool]:
    return {"roc_auc_above_one_half": strictly_greater(roc_auc, 0.5)}


def secondary_fold_decision(metrics: Mapping[str, float]) -> dict[str, bool]:
    decision = {
        "balanced_accuracy_above_one_half": strictly_greater(metrics["balanced_accuracy"], 0.5),
        "recall_negative_positive": metrics["recall_negative"] > 0.0,
        "recall_positive_positive": metrics["recall_positive"] > 0.0,
    }
    decision["all_fold_conditions_met"] = all(decision.values())
    return decision


def aggregate_primary_decision(fold_roc_auc: Sequence[float]) -> dict[str, float | bool | int]:
    if len(fold_roc_auc) != 4:
        raise AssertionError("EXP_004 requires exactly four Joint OOS folds.")
    mean_roc_auc = float(np.mean(fold_roc_auc))
    all_four = all(strictly_greater(value, 0.5) for value in fold_roc_auc)
    mean_above = strictly_greater(mean_roc_auc, 0.5)
    return {"n_joint_folds": 4, "all_four_roc_auc_above_one_half": all_four, "mean_joint_roc_auc": mean_roc_auc, "mean_joint_roc_auc_above_one_half": mean_above, "promising_oos_sign_discrimination": all_four and mean_above}


def aggregate_secondary_decision(fold_decisions: Sequence[Mapping[str, bool]], fold_ba_deltas: Sequence[float]) -> dict[str, float | bool | int]:
    if len(fold_decisions) != 4 or len(fold_ba_deltas) != 4:
        raise AssertionError("EXP_004 requires exactly four Joint OOS folds.")
    mean_delta = float(np.mean(fold_ba_deltas))
    all_fold_conditions = all(bool(decision["all_fold_conditions_met"]) for decision in fold_decisions)
    mean_positive = strictly_greater(mean_delta, 0.0)
    return {"n_joint_folds": 4, "all_four_fixed_threshold_conditions_met": all_fold_conditions, "mean_joint_balanced_accuracy_delta_vs_baseline": mean_delta, "mean_joint_balanced_accuracy_delta_positive": mean_positive, "fixed_threshold_sign_classification_evidence": all_fold_conditions and mean_positive}

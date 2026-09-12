"""Utilities for frozen EXP_003 conditional directional-sign evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.exp001_missingness import COMPARISON_ATOL, LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS, strictly_greater


SIGN_CLASS_ORDER: tuple[int, int] = (0, 1)
PREDICTION_THRESHOLD = 0.5
DIRECTIONAL_REOD_VALUES = (-1, 1)


@dataclass(frozen=True)
class EvaluableDirectionalSubset:
    """One frozen subset after directional conditioning and fixed evaluability."""

    feature_matrix: pd.DataFrame
    sign_target: pd.Series
    manifest: dict[str, int]


def _validate_return_columns(frame: pd.DataFrame) -> None:
    missing_returns = set(RETURN_COLUMNS).difference(frame.columns)
    if missing_returns:
        raise AssertionError(f"Missing EXP_003 return columns: {sorted(missing_returns)}")


def build_sign_target(reod: pd.Series) -> pd.Series:
    """Map directional ternary labels to negative/positive sign labels."""
    if reod.isna().any() or not set(reod.unique()).issubset(set(DIRECTIONAL_REOD_VALUES)):
        raise AssertionError("EXP_003 sign target requires only reod values -1 and +1.")
    return pd.Series(np.where(reod.eq(-1), 0, 1), index=reod.index, name="S", dtype="int8")


def prepare_evaluable_directional_subset(features: pd.DataFrame, reod: pd.Series) -> EvaluableDirectionalSubset:
    """Apply frozen D then n_obs>=1 rules and construct the sole R_obs feature."""
    if not features.index.equals(reod.index):
        raise AssertionError("Features and reod must have identical indices.")
    _validate_return_columns(features)
    if reod.isna().any() or not set(reod.unique()).issubset({-1, 0, 1}):
        raise AssertionError("Unexpected reod labels before EXP_003 conditioning.")

    directional = reod.isin(DIRECTIONAL_REOD_VALUES)
    directional_features = features.loc[directional]
    directional_reod = reod.loc[directional]
    n_obs = directional_features.loc[:, RETURN_COLUMNS].notna().sum(axis=1).astype("int8")
    evaluable = n_obs.ge(1)
    retained_features = directional_features.loc[evaluable]
    retained_reod = directional_reod.loc[evaluable]
    retained_n_obs = n_obs.loc[evaluable]

    # This sum is only calculated for n_obs>=1 rows. Missing cells are skipped,
    # and no all-NaN row receives a synthetic zero cumulative return.
    r_obs = retained_features.loc[:, RETURN_COLUMNS].sum(axis=1, skipna=True).astype("float64")
    feature_matrix = pd.DataFrame({"R_obs": r_obs}, index=retained_features.index)
    sign_target = build_sign_target(retained_reod)
    validate_r_obs_matrix(feature_matrix, expected_index=sign_target.index)

    before_counts = directional_reod.value_counts().reindex(DIRECTIONAL_REOD_VALUES, fill_value=0)
    after_counts = retained_reod.value_counts().reindex(DIRECTIONAL_REOD_VALUES, fill_value=0)
    manifest = {
        "n_directional_before_evaluability": int(len(directional_reod)),
        "n_all_nan_directional_excluded": int((~evaluable).sum()),
        "n_evaluable_directional_retained": int(len(retained_reod)),
        "n_negative_before_evaluability": int(before_counts.loc[-1]),
        "n_positive_before_evaluability": int(before_counts.loc[1]),
        "n_negative_after_evaluability": int(after_counts.loc[-1]),
        "n_positive_after_evaluability": int(after_counts.loc[1]),
    }
    if manifest["n_directional_before_evaluability"] != (
        manifest["n_all_nan_directional_excluded"] + manifest["n_evaluable_directional_retained"]
    ):
        raise AssertionError("EXP_003 evaluability counts do not reconcile.")
    if not retained_n_obs.ge(1).all() or not retained_reod.isin(DIRECTIONAL_REOD_VALUES).all():
        raise AssertionError("Non-evaluable or neutral row retained in EXP_003.")
    return EvaluableDirectionalSubset(feature_matrix, sign_target, manifest)


def validate_r_obs_matrix(matrix: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    """Assert that exactly the one allowed cumulative-observed-return feature remains."""
    if list(matrix.columns) != ["R_obs"]:
        raise AssertionError("EXP_003 predictive matrix must contain exactly R_obs.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("R_obs changed row identity or order.")
    values = matrix.to_numpy(dtype=float)
    if matrix.isna().any().any() or not np.isfinite(values).all():
        raise AssertionError("R_obs must be finite for evaluable rows.")


def fit_scaler_on_fit(feature_matrix: pd.DataFrame) -> tuple[StandardScaler, pd.DataFrame]:
    """Fit the sole permitted scaler on evaluable directional fit rows only."""
    validate_r_obs_matrix(feature_matrix)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(feature_matrix)
    if scaler.n_features_in_ != 1 or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or not (scaler.scale_ > 0).all():
        raise AssertionError("EXP_003 fit scaler requires one finite positive-scale feature.")
    return scaler, pd.DataFrame(transformed, index=feature_matrix.index, columns=["R_obs"])


def transform_with_fit_scaler(scaler: StandardScaler, feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply an already-fitted fit-only scaler without refitting."""
    validate_r_obs_matrix(feature_matrix)
    if getattr(scaler, "n_features_in_", None) != 1 or not np.isfinite(scaler.scale_).all() or not (scaler.scale_ > 0).all():
        raise AssertionError("EXP_003 scaler is not a valid fitted one-feature scaler.")
    transformed = scaler.transform(feature_matrix)
    return pd.DataFrame(transformed, index=feature_matrix.index, columns=["R_obs"])


def make_sign_logistic_regression() -> LogisticRegression:
    """Build the exact frozen binary LogisticRegression configuration."""
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    for name, expected in LOGISTIC_REGRESSION_PARAMS.items():
        if classifier.get_params()[name] != expected:
            raise AssertionError(f"LogisticRegression parameter {name!r} differs from EXP_003.")
    return classifier


def sign_majority_class(fit_target: pd.Series) -> int:
    """Fit-only binary sign majority, with the frozen [0,1] tie order."""
    if fit_target.empty or not set(fit_target.unique()).issubset(SIGN_CLASS_ORDER):
        raise AssertionError("Invalid fit target for EXP_003 majority baseline.")
    counts = fit_target.value_counts().reindex(SIGN_CLASS_ORDER, fill_value=0)
    maximum = counts.max()
    return next(label for label in SIGN_CLASS_ORDER if counts.loc[label] == maximum)


def assert_both_sign_classes(target: pd.Series, *, context: str) -> None:
    if set(target.unique()) != set(SIGN_CLASS_ORDER):
        raise AssertionError(f"Both sign classes are required for {context}.")


def positive_probability(classifier: LogisticRegression, feature_matrix: pd.DataFrame) -> np.ndarray:
    """Extract P(S=1) explicitly from the fitted classifier class labels."""
    if tuple(classifier.classes_) != SIGN_CLASS_ORDER:
        raise AssertionError("EXP_003 classifier classes must be ordered (0, 1).")
    probability = classifier.predict_proba(feature_matrix)[:, int(np.flatnonzero(classifier.classes_ == 1)[0])]
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError("Positive-sign probabilities must be finite and in [0, 1].")
    return values


def threshold_predictions(probability: np.ndarray, *, threshold: float = PREDICTION_THRESHOLD) -> np.ndarray:
    if threshold != PREDICTION_THRESHOLD:
        raise AssertionError("EXP_003 threshold must be exactly 0.5.")
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError("Threshold input must be finite probabilities in [0, 1].")
    return (values >= threshold).astype("int8")


def evaluate_sign_predictions(target: pd.Series, prediction: np.ndarray, *, probability: np.ndarray | None = None) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate conditional sign predictions in the frozen label order."""
    assert_both_sign_classes(target, context="OOS evaluation")
    truth = target.to_numpy(dtype="int8")
    predicted = np.asarray(prediction, dtype="int8")
    if len(truth) != len(predicted) or not np.isin(predicted, SIGN_CLASS_ORDER).all():
        raise AssertionError("EXP_003 predictions must be aligned binary sign labels.")
    recalls = recall_score(truth, predicted, labels=SIGN_CLASS_ORDER, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "recall_negative": float(recalls[0]),
        "recall_positive": float(recalls[1]),
    }
    if probability is not None:
        values = np.asarray(probability, dtype=float)
        if len(values) != len(truth):
            raise AssertionError("Probability length does not match sign target.")
        metrics["roc_auc"] = float(roc_auc_score(truth, values))
    return metrics, confusion_matrix(truth, predicted, labels=SIGN_CLASS_ORDER).astype(int).tolist()


def evaluate_sign_majority_baseline(target: pd.Series, predicted_class: int) -> tuple[dict[str, float], list[list[int]]]:
    if predicted_class not in SIGN_CLASS_ORDER:
        raise AssertionError("EXP_003 baseline class must be 0 or 1.")
    return evaluate_sign_predictions(target, np.full(len(target), predicted_class, dtype="int8"))


def delta_vs_baseline(value: float, reference: float) -> float:
    delta = value - reference
    return 0.0 if np.isclose(delta, 0.0, rtol=0.0, atol=COMPARISON_ATOL) else float(delta)


def coefficient_beta(classifier: LogisticRegression) -> float:
    if tuple(classifier.classes_) != SIGN_CLASS_ORDER or classifier.coef_.shape != (1, 1):
        raise AssertionError("EXP_003 requires one fitted binary R_obs coefficient.")
    return float(classifier.coef_[0, 0])


def coefficient_sign(beta: float) -> str:
    return "positive" if beta > 0.0 else "negative" if beta < 0.0 else "zero"


def coefficient_sign_consistent(betas: Sequence[float]) -> bool:
    if len(betas) != 4:
        raise AssertionError("EXP_003 coefficient diagnostic requires four folds.")
    return all(beta > 0.0 for beta in betas) or all(beta < 0.0 for beta in betas)


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
        raise AssertionError("EXP_003 requires exactly four Joint OOS folds.")
    mean_roc_auc = float(np.mean(fold_roc_auc))
    all_four = all(strictly_greater(value, 0.5) for value in fold_roc_auc)
    mean_above = strictly_greater(mean_roc_auc, 0.5)
    return {"n_joint_folds": 4, "all_four_roc_auc_above_one_half": all_four, "mean_joint_roc_auc": mean_roc_auc, "mean_joint_roc_auc_above_one_half": mean_above, "promising_oos_sign_discrimination": all_four and mean_above}


def aggregate_secondary_decision(fold_decisions: Sequence[Mapping[str, bool]], fold_ba_deltas: Sequence[float]) -> dict[str, float | bool | int]:
    if len(fold_decisions) != 4 or len(fold_ba_deltas) != 4:
        raise AssertionError("EXP_003 requires exactly four Joint OOS folds.")
    mean_delta = float(np.mean(fold_ba_deltas))
    all_fold_conditions = all(bool(decision["all_fold_conditions_met"]) for decision in fold_decisions)
    mean_positive = strictly_greater(mean_delta, 0.0)
    return {"n_joint_folds": 4, "all_four_fixed_threshold_conditions_met": all_fold_conditions, "mean_joint_balanced_accuracy_delta_vs_baseline": mean_delta, "mean_joint_balanced_accuracy_delta_positive": mean_positive, "fixed_threshold_sign_classification_evidence": all_fold_conditions and mean_positive}

"""Binary missingness utilities specified by the frozen EXP_002 contract."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from src.exp001_missingness import COMPARISON_ATOL, LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS, strictly_greater


BINARY_CLASS_ORDER: tuple[int, int] = (0, 1)
PREDICTION_THRESHOLD = 0.5
STRUCTURAL_SESSION_DAYS: tuple[int, ...] = (112, 134, 229, 314, 438, 469)


def build_binary_target(reod: pd.Series) -> pd.Series:
    """Map the ternary target to EXP_002's neutral/directional target."""
    if reod.isna().any():
        raise AssertionError("reod cannot contain missing values.")
    observed = set(reod.unique())
    allowed = {-1, 0, 1}
    if not observed.issubset(allowed):
        raise AssertionError(f"Unexpected reod labels: {sorted(observed.difference(allowed))}")
    return pd.Series(np.where(reod.eq(0), 0, 1), index=reod.index, name="Z", dtype="int8")


def build_missing_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the one deterministic missingness feature in EXP_002."""
    missing_returns = set(RETURN_COLUMNS).difference(frame.columns)
    if missing_returns:
        raise AssertionError(f"Missing EXP_002 return columns: {sorted(missing_returns)}")
    ratio = frame.loc[:, RETURN_COLUMNS].isna().mean(axis=1).astype("float32")
    matrix = pd.DataFrame({"missing_ratio": ratio}, index=frame.index)
    validate_missing_ratio(matrix, expected_index=frame.index)
    return matrix


def validate_missing_ratio(matrix: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    """Assert the exact one-column feature schema and its deterministic bounds."""
    if list(matrix.columns) != ["missing_ratio"]:
        raise AssertionError("EXP_002 requires exactly the missing_ratio feature.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("missing_ratio changed row identity or order.")
    if matrix.isna().any().any():
        raise AssertionError("missing_ratio contains missing feature values.")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError("missing_ratio values must be finite and in [0, 1].")


def make_binary_logistic_regression() -> LogisticRegression:
    """Construct the contract's fixed binary logistic-regression classifier."""
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    for name, expected in LOGISTIC_REGRESSION_PARAMS.items():
        if classifier.get_params()[name] != expected:
            raise AssertionError(f"LogisticRegression parameter {name!r} differs from EXP_002.")
    return classifier


def binary_majority_class(fit_target: pd.Series) -> int:
    """Return the fit-only binary majority class; ties select neutral (0)."""
    if fit_target.empty:
        raise AssertionError("Cannot compute a majority class from an empty fit target.")
    if not set(fit_target.unique()).issubset(BINARY_CLASS_ORDER):
        raise AssertionError("Binary majority baseline received an invalid target label.")
    counts = fit_target.value_counts().reindex(BINARY_CLASS_ORDER, fill_value=0)
    maximum = counts.max()
    return next(label for label in BINARY_CLASS_ORDER if counts.loc[label] == maximum)


def binary_class_counts(target: pd.Series) -> dict[str, int]:
    """Return binary target counts in the frozen neutral/directional order."""
    counts = target.value_counts().reindex(BINARY_CLASS_ORDER, fill_value=0)
    return {"n_neutral": int(counts.loc[0]), "n_directional": int(counts.loc[1])}


def assert_both_binary_classes(target: pd.Series, *, context: str) -> None:
    """Require evaluability of binary recalls, balanced accuracy, and ROC-AUC."""
    if set(target.unique()) != set(BINARY_CLASS_ORDER):
        raise AssertionError(f"Both binary classes are required for {context}.")


def directional_probability(classifier: LogisticRegression, matrix: pd.DataFrame) -> np.ndarray:
    """Return P(Z=1), locating class 1 rather than assuming a column position."""
    if tuple(classifier.classes_) != BINARY_CLASS_ORDER:
        raise AssertionError("Classifier classes must be ordered (0, 1) for EXP_002.")
    probabilities = classifier.predict_proba(matrix)
    directional_column = int(np.flatnonzero(classifier.classes_ == 1)[0])
    probability = np.asarray(probabilities[:, directional_column], dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise AssertionError("Directional probabilities must be finite and in [0, 1].")
    return probability


def threshold_predictions(probability: np.ndarray, *, threshold: float = PREDICTION_THRESHOLD) -> np.ndarray:
    """Apply the frozen probability threshold; equality is directional by definition."""
    if threshold != PREDICTION_THRESHOLD:
        raise AssertionError("EXP_002 threshold must be exactly 0.5.")
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError("Threshold input must be finite probabilities in [0, 1].")
    return (values >= threshold).astype("int8")


def evaluate_binary_predictions(
    target: pd.Series,
    prediction: np.ndarray,
    *,
    directional_probability_values: np.ndarray | None = None,
) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate EXP_002 binary predictions, optionally including ROC-AUC."""
    if target.empty:
        raise AssertionError("Cannot evaluate an empty target subset.")
    assert_both_binary_classes(target, context="OOS evaluation")
    truth = target.to_numpy(dtype="int8")
    predicted = np.asarray(prediction, dtype="int8")
    if len(truth) != len(predicted):
        raise AssertionError("Prediction length does not match the target.")
    if not np.isin(predicted, BINARY_CLASS_ORDER).all():
        raise AssertionError("Predictions must be binary labels 0 or 1.")
    recalls = recall_score(truth, predicted, labels=BINARY_CLASS_ORDER, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "recall_neutral": float(recalls[0]),
        "recall_directional": float(recalls[1]),
    }
    if directional_probability_values is not None:
        probability = np.asarray(directional_probability_values, dtype=float)
        if len(probability) != len(truth):
            raise AssertionError("Probability length does not match the target.")
        metrics["roc_auc"] = float(roc_auc_score(truth, probability))
    matrix = confusion_matrix(truth, predicted, labels=BINARY_CLASS_ORDER).astype(int).tolist()
    return metrics, matrix


def evaluate_binary_majority_baseline(target: pd.Series, predicted_class: int) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate the binary constant baseline using the fit-selected class."""
    if predicted_class not in BINARY_CLASS_ORDER:
        raise AssertionError("Binary baseline class must be 0 or 1.")
    prediction = np.full(len(target), predicted_class, dtype="int8")
    return evaluate_binary_predictions(target, prediction)


def delta_vs_baseline(value: float, reference: float) -> float:
    """Return a stable model-minus-baseline delta."""
    delta = value - reference
    return 0.0 if np.isclose(delta, 0.0, rtol=0.0, atol=COMPARISON_ATOL) else float(delta)


def primary_fold_decision(roc_auc: float) -> dict[str, bool]:
    """Evaluate EXP_002's threshold-independent Joint OOS fold condition."""
    return {"roc_auc_above_one_half": strictly_greater(roc_auc, 0.5)}


def secondary_fold_decision(metrics: Mapping[str, float]) -> dict[str, bool]:
    """Evaluate EXP_002's fixed-threshold Joint OOS fold conditions."""
    decision = {
        "balanced_accuracy_above_one_half": strictly_greater(metrics["balanced_accuracy"], 0.5),
        "recall_neutral_positive": metrics["recall_neutral"] > 0.0,
        "recall_directional_positive": metrics["recall_directional"] > 0.0,
    }
    decision["all_fold_conditions_met"] = all(decision.values())
    return decision


def aggregate_primary_decision(fold_roc_auc: Sequence[float]) -> dict[str, float | bool | int]:
    """Aggregate the frozen all-fold and mean ROC-AUC primary decision."""
    if len(fold_roc_auc) != 4:
        raise AssertionError("EXP_002 requires exactly four Joint OOS folds.")
    mean_roc_auc = float(np.mean(fold_roc_auc))
    all_four = all(strictly_greater(value, 0.5) for value in fold_roc_auc)
    mean_above = strictly_greater(mean_roc_auc, 0.5)
    return {
        "n_joint_folds": len(fold_roc_auc),
        "all_four_roc_auc_above_one_half": all_four,
        "mean_joint_roc_auc": mean_roc_auc,
        "mean_joint_roc_auc_above_one_half": mean_above,
        "promising_oos_discrimination": all_four and mean_above,
    }


def aggregate_secondary_decision(fold_decisions: Sequence[Mapping[str, bool]], fold_ba_deltas: Sequence[float]) -> dict[str, float | bool | int]:
    """Aggregate the frozen fixed-threshold classification decision."""
    if len(fold_decisions) != 4 or len(fold_ba_deltas) != 4:
        raise AssertionError("EXP_002 requires exactly four Joint OOS folds.")
    mean_delta = float(np.mean(fold_ba_deltas))
    all_fold_conditions = all(bool(decision["all_fold_conditions_met"]) for decision in fold_decisions)
    mean_delta_positive = strictly_greater(mean_delta, 0.0)
    return {
        "n_joint_folds": len(fold_decisions),
        "all_four_fixed_threshold_conditions_met": all_fold_conditions,
        "mean_joint_balanced_accuracy_delta_vs_baseline": mean_delta,
        "mean_joint_balanced_accuracy_delta_positive": mean_delta_positive,
        "fixed_threshold_classification_evidence": all_fold_conditions and mean_delta_positive,
    }


def assert_subset_matches_frozen_split(actual: pd.DataFrame, expected: pd.DataFrame, *, subset_name: str) -> None:
    """Detect any post-split filtering, including structural-session exclusion."""
    if not actual.index.equals(expected.index) or not actual["ID"].equals(expected["ID"]):
        raise AssertionError(f"{subset_name} differs from the frozen shared splitter output.")

"""Fixed missingness-only representations and evaluation for EXP_001.

The functions in this module intentionally exclude returns and identifiers from
the predictive matrices.  They implement only the representations and fixed
classifier stated in ``experiments/EXP_001_missingness_signal.md``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from src.exp000_validation import CLASS_ORDER


RETURN_COLUMNS: tuple[str, ...] = tuple(f"r{position}" for position in range(53))
M2_WINDOWS: Mapping[str, tuple[int, int]] = {
    "missing_early": (0, 16),
    "missing_mid": (17, 34),
    "missing_late": (35, 52),
}
VARIANT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "M1": ("missing_ratio",),
    "M2": tuple(M2_WINDOWS),
    "M3": tuple(f"m{position}" for position in range(53)),
}
LOGISTIC_REGRESSION_PARAMS: Mapping[str, object] = {
    "solver": "lbfgs",
    "penalty": "l2",
    "C": 1.0,
    "fit_intercept": True,
    "class_weight": None,
    "max_iter": 1000,
    "tol": 1e-4,
    "random_state": 20260908,
}
COMPARISON_ATOL = 1e-12


def build_missingness_representations(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the three deterministic, missingness-only EXP_001 matrices.

    ``frame`` must contain exactly the contract's return columns.  Only their
    missing-value indicators are used; observed zero and nonzero returns both
    become zero in the mask.
    """
    missing_returns = set(RETURN_COLUMNS).difference(frame.columns)
    if missing_returns:
        raise AssertionError(f"Missing EXP_001 return columns: {sorted(missing_returns)}")

    mask = frame.loc[:, RETURN_COLUMNS].isna().astype("int8")
    m1 = pd.DataFrame({"missing_ratio": mask.mean(axis=1).astype("float32")}, index=frame.index)
    m2 = pd.DataFrame(index=frame.index)
    for name, (start, end) in M2_WINDOWS.items():
        m2[name] = mask.iloc[:, start : end + 1].mean(axis=1).astype("float32")
    m3 = mask.rename(columns={f"r{position}": f"m{position}" for position in range(53)})

    representations = {"M1": m1, "M2": m2, "M3": m3}
    for variant, matrix in representations.items():
        validate_representation(variant, matrix, expected_index=frame.index)
    return representations


def validate_representation(
    variant: str,
    matrix: pd.DataFrame,
    *,
    expected_index: pd.Index | None = None,
) -> None:
    """Assert exact EXP_001 schema, row identity, and missingness bounds."""
    if variant not in VARIANT_COLUMNS:
        raise AssertionError(f"Unknown EXP_001 variant: {variant}")
    expected_columns = list(VARIANT_COLUMNS[variant])
    if list(matrix.columns) != expected_columns:
        raise AssertionError(f"{variant} columns do not match the EXP_001 contract.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError(f"{variant} changed row identity or order.")
    if matrix.isna().any().any():
        raise AssertionError(f"{variant} contains missing feature values.")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError(f"{variant} values must be finite and in [0, 1].")
    if variant == "M3" and not np.isin(values, [0.0, 1.0]).all():
        raise AssertionError("M3 must be a binary missingness mask.")


def make_logistic_regression() -> LogisticRegression:
    """Return the exact fixed multinomial-compatible EXP_001 classifier."""
    # ``multi_class`` is deliberately absent: sklearn 1.6 deprecates passing
    # it explicitly, while lbfgs retains multinomial behavior for three classes.
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    for name, expected in LOGISTIC_REGRESSION_PARAMS.items():
        if classifier.get_params()[name] != expected:
            raise AssertionError(f"LogisticRegression parameter {name!r} differs from EXP_001.")
    return classifier


def evaluate_predictions(target: pd.Series, prediction: np.ndarray) -> tuple[dict[str, float], list[list[int]]]:
    """Calculate EXP_001 metrics and class recalls in the stable label order."""
    if target.empty:
        raise AssertionError("Cannot evaluate an empty target subset.")
    truth = target.to_numpy(dtype=np.int8)
    predicted = np.asarray(prediction, dtype=np.int8)
    if len(truth) != len(predicted):
        raise AssertionError("Prediction length does not match the target.")
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
    }
    recalls = recall_score(truth, predicted, labels=CLASS_ORDER, average=None, zero_division=0)
    metrics.update({f"recall_class_{label}": float(value) for label, value in zip(CLASS_ORDER, recalls, strict=True)})
    matrix = confusion_matrix(truth, predicted, labels=CLASS_ORDER).astype(int).tolist()
    return metrics, matrix


def joint_decision(metrics: Mapping[str, float], baseline_accuracy: float) -> dict[str, bool]:
    """Evaluate the strict, pre-specified EXP_001 Joint OOS fold conditions."""
    decision = {
        "accuracy_above_baseline": strictly_greater(metrics["accuracy"], baseline_accuracy),
        "balanced_accuracy_above_one_third": strictly_greater(metrics["balanced_accuracy"], 1.0 / 3.0),
        "recall_negative_positive": metrics["recall_class_-1"] > 0.0,
        "recall_positive_positive": metrics["recall_class_1"] > 0.0,
    }
    decision["all_fold_conditions_met"] = all(decision.values())
    return decision


def strictly_greater(value: float, reference: float) -> bool:
    """Apply a strict comparison while treating round-off equality as equality."""
    return value > reference and not np.isclose(value, reference, rtol=0.0, atol=COMPARISON_ATOL)


def delta_vs_baseline(value: float, reference: float) -> float:
    """Return a comparison delta, collapsing floating-point equality to zero."""
    delta = value - reference
    return 0.0 if np.isclose(delta, 0.0, rtol=0.0, atol=COMPARISON_ATOL) else float(delta)

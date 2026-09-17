"""Frozen decision-layer utilities for D004; no data loading or execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS
from src.exp000_validation import CLASS_ORDER, majority_class
from src.exp002_neutral_directional import build_binary_target, build_missing_ratio
from src.exp008_recent_movement_intensity import (
    RecentIntensityTransformer,
    assert_identical_representation_rows,
    build_exp008_representations,
    build_recent_window_structure,
)


RELIABILITY_EDGES: tuple[float, ...] = tuple(index / 10.0 for index in range(11))
RELIABILITY_LABELS: tuple[str, ...] = tuple(
    f"[{RELIABILITY_EDGES[index]:.1f}, {RELIABILITY_EDGES[index + 1]:.1f}{']' if index == 9 else ')'}"
    for index in range(10)
)


@dataclass(frozen=True)
class DirectionalPriors:
    pi_minus: float
    pi_plus: float
    pi_max: float
    threshold: float


def fit_directional_priors(fit_reod: pd.Series) -> DirectionalPriors:
    """Estimate D004 directional priors only from the supplied Fit labels."""
    if fit_reod.empty or not set(fit_reod.unique()).issubset(set(CLASS_ORDER)):
        raise AssertionError("D004 Fit labels must be nonempty ternary labels.")
    n_minus = int(fit_reod.eq(-1).sum())
    n_plus = int(fit_reod.eq(1).sum())
    denominator = n_minus + n_plus
    if denominator == 0:
        raise AssertionError("D004 Fit requires at least one directional label for directional priors.")
    pi_minus = n_minus / denominator
    pi_plus = n_plus / denominator
    if not np.isclose(pi_minus + pi_plus, 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("D004 directional priors must sum to one.")
    pi_max = max(pi_minus, pi_plus)
    threshold = 1.0 / (1.0 + pi_max)
    return DirectionalPriors(pi_minus, pi_plus, pi_max, threshold)


def directional_probability(model, matrix: pd.DataFrame) -> np.ndarray:
    """Return raw P(Z=1), locating the column by class value, not position."""
    classes = np.asarray(model.classes_)
    locations = np.flatnonzero(classes == 1)
    if len(locations) != 1 or set(classes.tolist()) != {0, 1}:
        raise AssertionError("D004 intensity model must contain exactly binary classes 0 and 1.")
    probability = np.asarray(model.predict_proba(matrix)[:, int(locations[0])], dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise AssertionError("D004 p_D must be finite and in [0, 1].")
    return probability


def hierarchical_probabilities(p_d: np.ndarray, priors: DirectionalPriors) -> pd.DataFrame:
    """Build and validate the frozen no-sign-alpha ternary probability map."""
    values = np.asarray(p_d, dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError("D004 p_D values must be finite probabilities.")
    probabilities = pd.DataFrame({
        "P_minus": values * priors.pi_minus,
        "P_zero": 1.0 - values,
        "P_plus": values * priors.pi_plus,
    })
    if not np.allclose(probabilities.sum(axis=1).to_numpy(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("D004 hierarchical probabilities must sum to one rowwise.")
    return probabilities


def hierarchical_predictions(p_d: np.ndarray, priors: DirectionalPriors) -> np.ndarray:
    """Apply D004's explicit strict directional-routing and tie convention."""
    values = np.asarray(p_d, dtype=float)
    _ = hierarchical_probabilities(values, priors)
    predicted = np.zeros(len(values), dtype=np.int8)
    directional = values > priors.threshold  # exact: equality routes to neutral.
    if priors.pi_minus >= priors.pi_plus:
        predicted[directional] = -1
    else:
        predicted[directional] = 1
    return predicted


def ternary_metrics(target: pd.Series, prediction: np.ndarray) -> tuple[dict[str, float], list[list[int]]]:
    """Return fixed ternary diagnostics for candidate or constant baseline."""
    truth = target.to_numpy(dtype=np.int8)
    predicted = np.asarray(prediction, dtype=np.int8)
    if len(truth) != len(predicted) or not np.isin(predicted, CLASS_ORDER).all():
        raise AssertionError("D004 ternary predictions must align and use labels -1, 0, +1.")
    recalls = recall_score(truth, predicted, labels=CLASS_ORDER, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "recall_negative": float(recalls[0]),
        "recall_neutral": float(recalls[1]),
        "recall_positive": float(recalls[2]),
    }
    return metrics, confusion_matrix(truth, predicted, labels=CLASS_ORDER).astype(int).tolist()


def fit_majority_predictions(fit_reod: pd.Series, n_rows: int) -> tuple[int, np.ndarray]:
    """Apply EXP_000's fit-only ternary majority baseline and tie ordering."""
    selected = majority_class(fit_reod)
    return selected, np.full(n_rows, selected, dtype=np.int8)


def calibration_diagnostics(target_z: pd.Series, p_d: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    """Return D004's descriptive raw-probability calibration diagnostics."""
    truth = target_z.to_numpy(dtype=np.int8)
    probability = np.asarray(p_d, dtype=float)
    if len(truth) != len(probability) or set(truth.tolist()) - {0, 1}:
        raise AssertionError("D004 calibration requires aligned binary target labels.")
    if not np.isfinite(probability).all() or ((probability < 0.0) | (probability > 1.0)).any():
        raise AssertionError("D004 calibration requires probabilities in [0, 1].")
    bins = np.minimum((probability * 10).astype(int), 9)
    rows: list[dict[str, float | int | str]] = []
    for index, label in enumerate(RELIABILITY_LABELS):
        member = bins == index
        rows.append({
            "bin": label,
            "n_rows": int(member.sum()),
            "mean_predicted_p_d": float(probability[member].mean()) if member.any() else np.nan,
            "empirical_directional_frequency": float(truth[member].mean()) if member.any() else np.nan,
        })
    table = pd.DataFrame(rows)
    if int(table["n_rows"].sum()) != len(probability):
        raise AssertionError("D004 reliability bins must cover every probability exactly once.")
    return {
        "brier_score": float(brier_score_loss(truth, probability)),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1])),
    }, table


def fit_intensity_transformer(fit_features: pd.DataFrame) -> tuple[pd.DataFrame, RecentIntensityTransformer]:
    """Reuse the exact EXP_008 Fit-only recent-intensity transformer."""
    structure = build_recent_window_structure(fit_features)
    return structure, RecentIntensityTransformer().fit(structure["raw_I_recent"])


def build_d004_matrix(
    features: pd.DataFrame, target: pd.Series, transformer: RecentIntensityTransformer
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct exact EXP_008 B and structural quantities without learning parameters."""
    structure = build_recent_window_structure(features)
    intensity_z = transformer.transform(structure["raw_I_recent"])
    representations = build_exp008_representations(build_missing_ratio(features), structure, intensity_z)
    assert_identical_representation_rows(representations, features, target)
    return representations["B"], structure


__all__ = [
    "DISCOVERY_FOLDS", "DirectionalPriors", "RELIABILITY_EDGES", "RELIABILITY_LABELS",
    "build_d004_matrix", "calibration_diagnostics", "directional_probability",
    "fit_directional_priors", "fit_intensity_transformer", "fit_majority_predictions",
    "hierarchical_predictions", "hierarchical_probabilities", "ternary_metrics",
]

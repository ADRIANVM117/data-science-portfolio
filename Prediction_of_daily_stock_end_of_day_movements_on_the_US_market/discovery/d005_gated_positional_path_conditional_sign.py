"""Frozen D005 gate-to-positional-sign orchestration utilities.

D005 adds no gate or path representation: it composes authoritative D004 and
EXP_005 machinery in the frozen population-construction order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from discovery.d003_incremental_cross_sectional_relative_path import DISCOVERY_FOLDS
from discovery.d004_hierarchical_recent_intensity_ternary_decision import DirectionalPriors
from src.exp003_conditional_directional_sign import (
    assert_both_sign_classes,
    evaluate_sign_predictions,
    positive_probability,
    sign_majority_class,
    threshold_predictions,
)
from src.exp005_positional_path_signal import (
    PATH_MASK_COLUMNS,
    EvaluableDirectionalRows,
    PositionalPathTransformer,
    build_path_mask,
    make_positional_logistic_regression,
    prepare_evaluable_directional_rows,
)


SIGN_FORBIDDEN_FEATURES: tuple[str, ...] = (
    "p_D", "p_D_star", "gate_margin", "I_recent", "q", "pi_minus", "pi_plus", "pi_max",
)


@dataclass(frozen=True)
class GatedDirectionalRows:
    """A D005 subset after gate, direction, and inherited evaluability."""

    raw_returns: pd.DataFrame
    sign_target: pd.Series
    gate: pd.Series
    manifest: dict[str, int | float]


def strict_d004_gate(probability: np.ndarray, priors: DirectionalPriors, *, index: pd.Index) -> pd.Series:
    """Apply D004's strict gate: equality at the Fit-derived threshold is out."""
    values = np.asarray(probability, dtype=float)
    if len(values) != len(index) or not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise AssertionError("D005 requires finite D004 p_D values aligned to the subset.")
    if not np.isfinite(priors.threshold) or not (0.0 < priors.threshold < 1.0):
        raise AssertionError("D005 requires a valid Fit-derived D004 threshold.")
    return pd.Series(values > priors.threshold, index=index, name="G", dtype=bool)


def prepare_gated_directional_rows(
    features: pd.DataFrame, reod: pd.Series, probability: np.ndarray, priors: DirectionalPriors
) -> GatedDirectionalRows:
    """Preserve frozen order: gate -> direction -> all-NaN evaluability -> S."""
    if not features.index.equals(reod.index):
        raise AssertionError("D005 features and labels must align before gate construction.")
    gate = strict_d004_gate(probability, priors, index=features.index)
    gated_features = features.loc[gate].copy()
    gated_reod = reod.loc[gate].copy()
    prepared: EvaluableDirectionalRows = prepare_evaluable_directional_rows(gated_features, gated_reod)
    manifest: dict[str, int | float] = {
        "n_physical_rows_before_gate": int(len(features)),
        "n_gate_1": int(gate.sum()),
        "n_gate_0": int((~gate).sum()),
        **prepared.manifest,
    }
    if manifest["n_gate_1"] != int(len(gated_features)):
        raise AssertionError("D005 gate count does not reconcile with gated rows.")
    if not prepared.raw_returns.index.equals(prepared.sign_target.index):
        raise AssertionError("D005 gated retained returns and sign target are not aligned.")
    if not prepared.raw_returns.index.isin(features.index).all():
        raise AssertionError("D005 retained rows are outside their physical subset.")
    return GatedDirectionalRows(prepared.raw_returns, prepared.sign_target, gate, manifest)


def build_d005_path_matrix(transformer: PositionalPathTransformer, rows: GatedDirectionalRows) -> pd.DataFrame:
    """Build exactly EXP_005's path-and-original-mask representation."""
    matrix = build_path_mask(transformer, rows.raw_returns)
    if tuple(matrix.columns) != PATH_MASK_COLUMNS or matrix.shape[1] != 106:
        raise AssertionError("D005 sign representation must contain exactly the EXP_005 106 columns.")
    if not matrix.index.equals(rows.sign_target.index):
        raise AssertionError("D005 sign matrix changed retained row alignment.")
    forbidden = set(SIGN_FORBIDDEN_FEATURES).intersection(matrix.columns)
    if forbidden:
        raise AssertionError(f"D005 sign matrix contains forbidden gate variables: {sorted(forbidden)}")
    return matrix


def evaluate_d005_sign_model(target: pd.Series, matrix: pd.DataFrame, classifier) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate probability AUC and fixed-threshold diagnostics without adaptation."""
    if not matrix.index.equals(target.index):
        raise AssertionError("D005 evaluation matrix and sign target must align.")
    assert_both_sign_classes(target, context="D005 sign evaluation")
    probability = positive_probability(classifier, matrix)
    prediction = threshold_predictions(probability)
    return evaluate_sign_predictions(target, prediction, probability=probability)


def descriptive_candidate_screen(validation_aucs: Sequence[float]) -> dict[str, bool | float | int]:
    """D005's non-confirmatory, strict three-fold recurrence screen."""
    if len(validation_aucs) != 3:
        raise AssertionError("D005 requires exactly three Discovery Validation AUCs.")
    values = np.asarray(validation_aucs, dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("D005 Validation AUCs must be finite.")
    return {
        "n_discovery_folds": 3,
        "auc_above_one_half_all_three": bool((values > 0.5).all()),
        "mean_validation_auc": float(values.mean()),
        "sample_std_validation_auc": float(values.std(ddof=1)),
        "descriptive_candidate_screen_met": bool((values > 0.5).all()),
    }


__all__ = [
    "DISCOVERY_FOLDS", "GatedDirectionalRows", "SIGN_FORBIDDEN_FEATURES",
    "PositionalPathTransformer", "build_d005_path_matrix", "descriptive_candidate_screen",
    "evaluate_d005_sign_model", "make_positional_logistic_regression",
    "prepare_gated_directional_rows", "sign_majority_class", "strict_d004_gate",
]

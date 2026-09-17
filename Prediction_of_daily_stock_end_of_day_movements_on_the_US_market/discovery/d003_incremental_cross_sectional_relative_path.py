"""Frozen target-free feature construction and evaluation helpers for D003."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.exp000_validation import FoldSpec
from src.exp001_missingness import RETURN_COLUMNS
from src.exp003_conditional_directional_sign import (
    assert_both_sign_classes,
    evaluate_sign_predictions,
    positive_probability,
    threshold_predictions,
)
from src.exp005_positional_path_signal import (
    MASK_COLUMNS,
    PositionalPathTransformer,
    build_original_mask,
    build_path_mask,
    make_positional_logistic_regression,
    prepare_evaluable_directional_rows,
)


DISCOVERY_FOLDS: tuple[FoldSpec, ...] = (
    FoldSpec(1, 0, 202, 203, 252),
    FoldSpec(2, 0, 252, 253, 302),
    FoldSpec(3, 0, 302, 303, 352),
)
RELATIVE_SCALED_COLUMNS: tuple[str, ...] = tuple(f"x_u{position}" for position in range(53))
CONTROL_COLUMNS: tuple[str, ...] = tuple(f"x_r{position}" for position in range(53)) + MASK_COLUMNS
CANDIDATE_COLUMNS: tuple[str, ...] = CONTROL_COLUMNS + RELATIVE_SCALED_COLUMNS


@dataclass(frozen=True)
class DiscoverySubsets:
    fold: FoldSpec
    fit_features: pd.DataFrame
    fit_reod: pd.Series
    validation_features: pd.DataFrame
    validation_reod: pd.Series


def validate_discovery_features(features: pd.DataFrame) -> None:
    """Require the physical input-only Discovery boundary, never raw training."""
    expected = {"ID", "day", "equity", *RETURN_COLUMNS}
    if set(features.columns) != expected or "reod" in features.columns:
        raise AssertionError("D003 requires exactly the physical Discovery input schema.")
    if not features["day"].between(0, 352).all():
        raise AssertionError("D003 may not access days outside 0 through 352.")
    values = features.loc[:, RETURN_COLUMNS].to_numpy(dtype=float)
    observed = ~np.isnan(values)
    if not np.isfinite(values[observed]).all():
        raise AssertionError("D003 observed returns must be finite.")


def assert_e_dev_only(features: pd.DataFrame, partition: pd.DataFrame) -> None:
    """Require physical Discovery membership to be a subset of frozen E_dev."""
    if set(partition.columns) != {"equity", "partition"}:
        raise AssertionError("D003 frozen equity partition has an unexpected schema.")
    dev_equities = set(partition.loc[partition["partition"].eq("E_dev"), "equity"])
    observed_equities = set(features["equity"])
    if not observed_equities.issubset(dev_equities):
        raise AssertionError("D003 physical Discovery input contains a non-E_dev equity.")


def build_common_component(features: pd.DataFrame) -> pd.DataFrame:
    """Build all-row, same-day medians without targets or cross-day pooling."""
    validate_discovery_features(features)
    daily_medians = features.loc[:, RETURN_COLUMNS].groupby(features["day"], sort=True).median()
    by_row = daily_medians.reindex(features["day"].to_numpy()).set_axis(features.index)
    by_row.columns = RETURN_COLUMNS
    return by_row


def build_relative_returns(features: pd.DataFrame) -> pd.DataFrame:
    """Construct u = r - same-day all-row median and assert mask identity."""
    common = build_common_component(features)
    own = features.loc[:, RETURN_COLUMNS]
    relative = own - common
    if not relative.index.equals(own.index) or list(relative.columns) != list(RETURN_COLUMNS):
        raise AssertionError("D003 relative path changed row identity or positional schema.")
    if not relative.isna().equals(own.isna()):
        raise AssertionError("D003 relative missingness must equal own-return missingness exactly.")
    values = relative.to_numpy(dtype=float)
    observed = ~np.isnan(values)
    if not np.isfinite(values[observed]).all():
        raise AssertionError("D003 observed relative returns must be finite.")
    return relative


def split_discovery_fold(features: pd.DataFrame, reod: pd.Series, fold: FoldSpec) -> DiscoverySubsets:
    """Split only the physical Discovery boundary into the frozen chronology."""
    validate_discovery_features(features)
    if not features.index.equals(reod.index):
        raise AssertionError("D003 input features and labels must be index-aligned.")
    fit = features["day"].between(fold.train_start, fold.train_end)
    validation = features["day"].between(fold.oos_start, fold.oos_end)
    subsets = DiscoverySubsets(
        fold, features.loc[fit].copy(), reod.loc[fit].copy(),
        features.loc[validation].copy(), reod.loc[validation].copy(),
    )
    if subsets.fit_features.empty or subsets.validation_features.empty:
        raise AssertionError("D003 Discovery fold has an empty subset.")
    if not subsets.fit_features["day"].between(fold.train_start, fold.train_end).all():
        raise AssertionError("D003 Fit day boundary violation.")
    if not subsets.validation_features["day"].between(fold.oos_start, fold.oos_end).all():
        raise AssertionError("D003 Validation day boundary violation.")
    if set(subsets.fit_features["ID"]) & set(subsets.validation_features["ID"]):
        raise AssertionError("D003 Fit and Validation subsets share IDs.")
    return subsets


def prepare_directional_subset(
    features: pd.DataFrame, reod: pd.Series, relative_all: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, int]]:
    """Apply EXP_005 directional/evaluability after target-free relative construction."""
    prepared = prepare_evaluable_directional_rows(features, reod)
    raw_own = prepared.raw_returns
    raw_relative = relative_all.loc[raw_own.index, RETURN_COLUMNS].copy()
    if not raw_relative.isna().equals(raw_own.isna()):
        raise AssertionError("D003 retained relative and own availability masks must be identical.")
    return raw_own, raw_relative, prepared.sign_target, prepared.manifest


def build_d003_matrices(
    own_transformer: PositionalPathTransformer,
    relative_transformer: PositionalPathTransformer,
    raw_own: pd.DataFrame,
    raw_relative: pd.DataFrame,
    target: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Build aligned A/B; B adds only separate fit-preprocessed relatives."""
    own_matrix = build_path_mask(own_transformer, raw_own)
    relative_scaled = relative_transformer.transform_return_block(raw_relative)
    relative_scaled.columns = RELATIVE_SCALED_COLUMNS
    candidate = pd.concat([own_matrix, relative_scaled], axis=1)
    if tuple(own_matrix.columns) != CONTROL_COLUMNS or tuple(candidate.columns) != CANDIDATE_COLUMNS:
        raise AssertionError("D003 A/B predictor schemas are not frozen.")
    if not own_matrix.index.equals(candidate.index) or not own_matrix.index.equals(target.index):
        raise AssertionError("D003 A/B/target rows must be identical and ordered.")
    if not np.isfinite(candidate.to_numpy(dtype=float)).all():
        raise AssertionError("D003 candidate predictors must be finite after fit-only preprocessing.")
    if not own_matrix.equals(candidate.loc[:, CONTROL_COLUMNS]):
        raise AssertionError("D003 B must differ from A only by relative numeric positions.")
    return {"A": own_matrix, "B": candidate}


def evaluate_d003_model(target: pd.Series, matrix: pd.DataFrame, model) -> tuple[dict[str, float], list[list[int]]]:
    """Evaluate a fitted D003 model with frozen EXP_005 sign utilities."""
    assert_both_sign_classes(target, context="D003 evaluation")
    probability = positive_probability(model, matrix)
    prediction = threshold_predictions(probability)
    return evaluate_sign_predictions(target, prediction, probability=probability)


def descriptive_candidate_rule(deltas: Sequence[float]) -> dict[str, bool | float | int]:
    """Record recurrence only; materiality remains Human + Sol review."""
    if len(deltas) != 3:
        raise AssertionError("D003 requires exactly three Discovery folds.")
    values = np.asarray(deltas, dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("D003 AUC deltas must be finite.")
    return {
        "n_discovery_folds": 3,
        "delta_auc_positive_all_three": bool((values > 0.0).all()),
        "mean_delta_auc": float(values.mean()),
        "human_sol_materiality_review_required": True,
    }


__all__ = [
    "CANDIDATE_COLUMNS", "CONTROL_COLUMNS", "DISCOVERY_FOLDS", "RELATIVE_SCALED_COLUMNS",
    "assert_e_dev_only",
    "build_common_component", "build_d003_matrices", "build_relative_returns",
    "descriptive_candidate_rule", "evaluate_d003_model", "prepare_directional_subset",
    "split_discovery_fold", "validate_discovery_features", "make_positional_logistic_regression",
]

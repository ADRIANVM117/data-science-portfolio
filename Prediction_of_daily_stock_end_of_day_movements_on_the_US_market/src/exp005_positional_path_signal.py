"""Utilities for frozen EXP_005 positional-path signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.exp001_missingness import COMPARISON_ATOL, LOGISTIC_REGRESSION_PARAMS, RETURN_COLUMNS, strictly_greater
from src.exp003_conditional_directional_sign import (
    DIRECTIONAL_REOD_VALUES,
    SIGN_CLASS_ORDER,
    assert_both_sign_classes,
    build_sign_target,
    evaluate_sign_majority_baseline,
    evaluate_sign_predictions,
    positive_probability,
    sign_majority_class,
    threshold_predictions,
)


MASK_COLUMNS: tuple[str, ...] = tuple(f"m{position}" for position in range(53))
SCALED_RETURN_COLUMNS: tuple[str, ...] = tuple(f"x_r{position}" for position in range(53))
PATH_MASK_COLUMNS: tuple[str, ...] = SCALED_RETURN_COLUMNS + MASK_COLUMNS


@dataclass(frozen=True)
class EvaluableDirectionalRows:
    """Raw retained returns and target after EXP_005's fixed population rules."""

    raw_returns: pd.DataFrame
    sign_target: pd.Series
    manifest: dict[str, int]


def _validate_return_columns(frame: pd.DataFrame) -> None:
    missing = set(RETURN_COLUMNS).difference(frame.columns)
    if missing:
        raise AssertionError(f"Missing EXP_005 return columns: {sorted(missing)}")


def prepare_evaluable_directional_rows(features: pd.DataFrame, reod: pd.Series) -> EvaluableDirectionalRows:
    """Apply frozen split-independent D then N_obs>=1 rules without features."""
    if not features.index.equals(reod.index):
        raise AssertionError("Features and reod must have identical indices.")
    _validate_return_columns(features)
    if reod.isna().any() or not set(reod.unique()).issubset({-1, 0, 1}):
        raise AssertionError("Unexpected reod labels before EXP_005 conditioning.")

    directional = reod.isin(DIRECTIONAL_REOD_VALUES)
    directional_returns = features.loc[directional, RETURN_COLUMNS].copy()
    directional_reod = reod.loc[directional]
    n_obs = directional_returns.notna().sum(axis=1).astype("int8")
    evaluable = n_obs.ge(1)
    retained_returns = directional_returns.loc[evaluable].copy()
    retained_reod = directional_reod.loc[evaluable]
    target = build_sign_target(retained_reod)

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
    if manifest["n_directional_before_evaluability"] != (
        manifest["n_all_nan_directional_excluded"] + manifest["n_evaluable_directional_retained"]
    ):
        raise AssertionError("EXP_005 evaluability counts do not reconcile.")
    if not retained_returns.index.equals(target.index) or retained_returns.empty:
        raise AssertionError("EXP_005 retained rows and target are not aligned.")
    if retained_returns.isna().all(axis=1).any() or not retained_reod.isin(DIRECTIONAL_REOD_VALUES).all():
        raise AssertionError("Neutral or all-NaN row retained in EXP_005.")
    return EvaluableDirectionalRows(retained_returns, target, manifest)


def build_original_mask(raw_returns: pd.DataFrame) -> pd.DataFrame:
    """Construct the unscaled original-data mask before any imputation."""
    _validate_return_columns(raw_returns)
    mask = raw_returns.loc[:, RETURN_COLUMNS].isna().astype("int8")
    mask.columns = MASK_COLUMNS
    validate_mask_matrix(mask, expected_index=raw_returns.index)
    return mask


def validate_mask_matrix(matrix: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    if list(matrix.columns) != list(MASK_COLUMNS):
        raise AssertionError("EXP_005 A_mask must contain exactly m0 through m52.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("EXP_005 A_mask changed row identity or order.")
    values = matrix.to_numpy()
    if matrix.isna().any().any() or not np.isin(values, [0, 1]).all():
        raise AssertionError("EXP_005 masks must be binary and finite.")


@dataclass
class PositionalPathTransformer:
    """Fit-only median imputation and StandardScaler for the 53 return positions."""

    medians_: pd.Series | None = None
    scaler_: StandardScaler | None = None
    fitted_: bool = False

    def fit(self, fit_raw_returns: pd.DataFrame) -> "PositionalPathTransformer":
        _validate_return_columns(fit_raw_returns)
        fit_returns = fit_raw_returns.loc[:, RETURN_COLUMNS]
        self.medians_ = fit_returns.median(axis=0)
        if not np.isfinite(self.medians_.to_numpy(dtype=float)).all():
            raise AssertionError(
                "EXP_005 integrity/evaluability failure: every fit-fold positional median must be finite."
            )
        imputed = fit_returns.fillna(self.medians_)
        if imputed.isna().any().any() or not np.isfinite(imputed.to_numpy(dtype=float)).all():
            raise AssertionError("EXP_005 fit-only median imputation produced non-finite returns.")
        self.scaler_ = StandardScaler().fit(imputed)
        if (
            self.scaler_.n_features_in_ != len(RETURN_COLUMNS)
            or not np.isfinite(self.scaler_.mean_).all()
            or not np.isfinite(self.scaler_.scale_).all()
            or not (self.scaler_.scale_ > 0.0).all()
        ):
            raise AssertionError("EXP_005 fit-only StandardScaler is invalid.")
        self.fitted_ = True
        return self

    def transform_return_block(self, raw_returns: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or self.medians_ is None or self.scaler_ is None:
            raise AssertionError("EXP_005 transformer must be fitted on fit rows before OOS transform.")
        _validate_return_columns(raw_returns)
        values = raw_returns.loc[:, RETURN_COLUMNS].fillna(self.medians_)
        if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise AssertionError("EXP_005 transform has non-finite imputed returns.")
        transformed = self.scaler_.transform(values)
        if not np.isfinite(transformed).all():
            raise AssertionError("EXP_005 StandardScaler transform produced non-finite returns.")
        return pd.DataFrame(transformed, index=raw_returns.index, columns=SCALED_RETURN_COLUMNS)

    def parameter_rows(self, *, fold: int) -> list[dict[str, int | float | str | bool]]:
        if not self.fitted_ or self.medians_ is None or self.scaler_ is None:
            raise AssertionError("Cannot report EXP_005 preprocessing parameters before fit.")
        return [
            {
                "fold": fold,
                "return_column": column,
                "fit_median": float(self.medians_.loc[column]),
                "scaler_mean": float(self.scaler_.mean_[position]),
                "scaler_scale": float(self.scaler_.scale_[position]),
                "fit_median_finite": bool(np.isfinite(self.medians_.loc[column])),
            }
            for position, column in enumerate(RETURN_COLUMNS)
        ]


def build_path_mask(transformer: PositionalPathTransformer, raw_returns: pd.DataFrame) -> pd.DataFrame:
    """Construct the exact 106-column B_path_mask representation."""
    mask = build_original_mask(raw_returns)
    scaled_returns = transformer.transform_return_block(raw_returns)
    matrix = pd.concat([scaled_returns, mask], axis=1)
    validate_path_mask_matrix(matrix, expected_index=raw_returns.index)
    return matrix


def validate_path_mask_matrix(matrix: pd.DataFrame, *, expected_index: pd.Index | None = None) -> None:
    if list(matrix.columns) != list(PATH_MASK_COLUMNS):
        raise AssertionError("EXP_005 B_path_mask must contain exactly 53 returns then 53 masks.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("EXP_005 B_path_mask changed row identity or order.")
    return_block = matrix.loc[:, SCALED_RETURN_COLUMNS]
    mask_block = matrix.loc[:, MASK_COLUMNS]
    if return_block.isna().any().any() or not np.isfinite(return_block.to_numpy(dtype=float)).all():
        raise AssertionError("EXP_005 scaled return block must be finite.")
    validate_mask_matrix(mask_block, expected_index=matrix.index)


def assert_identical_representation_rows(a_mask: pd.DataFrame, b_path_mask: pd.DataFrame, target: pd.Series) -> None:
    if not a_mask.index.equals(b_path_mask.index) or not a_mask.index.equals(target.index):
        raise AssertionError("EXP_005 A_mask, B_path_mask, and target must retain identical row identities.")


def make_positional_logistic_regression() -> LogisticRegression:
    classifier = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    for name, expected in LOGISTIC_REGRESSION_PARAMS.items():
        if classifier.get_params()[name] != expected:
            raise AssertionError(f"LogisticRegression parameter {name!r} differs from EXP_005.")
    return classifier


def delta_vs_baseline(value: float, reference: float) -> float:
    delta = value - reference
    return 0.0 if np.isclose(delta, 0.0, rtol=0.0, atol=COMPARISON_ATOL) else float(delta)


def primary_fold_decision(path_auc: float, path_minus_mask_auc: float) -> dict[str, bool]:
    return {
        "path_roc_auc_above_one_half": strictly_greater(path_auc, 0.5),
        "path_minus_mask_auc_positive": strictly_greater(path_minus_mask_auc, 0.0),
    }


def aggregate_primary_decision(path_aucs: Sequence[float], path_minus_mask_aucs: Sequence[float]) -> dict[str, float | bool | int]:
    if len(path_aucs) != 4 or len(path_minus_mask_aucs) != 4:
        raise AssertionError("EXP_005 requires exactly four Joint OOS folds.")
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
        "incremental_positional_return_evidence": (
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


def aggregate_secondary_decision(fold_decisions: Sequence[Mapping[str, bool]], path_ba_deltas: Sequence[float]) -> dict[str, float | bool | int]:
    if len(fold_decisions) != 4 or len(path_ba_deltas) != 4:
        raise AssertionError("EXP_005 requires exactly four Joint OOS folds.")
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


__all__ = [
    "EvaluableDirectionalRows",
    "MASK_COLUMNS",
    "PATH_MASK_COLUMNS",
    "PositionalPathTransformer",
    "SCALED_RETURN_COLUMNS",
    "aggregate_primary_decision",
    "aggregate_secondary_decision",
    "assert_both_sign_classes",
    "assert_identical_representation_rows",
    "build_original_mask",
    "build_path_mask",
    "delta_vs_baseline",
    "evaluate_sign_majority_baseline",
    "evaluate_sign_predictions",
    "make_positional_logistic_regression",
    "positive_probability",
    "prepare_evaluable_directional_rows",
    "primary_fold_decision",
    "secondary_fold_decision",
    "sign_majority_class",
    "threshold_predictions",
    "validate_mask_matrix",
    "validate_path_mask_matrix",
]

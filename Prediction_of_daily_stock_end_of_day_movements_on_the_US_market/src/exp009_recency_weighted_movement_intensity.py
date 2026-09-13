"""Frozen feature and decision utilities for EXP_009.

The module contains no data loading.  It implements the complete-W isolation
population and the two pre-specified recent movement-intensity features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.exp001_missingness import strictly_greater
from src.exp002_neutral_directional import delta_vs_baseline


RECENT_WINDOW_COLUMNS: tuple[str, ...] = tuple(f"r{position}" for position in range(41, 53))
RECENT_WINDOW_SIZE = 12
RECENCY_WEIGHTS = np.arange(1, RECENT_WINDOW_SIZE + 1, dtype=np.float64)
RECENCY_WEIGHT_SUM = float(RECENCY_WEIGHTS.sum())
REPRESENTATION_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "C": ("missing_ratio", "I_recent_z"),
    "D": ("missing_ratio", "I_recent_z", "I_recency_z"),
}


def _availability_values(frame: pd.DataFrame) -> np.ndarray:
    """Read W after validating only positional schema and numeric coercion.

    Eligibility is intentionally based on NaN availability before finite-value
    checks.  Thus an infinity is observed, may make a row complete-W, and then
    causes an explicit retained-row integrity failure rather than being treated
    as missing.
    """
    if RECENT_WINDOW_COLUMNS != tuple(f"r{position}" for position in range(41, 53)):
        raise AssertionError("EXP_009 W must be exactly r41 through r52.")
    if len(RECENT_WINDOW_COLUMNS) != RECENT_WINDOW_SIZE:
        raise AssertionError("EXP_009 W must contain exactly 12 positions.")
    if not np.array_equal(RECENCY_WEIGHTS, np.arange(1, 13, dtype=np.float64)):
        raise AssertionError("EXP_009 recency weights must be exactly [1, ..., 12].")
    if RECENCY_WEIGHT_SUM != 78.0:
        raise AssertionError("EXP_009 recency weight sum must be exactly 78.")
    missing = set(RECENT_WINDOW_COLUMNS).difference(frame.columns)
    if missing:
        raise AssertionError(f"Missing EXP_009 recent-window columns: {sorted(missing)}")
    try:
        return frame.loc[:, RECENT_WINDOW_COLUMNS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AssertionError("EXP_009 W values must be numeric.") from error


def build_complete_window_structure(frame: pd.DataFrame) -> pd.DataFrame:
    """Build target-free NaN availability and complete-W membership only."""
    values = _availability_values(frame)
    n_obs = (~np.isnan(values)).sum(axis=1).astype("int16")
    structure = pd.DataFrame(
        {"N_obs_w": n_obs, "complete_w": n_obs == RECENT_WINDOW_SIZE}, index=frame.index
    )
    validate_complete_window_structure(structure, expected_index=frame.index)
    return structure


def validate_complete_window_structure(
    structure: pd.DataFrame, *, expected_index: pd.Index | None = None
) -> None:
    """Validate that complete-W is defined strictly from NaN availability."""
    if list(structure.columns) != ["N_obs_w", "complete_w"]:
        raise AssertionError("EXP_009 complete-W structure has an unexpected schema.")
    if expected_index is not None and not structure.index.equals(expected_index):
        raise AssertionError("EXP_009 complete-W structure changed row identity or order.")
    n_obs = structure["N_obs_w"].to_numpy(dtype=int)
    complete = structure["complete_w"].to_numpy(dtype=bool)
    if ((n_obs < 0) | (n_obs > RECENT_WINDOW_SIZE)).any():
        raise AssertionError("EXP_009 N_obs_w must be between zero and 12.")
    if not np.array_equal(complete, n_obs == RECENT_WINDOW_SIZE):
        raise AssertionError("EXP_009 complete_w must equal N_obs_w == 12.")


def filter_complete_window_rows(
    features: pd.DataFrame, target: pd.Series, structure: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Retain complete-W rows and then reject non-finite observed values."""
    if not features.index.equals(target.index) or not features.index.equals(structure.index):
        raise AssertionError("EXP_009 features, target, and complete-W structure must align.")
    validate_complete_window_structure(structure, expected_index=features.index)
    complete_index = structure.index[structure["complete_w"]]
    retained_features = features.loc[complete_index].copy()
    retained_target = target.loc[complete_index].copy()
    if retained_features.empty:
        raise AssertionError("EXP_009 complete-W filtering produced an empty subset.")
    retained_structure = structure.loc[complete_index]
    if not (retained_structure["N_obs_w"] == RECENT_WINDOW_SIZE).all():
        raise AssertionError("EXP_009 retained a row with N_obs_w different from 12.")
    values = _availability_values(retained_features)
    if np.isnan(values).any():
        raise AssertionError("EXP_009 retained complete-W rows cannot contain NaN values.")
    if not np.isfinite(values).all():
        raise AssertionError("EXP_009 retained observed W values must be finite.")
    return retained_features, retained_target


def build_raw_intensities(complete_features: pd.DataFrame) -> pd.DataFrame:
    """Calculate the frozen unweighted and linearly recency-weighted RMS values."""
    values = _availability_values(complete_features)
    if np.isnan(values).any():
        raise AssertionError("EXP_009 intensity calculation requires complete W rows.")
    if not np.isfinite(values).all():
        raise AssertionError("EXP_009 intensity calculation requires finite W values.")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            squared = np.square(values)
            recent = np.sqrt(squared.sum(axis=1) / RECENT_WINDOW_SIZE)
            recency = np.sqrt((squared * RECENCY_WEIGHTS).sum(axis=1) / RECENCY_WEIGHT_SUM)
    except FloatingPointError as error:
        raise AssertionError("EXP_009 intensity calculation was non-finite.") from error
    raw = pd.DataFrame({"raw_I_recent": recent, "raw_I_recency": recency}, index=complete_features.index)
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise AssertionError("EXP_009 raw intensities must be finite for complete-W rows.")
    return raw


@dataclass
class CompleteWindowIntensityTransformer:
    """Independent fit-only scalers for EXP_009's two raw intensity features."""

    scaler_recent_: StandardScaler | None = field(default=None, repr=False)
    scaler_recency_: StandardScaler | None = field(default=None, repr=False)

    @staticmethod
    def _validated_column(raw: pd.Series, *, name: str) -> np.ndarray:
        values = raw.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError(f"EXP_009 {name} must be finite; no imputation is permitted.")
        return values.reshape(-1, 1)

    def fit(self, raw: pd.DataFrame) -> "CompleteWindowIntensityTransformer":
        if list(raw.columns) != ["raw_I_recent", "raw_I_recency"] or raw.empty:
            raise AssertionError("EXP_009 fit requires both non-empty raw intensity columns.")
        recent = self._validated_column(raw["raw_I_recent"], name="raw_I_recent")
        recency = self._validated_column(raw["raw_I_recency"], name="raw_I_recency")
        self.scaler_recent_ = StandardScaler().fit(recent)
        self.scaler_recency_ = StandardScaler().fit(recency)
        for scaler_name, scaler in (("recent", self.scaler_recent_), ("recency", self.scaler_recency_)):
            if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
                raise AssertionError(f"EXP_009 fit-only {scaler_name} scaler parameters must be finite.")
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_recent_ is None or self.scaler_recency_ is None:
            raise AssertionError("EXP_009 intensity scalers must be fit before transform.")
        if list(raw.columns) != ["raw_I_recent", "raw_I_recency"]:
            raise AssertionError("EXP_009 transform requires both raw intensity columns.")
        recent = self._validated_column(raw["raw_I_recent"], name="raw_I_recent")
        recency = self._validated_column(raw["raw_I_recency"], name="raw_I_recency")
        transformed = pd.DataFrame(
            {
                "I_recent_z": self.scaler_recent_.transform(recent).reshape(-1),
                "I_recency_z": self.scaler_recency_.transform(recency).reshape(-1),
            },
            index=raw.index,
        )
        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise AssertionError("EXP_009 transformed intensity values must be finite.")
        return transformed


def build_exp009_representations(
    missing_ratio: pd.DataFrame, transformed_intensities: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Build the frozen C/D matrices from one retained row order."""
    if list(missing_ratio.columns) != ["missing_ratio"]:
        raise AssertionError("EXP_009 requires the exact EXP_002 missing_ratio matrix.")
    if list(transformed_intensities.columns) != ["I_recent_z", "I_recency_z"]:
        raise AssertionError("EXP_009 transformed intensity schema is unexpected.")
    if not transformed_intensities.index.equals(missing_ratio.index):
        raise AssertionError("EXP_009 matrices changed row identity or order.")
    representations = {
        "C": pd.concat([missing_ratio, transformed_intensities.loc[:, ["I_recent_z"]]], axis=1),
        "D": pd.concat([missing_ratio, transformed_intensities], axis=1),
    }
    for name, matrix in representations.items():
        validate_exp009_representation(name, matrix, expected_index=missing_ratio.index)
    return representations


def validate_exp009_representation(
    name: str, matrix: pd.DataFrame, *, expected_index: pd.Index | None = None
) -> None:
    """Validate exact schemas and the exclusion of q/prohibited predictors."""
    if name not in REPRESENTATION_COLUMNS:
        raise AssertionError(f"Unknown EXP_009 representation: {name}")
    if list(matrix.columns) != list(REPRESENTATION_COLUMNS[name]):
        raise AssertionError(f"EXP_009 {name} has an unexpected predictor schema.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError(f"EXP_009 {name} changed row identity or order.")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"EXP_009 {name} contains a non-finite predictor.")
    ratio = matrix["missing_ratio"].to_numpy(dtype=float)
    if ((ratio < 0.0) | (ratio > 1.0)).any():
        raise AssertionError("EXP_009 missing_ratio must remain in [0, 1].")
    if set(matrix.columns).intersection({"q", "ID", "day", "equity", "N_obs_w", "raw_I_recent", "raw_I_recency"}):
        raise AssertionError("EXP_009 prohibited a non-predictor from entering a matrix.")


def assert_identical_representation_rows(
    representations: Mapping[str, pd.DataFrame], features: pd.DataFrame, target: pd.Series
) -> None:
    """Require C/D to use the same retained source rows, targets, and order."""
    if not features.index.equals(target.index):
        raise AssertionError("EXP_009 features and target are not aligned.")
    if features["ID"].duplicated().any():
        raise AssertionError("EXP_009 retained source rows contain duplicate IDs.")
    for name in REPRESENTATION_COLUMNS:
        if name not in representations:
            raise AssertionError(f"EXP_009 representation {name} is missing.")
        matrix = representations[name]
        validate_exp009_representation(name, matrix, expected_index=features.index)
        if len(matrix) != len(target):
            raise AssertionError("EXP_009 C/D must retain identical rows.")


def primary_fold_decision(d_auc: float, recency_auc_delta: float) -> dict[str, bool]:
    """Apply the two strict per-fold primary conditions for EXP_009."""
    decision = {
        "d_roc_auc_above_one_half": strictly_greater(d_auc, 0.5),
        "recency_auc_delta_positive": strictly_greater(recency_auc_delta, 0.0),
    }
    decision["all_fold_primary_conditions_met"] = all(decision.values())
    return decision


def aggregate_primary_decision(
    d_aucs: Sequence[float], recency_auc_deltas: Sequence[float]
) -> dict[str, float | bool | int]:
    """Apply EXP_009's frozen four-condition primary verdict."""
    if len(d_aucs) != 4 or len(recency_auc_deltas) != 4:
        raise AssertionError("EXP_009 requires exactly four Joint OOS folds.")
    mean_d_auc = float(np.mean(d_aucs))
    mean_delta = float(np.mean(recency_auc_deltas))
    all_d = all(strictly_greater(value, 0.5) for value in d_aucs)
    all_delta = all(strictly_greater(value, 0.0) for value in recency_auc_deltas)
    return {
        "n_joint_folds": 4,
        "all_four_d_roc_auc_above_one_half": all_d,
        "mean_joint_d_roc_auc": mean_d_auc,
        "mean_joint_d_roc_auc_above_one_half": strictly_greater(mean_d_auc, 0.5),
        "all_four_recency_auc_delta_positive": all_delta,
        "mean_joint_recency_auc_delta": mean_delta,
        "mean_joint_recency_auc_delta_positive": strictly_greater(mean_delta, 0.0),
        "incremental_recency_weighted_intensity_evidence": all_d
        and strictly_greater(mean_d_auc, 0.5)
        and all_delta
        and strictly_greater(mean_delta, 0.0),
    }


def auc_delta(value: float, reference: float) -> float:
    """Return a stable model-minus-control AUC difference."""
    return delta_vs_baseline(value, reference)

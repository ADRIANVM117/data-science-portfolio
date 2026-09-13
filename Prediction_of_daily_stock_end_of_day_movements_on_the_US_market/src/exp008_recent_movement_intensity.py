"""Frozen feature and decision utilities for EXP_008.

This module implements only the positional recent-window representation
specified in ``experiments/EXP_008_recent_movement_intensity.md``.  It has no
data loading or competition-test dependency.
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
REPRESENTATION_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "A": ("missing_ratio",),
    "C": ("missing_ratio", "q"),
    "B": ("missing_ratio", "q", "I_recent_z"),
}


def _window_values(frame: pd.DataFrame) -> np.ndarray:
    """Return W values after enforcing the frozen positional schema."""
    if tuple(RECENT_WINDOW_COLUMNS) != tuple(f"r{position}" for position in range(41, 53)):
        raise AssertionError("EXP_008 W must be exactly r41 through r52.")
    if len(RECENT_WINDOW_COLUMNS) != RECENT_WINDOW_SIZE:
        raise AssertionError("EXP_008 W must contain exactly 12 positions.")
    missing = set(RECENT_WINDOW_COLUMNS).difference(frame.columns)
    if missing:
        raise AssertionError(f"Missing EXP_008 recent-window columns: {sorted(missing)}")
    try:
        values = frame.loc[:, RECENT_WINDOW_COLUMNS].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise AssertionError("EXP_008 W values must be numeric.") from error
    observed = ~np.isnan(values)
    if not np.isfinite(values[observed]).all():
        raise AssertionError("Observed non-NaN EXP_008 W values must be finite.")
    return values


def build_recent_window_structure(frame: pd.DataFrame) -> pd.DataFrame:
    """Build target-free N_obs,W, q, and raw RMS intensity for W.

    A raw intensity is deliberately NaN when all 12 W values are NaN.  This
    is the pre-imputation state required by the frozen contract.
    """
    values = _window_values(frame)
    observed = ~np.isnan(values)
    n_obs = observed.sum(axis=1).astype("int16")
    q = (n_obs == 0).astype("int8")
    raw_intensity = np.full(len(frame), np.nan, dtype=float)
    defined = n_obs > 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            observed_values = np.where(observed, values, 0.0)
            squared_sum = np.square(observed_values).sum(axis=1)
            raw_intensity[defined] = np.sqrt(squared_sum[defined] / n_obs[defined])
    except FloatingPointError as error:
        raise AssertionError("EXP_008 raw intensity computation was non-finite.") from error

    structure = pd.DataFrame(
        {"N_obs_w": n_obs, "q": q, "raw_I_recent": raw_intensity}, index=frame.index
    )
    validate_recent_window_structure(structure, expected_index=frame.index)
    return structure


def validate_recent_window_structure(
    structure: pd.DataFrame, *, expected_index: pd.Index | None = None
) -> None:
    """Validate the target-free structural quantities before preprocessing."""
    if list(structure.columns) != ["N_obs_w", "q", "raw_I_recent"]:
        raise AssertionError("EXP_008 recent-window structure has an unexpected schema.")
    if expected_index is not None and not structure.index.equals(expected_index):
        raise AssertionError("EXP_008 recent-window structure changed row identity or order.")
    n_obs = structure["N_obs_w"].to_numpy(dtype=int)
    q = structure["q"].to_numpy(dtype=int)
    raw = structure["raw_I_recent"].to_numpy(dtype=float)
    if ((n_obs < 0) | (n_obs > RECENT_WINDOW_SIZE)).any():
        raise AssertionError("N_obs_w must be between zero and 12.")
    if not np.isin(q, [0, 1]).all() or not np.array_equal(q, (n_obs == 0).astype(int)):
        raise AssertionError("q must equal one if and only if N_obs_w is zero.")
    if not np.isnan(raw[n_obs == 0]).all():
        raise AssertionError("raw_I_recent must be undefined before imputation when N_obs_w is zero.")
    if not np.isfinite(raw[n_obs > 0]).all():
        raise AssertionError("raw_I_recent must be finite whenever N_obs_w is positive.")


@dataclass
class RecentIntensityTransformer:
    """Fit-only median imputation and scaling for EXP_008 raw intensity."""

    fit_intensity_median_: float | None = None
    scaler_: StandardScaler | None = field(default=None, repr=False)

    @staticmethod
    def _validate_raw(raw_intensity: pd.Series) -> np.ndarray:
        values = raw_intensity.to_numpy(dtype=float)
        finite_or_nan = np.isfinite(values) | np.isnan(values)
        if not finite_or_nan.all():
            raise AssertionError("raw_I_recent contains a non-finite non-NaN value.")
        return values

    def fit(self, raw_intensity: pd.Series) -> "RecentIntensityTransformer":
        """Learn the median and scaler only from the supplied fit intensities."""
        values = self._validate_raw(raw_intensity)
        defined = ~np.isnan(values)
        if not defined.any():
            raise AssertionError("EXP_008 fit has no defined raw intensity for its fit-only median.")
        median = float(np.median(values[defined]))
        if not np.isfinite(median):
            raise AssertionError("EXP_008 fit intensity median must be finite.")
        imputed = np.where(defined, values, median).reshape(-1, 1)
        scaler = StandardScaler().fit(imputed)
        if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
            raise AssertionError("EXP_008 fit intensity scaler parameters must be finite.")
        self.fit_intensity_median_ = median
        self.scaler_ = scaler
        return self

    def transform(self, raw_intensity: pd.Series) -> pd.DataFrame:
        """Impute and scale an aligned fit or OOS raw-intensity series."""
        if self.fit_intensity_median_ is None or self.scaler_ is None:
            raise AssertionError("EXP_008 intensity transformer must be fit before transform.")
        values = self._validate_raw(raw_intensity)
        imputed = np.where(np.isnan(values), self.fit_intensity_median_, values).reshape(-1, 1)
        transformed = self.scaler_.transform(imputed).reshape(-1)
        if not np.isfinite(transformed).all():
            raise AssertionError("EXP_008 transformed intensity must be finite.")
        return pd.DataFrame({"I_recent_z": transformed}, index=raw_intensity.index)


def build_exp008_representations(
    missing_ratio: pd.DataFrame,
    structure: pd.DataFrame,
    intensity_z: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build A/C/B using exactly the frozen predictor sets and one row order."""
    if list(missing_ratio.columns) != ["missing_ratio"]:
        raise AssertionError("EXP_008 requires the exact EXP_002 missing_ratio matrix.")
    if list(intensity_z.columns) != ["I_recent_z"]:
        raise AssertionError("EXP_008 intensity transform has an unexpected schema.")
    for frame in (structure, intensity_z):
        if not frame.index.equals(missing_ratio.index):
            raise AssertionError("EXP_008 representations changed row identity or order.")
    representations = {
        "A": missing_ratio.copy(),
        "C": pd.concat([missing_ratio, structure.loc[:, ["q"]]], axis=1),
        "B": pd.concat([missing_ratio, structure.loc[:, ["q"]], intensity_z], axis=1),
    }
    for name, matrix in representations.items():
        validate_exp008_representation(name, matrix, expected_index=missing_ratio.index)
    return representations


def validate_exp008_representation(
    name: str, matrix: pd.DataFrame, *, expected_index: pd.Index | None = None
) -> None:
    """Assert exact schema, values, and predictor exclusion for A/C/B."""
    if name not in REPRESENTATION_COLUMNS:
        raise AssertionError(f"Unknown EXP_008 representation: {name}")
    if list(matrix.columns) != list(REPRESENTATION_COLUMNS[name]):
        raise AssertionError(f"EXP_008 {name} has an unexpected predictor schema.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError(f"EXP_008 {name} changed row identity or order.")
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"EXP_008 {name} contains a non-finite predictor.")
    if "missing_ratio" in matrix and ((matrix["missing_ratio"] < 0.0) | (matrix["missing_ratio"] > 1.0)).any():
        raise AssertionError("EXP_008 missing_ratio must remain in [0, 1].")
    if "q" in matrix and not np.isin(matrix["q"].to_numpy(dtype=int), [0, 1]).all():
        raise AssertionError("EXP_008 q must be binary.")


def assert_identical_representation_rows(
    representations: Mapping[str, pd.DataFrame], features: pd.DataFrame, target: pd.Series
) -> None:
    """Require every representation to retain the exact frozen subset rows."""
    if not features.index.equals(target.index):
        raise AssertionError("EXP_008 features and target are not aligned.")
    for name in REPRESENTATION_COLUMNS:
        if name not in representations:
            raise AssertionError(f"EXP_008 representation {name} is missing.")
        matrix = representations[name]
        validate_exp008_representation(name, matrix, expected_index=features.index)
        if len(matrix) != len(target):
            raise AssertionError("EXP_008 representations must retain identical rows.")
        if set(matrix.columns).intersection({"ID", "day", "equity", "raw_I_recent", "N_obs_w"}):
            raise AssertionError("EXP_008 prohibited a non-predictor from entering a matrix.")


def primary_fold_decision(b_auc: float, intensity_auc_delta: float) -> dict[str, bool]:
    """Apply EXP_008's two strict fold-level primary conditions."""
    decision = {
        "b_roc_auc_above_one_half": strictly_greater(b_auc, 0.5),
        "intensity_auc_delta_positive": strictly_greater(intensity_auc_delta, 0.0),
    }
    decision["all_fold_primary_conditions_met"] = all(decision.values())
    return decision


def aggregate_primary_decision(
    b_aucs: Sequence[float], intensity_auc_deltas: Sequence[float]
) -> dict[str, float | bool | int]:
    """Apply the frozen four-fold B and B-C primary verdict."""
    if len(b_aucs) != 4 or len(intensity_auc_deltas) != 4:
        raise AssertionError("EXP_008 requires exactly four Joint OOS folds.")
    mean_b_auc = float(np.mean(b_aucs))
    mean_delta = float(np.mean(intensity_auc_deltas))
    all_b = all(strictly_greater(value, 0.5) for value in b_aucs)
    all_delta = all(strictly_greater(value, 0.0) for value in intensity_auc_deltas)
    return {
        "n_joint_folds": 4,
        "all_four_b_roc_auc_above_one_half": all_b,
        "mean_joint_b_roc_auc": mean_b_auc,
        "mean_joint_b_roc_auc_above_one_half": strictly_greater(mean_b_auc, 0.5),
        "all_four_intensity_auc_delta_positive": all_delta,
        "mean_joint_intensity_auc_delta": mean_delta,
        "mean_joint_intensity_auc_delta_positive": strictly_greater(mean_delta, 0.0),
        "incremental_recent_intensity_evidence": all_b
        and strictly_greater(mean_b_auc, 0.5)
        and all_delta
        and strictly_greater(mean_delta, 0.0),
    }


def auc_delta(value: float, reference: float) -> float:
    """Return a stable AUC difference with EXP_002 comparison semantics."""
    return delta_vs_baseline(value, reference)

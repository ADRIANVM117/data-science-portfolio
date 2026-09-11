"""Leakage-safe preprocessing utilities for intraday return sequences.

The competition data distinguishes an observed zero return from a missing
observation.  This module therefore preserves missing values by default and
exposes a separate missingness mask for downstream tree and sequence models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal, Sequence

import numpy as np
import pandas as pd


ImputationStrategy = Literal["none", "median"]
_RETURN_PATTERN = re.compile(r"r(\d+)")


def get_return_columns(columns: Sequence[str]) -> list[str]:
    """Return ``r0`` ... ``r52`` columns in chronological order.

    Raises:
        ValueError: If no columns with the expected return naming convention
            are present.
    """
    matched = [column for column in columns if _RETURN_PATTERN.fullmatch(column)]
    if not matched:
        raise ValueError("No return columns named r0, r1, ... were found.")
    return sorted(matched, key=lambda column: int(_RETURN_PATTERN.fullmatch(column).group(1)))


def load_training_data(
    data_dir: str | Path,
    input_filename: str = "input_training.csv",
    target_filename: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load features and labels, aligning them by ``ID`` rather than row order.

    If ``target_filename`` is omitted, the single file matching
    ``output_training_*.csv`` is used. The returned feature frame includes
    ``ID``, ``day`` and ``equity``; callers decide which identifiers to use.
    """
    data_path = Path(data_dir)
    feature_path = data_path / input_filename

    if target_filename is None:
        candidates = sorted(data_path.glob("output_training_*.csv"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Expected exactly one output_training_*.csv file; "
                f"found {len(candidates)} in {data_path}."
            )
        target_path = candidates[0]
    else:
        target_path = data_path / target_filename

    features = pd.read_csv(feature_path)
    targets = pd.read_csv(target_path)
    required_target_columns = {"ID", "reod"}
    if not required_target_columns.issubset(targets.columns):
        raise ValueError(f"Target file must contain {required_target_columns}.")
    if "ID" not in features.columns:
        raise ValueError("Feature file must contain an ID column.")

    labelled = features.merge(
        targets[["ID", "reod"]],
        on="ID",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(labelled) != len(features):
        raise ValueError("Some feature rows do not have exactly one target label.")

    target = labelled.pop("reod").astype("int8")
    target.name = "reod"
    return labelled, target


def build_missingness_mask(
    return_frame: pd.DataFrame,
    prefix: str = "missing_",
) -> pd.DataFrame:
    """Create one binary feature per return, where 1 denotes an original NaN."""
    mask = return_frame.isna().astype("int8")
    return mask.rename(columns={column: f"{prefix}{column}" for column in mask.columns})


def _longest_missing_streak(mask: np.ndarray) -> np.ndarray:
    """Calculate each row's longest consecutive sequence of missing values."""
    longest = np.zeros(mask.shape[0], dtype=np.int16)
    current = np.zeros(mask.shape[0], dtype=np.int16)
    for position in range(mask.shape[1]):
        current = np.where(mask[:, position], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def make_missingness_features(return_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize absence patterns without replacing the original NaNs.

    Segment features distinguish missingness near the market open, middle of
    the observed session, and the period immediately before 14:00.
    """
    raw_mask = return_frame.isna().to_numpy(dtype=bool)
    number_of_steps = return_frame.shape[1]
    first_cut = number_of_steps // 3
    second_cut = 2 * number_of_steps // 3

    features = pd.DataFrame(index=return_frame.index)
    features["n_missing"] = raw_mask.sum(axis=1).astype("int16")
    features["missing_ratio"] = raw_mask.mean(axis=1).astype("float32")
    features["longest_missing_streak"] = _longest_missing_streak(raw_mask)
    features["missing_open"] = raw_mask[:, :first_cut].mean(axis=1).astype("float32")
    features["missing_middle"] = raw_mask[:, first_cut:second_cut].mean(axis=1).astype("float32")
    features["missing_pre_close"] = raw_mask[:, second_cut:].mean(axis=1).astype("float32")
    return features


def summarize_missingness(return_frame: pd.DataFrame) -> pd.DataFrame:
    """Return a per-timestamp missingness audit table for a notebook/report."""
    summary = pd.DataFrame(
        {
            "missing_count": return_frame.isna().sum(),
            "missing_ratio": return_frame.isna().mean(),
            "zero_ratio_observed": return_frame.eq(0).mean(),
        }
    )
    summary.index.name = "return_column"
    return summary


@dataclass
class ReturnPreprocessor:
    """Fit return-value transformations on a training fold only.

    ``imputation='none'`` is the default for CatBoost/LightGBM, which can use
    missing values directly. Sequence models still receive a zero-filled copy
    through :meth:`sequence_inputs`, paired with the original missingness mask.
    """

    imputation: ImputationStrategy = "none"
    scale: bool = False
    clip_value: float | None = 12.0
    return_columns: list[str] = field(default_factory=list, init=False)
    medians_: pd.Series | None = field(default=None, init=False)
    scales_: pd.Series | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, train_frame: pd.DataFrame) -> "ReturnPreprocessor":
        """Learn medians and robust scales using only a training partition."""
        if self.imputation not in {"none", "median"}:
            raise ValueError("imputation must be either 'none' or 'median'.")

        self.return_columns = get_return_columns(train_frame.columns)
        returns = train_frame[self.return_columns]
        self.medians_ = returns.median(axis=0)

        if self.scale:
            q1 = returns.quantile(0.25, axis=0)
            q3 = returns.quantile(0.75, axis=0)
            iqr = (q3 - q1).replace(0, 1.0).fillna(1.0)
            self.scales_ = iqr

        self.fitted_ = True
        return self

    def transform_returns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform return columns while preserving row order and index."""
        if not self.fitted_:
            raise RuntimeError("Call fit on a training fold before transform.")
        missing_columns = set(self.return_columns).difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Frame is missing return columns: {sorted(missing_columns)}")

        transformed = frame[self.return_columns].copy()
        if self.imputation == "median":
            transformed = transformed.fillna(self.medians_)

        if self.scale:
            transformed = (transformed - self.medians_) / self.scales_
            if self.clip_value is not None:
                transformed = transformed.clip(-self.clip_value, self.clip_value)
        return transformed.astype("float32")

    def tabular_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return values plus missingness indicators and summary features."""
        raw_returns = frame[self.return_columns]
        transformed_returns = self.transform_returns(frame)
        return pd.concat(
            [
                transformed_returns,
                build_missingness_mask(raw_returns),
                make_missingness_features(raw_returns),
            ],
            axis=1,
        )

    def sequence_inputs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return values and an observed-value mask for a sequence model.

        NaNs are zero-filled only for numerical compatibility. The second
        array tells the network which values are observed (1) or missing (0).
        """
        raw_returns = frame[self.return_columns]
        values = self.transform_returns(frame).fillna(0.0).to_numpy(dtype=np.float32)
        observed_mask = (~raw_returns.isna()).to_numpy(dtype=np.float32)
        return values, observed_mask

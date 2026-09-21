"""Frozen D008 representation and decision utilities; no data loading."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from discovery.d007_raw_path_masks_ternary_xgboost import (
    MASK_COLUMNS,
    build_masks,
    primary_screen as d007_primary_screen,
)
from src.exp001_missingness import RETURN_COLUMNS
from src.exp008_recent_movement_intensity import (
    RECENT_WINDOW_COLUMNS,
    RECENT_WINDOW_SIZE,
    RecentIntensityTransformer,
    build_recent_window_structure,
)


C_COLUMNS: tuple[str, ...] = MASK_COLUMNS + ("I_recent_z",)
P_COLUMNS: tuple[str, ...] = C_COLUMNS + RETURN_COLUMNS
EXPECTED_ROW_COUNTS: Mapping[int, Mapping[str, int]] = {
    1: {"fit": 271_897, "validation": 67_031},
    2: {"fit": 338_928, "validation": 66_989},
    3: {"fit": 405_917, "validation": 66_899},
}


def fit_intensity_transformer(fit_features: pd.DataFrame) -> tuple[pd.DataFrame, RecentIntensityTransformer]:
    """Construct raw EXP_008 structure then fit its transformer on Fit only."""
    structure = build_recent_window_structure(fit_features)
    return structure, RecentIntensityTransformer().fit(structure["raw_I_recent"])


def build_d008_representations(
    features: pd.DataFrame,
    intensity_z: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build exact C/P matrices from unchanged raw rows and shared intensity."""
    if list(intensity_z.columns) != ["I_recent_z"] or not intensity_z.index.equals(features.index):
        raise AssertionError("D008 intensity must be exactly one aligned I_recent_z column.")
    if not np.isfinite(intensity_z.to_numpy(dtype=float)).all():
        raise AssertionError("D008 I_recent_z must be finite after fit-only preprocessing.")
    masks = build_masks(features)
    raw = features.loc[:, RETURN_COLUMNS].copy()
    matrices = {
        "C": pd.concat([masks, intensity_z], axis=1),
        "P": pd.concat([masks, intensity_z, raw], axis=1),
    }
    for name, matrix in matrices.items():
        validate_d008_matrix(name, matrix, expected_index=features.index, source_features=features)
    if not matrices["C"].equals(matrices["P"].loc[:, C_COLUMNS]):
        raise AssertionError("D008 C must equal the C block of P exactly.")
    return matrices


def validate_d008_matrix(
    name: str,
    matrix: pd.DataFrame,
    *,
    expected_index: pd.Index | None = None,
    source_features: pd.DataFrame | None = None,
) -> None:
    """Fail closed on schema, raw-return, mask, or row-identity changes."""
    expected = C_COLUMNS if name == "C" else P_COLUMNS if name == "P" else None
    if expected is None or tuple(matrix.columns) != expected:
        raise AssertionError("D008 predictor schema differs from frozen C/P definitions.")
    if expected_index is not None and not matrix.index.equals(expected_index):
        raise AssertionError("D008 matrix changed source row identity or order.")
    mask_frame = matrix.loc[:, MASK_COLUMNS]
    if not all(dtype == np.dtype("int8") for dtype in mask_frame.dtypes):
        raise AssertionError("D008 masks must retain D007 int8 dtype.")
    masks = mask_frame.to_numpy(dtype=float)
    if not np.isfinite(masks).all() or not np.isin(masks, [0.0, 1.0]).all():
        raise AssertionError("D008 masks must be finite binary indicators.")
    if not np.isfinite(matrix.loc[:, ["I_recent_z"]].to_numpy(dtype=float)).all():
        raise AssertionError("D008 I_recent_z must be finite.")
    if name == "P":
        raw = matrix.loc[:, RETURN_COLUMNS].to_numpy(dtype=float)
        observed = ~np.isnan(raw)
        if not np.isfinite(raw[observed]).all():
            raise AssertionError("D008 observed raw returns must be finite.")
        if not np.array_equal(masks, np.isnan(raw).astype(float)):
            raise AssertionError("D008 masks must exactly encode native raw-return NaNs.")
        if source_features is not None and not matrix.loc[:, RETURN_COLUMNS].equals(source_features.loc[:, RETURN_COLUMNS]):
            raise AssertionError("D008 must preserve raw returns exactly.")


def assert_identical_d008_rows(
    representations: Mapping[str, pd.DataFrame], features: pd.DataFrame, target: pd.Series
) -> None:
    """Require C/P to use the exact same subset rows, IDs, targets, and order."""
    if not features.index.equals(target.index):
        raise AssertionError("D008 features and target are not aligned.")
    for name in ("C", "P"):
        if name not in representations:
            raise AssertionError(f"D008 representation {name} is missing.")
        validate_d008_matrix(name, representations[name], expected_index=features.index, source_features=features)
    if not representations["C"].equals(representations["P"].loc[:, C_COLUMNS]):
        raise AssertionError("D008 C/P shared columns differ.")
    if len(representations["C"]) != len(target) or not features["ID"].index.equals(target.index):
        raise AssertionError("D008 representations changed subset membership.")


def validate_expected_fold_count(fold: int, subset: str, n_rows: int) -> None:
    """Assert the frozen existing Discovery membership counts."""
    if fold not in EXPECTED_ROW_COUNTS or subset not in EXPECTED_ROW_COUNTS[fold]:
        raise AssertionError("D008 received an unknown frozen fold/subset.")
    if n_rows != EXPECTED_ROW_COUNTS[fold][subset]:
        raise AssertionError("D008 Discovery row count differs from the frozen manifest.")


def d008_primary_screen(deltas: Sequence[float]) -> dict[str, float | bool | int]:
    """Apply D008's identical all-three-fold positive-log-loss-delta screen."""
    return d007_primary_screen(deltas)


__all__ = [
    "C_COLUMNS", "EXPECTED_ROW_COUNTS", "P_COLUMNS", "RECENT_WINDOW_COLUMNS",
    "RECENT_WINDOW_SIZE", "assert_identical_d008_rows", "build_d008_representations",
    "d008_primary_screen", "fit_intensity_transformer", "validate_d008_matrix",
    "validate_expected_fold_count",
]

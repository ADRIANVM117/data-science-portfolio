"""Frozen utilities for CONF_001 recent-intensity Internal Confirmation.

CONF_001 deliberately reuses EXP_008's target-free feature construction and
fit-only preprocessing. This module adds only the three-block confirmation
schedule and its strict recurrent-positive incremental-AUC decision.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.exp000_validation import FoldSpec
from src.exp008_recent_movement_intensity import (
    RecentIntensityTransformer,
    assert_identical_representation_rows,
    build_exp008_representations,
    build_recent_window_structure,
)


CONFIRMATION_BLOCKS: tuple[FoldSpec, ...] = (
    FoldSpec(1, 0, 352, 353, 402),
    FoldSpec(2, 0, 402, 403, 452),
    FoldSpec(3, 0, 452, 453, 502),
)
CONTROL_COLUMNS: tuple[str, ...] = ("missing_ratio", "q")
CANDIDATE_COLUMNS: tuple[str, ...] = ("missing_ratio", "q", "I_recent_z")


def build_confirmation_representations(
    missing_ratio: pd.DataFrame,
    structure: pd.DataFrame,
    intensity_z: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return exactly C and B from the already-frozen EXP_008 pipeline."""
    exp008 = build_exp008_representations(missing_ratio, structure, intensity_z)
    representations = {"C": exp008["C"], "B": exp008["B"]}
    if tuple(representations["C"].columns) != CONTROL_COLUMNS:
        raise AssertionError("CONF_001 control schema must be [missing_ratio, q].")
    if tuple(representations["B"].columns) != CANDIDATE_COLUMNS:
        raise AssertionError("CONF_001 candidate schema must add only I_recent_z.")
    return representations


def prepare_confirmation_matrices(
    features: pd.DataFrame,
    target: pd.Series,
    transformer: RecentIntensityTransformer,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Construct target-free structure and aligned C/B matrices.

    ``transformer`` must already be fit exclusively on the corresponding Fit
    raw intensity; this function never learns a preprocessing parameter.
    """
    from src.exp002_neutral_directional import build_missing_ratio

    structure = build_recent_window_structure(features)
    intensity_z = transformer.transform(structure["raw_I_recent"])
    representations = build_confirmation_representations(
        build_missing_ratio(features), structure, intensity_z
    )
    # Reuse the EXP_008 identity assertion by supplying its full A/C/B map.
    full_representations = {
        "A": build_missing_ratio(features),
        "C": representations["C"],
        "B": representations["B"],
    }
    assert_identical_representation_rows(full_representations, features, target)
    return structure, representations


def fit_confirmation_transformer(fit_features: pd.DataFrame) -> tuple[pd.DataFrame, RecentIntensityTransformer]:
    """Build Fit structure then learn the sole fit-only CONF_001 transformer."""
    structure = build_recent_window_structure(fit_features)
    return structure, RecentIntensityTransformer().fit(structure["raw_I_recent"])


def confirmation_primary_decision(joint_auc_deltas: Sequence[float]) -> dict[str, float | bool | int]:
    """Apply CONF_001's sole primary recurrence rule to three Joint blocks."""
    if len(joint_auc_deltas) != 3:
        raise AssertionError("CONF_001 requires exactly three Joint Confirmation blocks.")
    deltas = np.asarray(joint_auc_deltas, dtype=float)
    if not np.isfinite(deltas).all():
        raise AssertionError("CONF_001 Joint AUC deltas must be finite.")
    # CONF_001 freezes the literal recurrence condition delta_auc_k > 0.
    # Unlike historical EXP utilities, it intentionally applies no tolerance.
    all_positive = bool((deltas > 0.0).all())
    return {
        "n_joint_blocks": 3,
        "delta_auc_block_1": float(deltas[0]),
        "delta_auc_block_2": float(deltas[1]),
        "delta_auc_block_3": float(deltas[2]),
        "all_three_joint_auc_deltas_positive": all_positive,
        "mean_joint_auc_delta": float(deltas.mean()),
        "std_joint_auc_delta_ddof_1": float(deltas.std(ddof=1)),
        "confirmation_criterion_satisfied": all_positive,
    }


def assert_confirmation_row_identity(
    representations: Mapping[str, pd.DataFrame], features: pd.DataFrame, target: pd.Series
) -> None:
    """Assert that C and B retain the same frozen rows, labels, and order."""
    if set(representations) != {"C", "B"}:
        raise AssertionError("CONF_001 requires exactly C and B representations.")
    for name, expected_columns in (("C", CONTROL_COLUMNS), ("B", CANDIDATE_COLUMNS)):
        matrix = representations[name]
        if tuple(matrix.columns) != expected_columns:
            raise AssertionError(f"CONF_001 {name} has an unexpected schema.")
        if not matrix.index.equals(features.index) or not features.index.equals(target.index):
            raise AssertionError("CONF_001 C/B changed row identity, target alignment, or order.")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise AssertionError("CONF_001 predictor matrices must be finite.")
    if not representations["C"].equals(representations["B"].loc[:, CONTROL_COLUMNS]):
        raise AssertionError("CONF_001 B must differ from C only by I_recent_z.")

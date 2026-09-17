"""D002 local-organization definitions and target-blind RMS matching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


WINDOW_COLUMNS: tuple[str, ...] = tuple(f"r{position}" for position in range(41, 53))
WINDOW_SIZE = 12
DISCOVERY_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("initial_discovery", 0, 202),
    ("block_1", 203, 252),
    ("block_2", 253, 302),
    ("block_3", 303, 352),
)


@dataclass(frozen=True)
class MatchedPairs:
    """One-to-one neutral-anchor matches with directional donors reusable by design."""

    pairs: pd.DataFrame
    neutral_available: int
    directional_available: int


def window_energy_and_rms(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return complete-W energy and RMS while preserving observed zeros."""
    if not set(WINDOW_COLUMNS).issubset(features.columns):
        raise AssertionError("D002 requires exactly r41 through r52.")
    values = features.loc[:, WINDOW_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError("D002 matching requires complete finite W values.")
    energy = np.square(values).sum(axis=1)
    if not np.isfinite(energy).all() or (energy < 0.0).any():
        raise AssertionError("D002 W energy must be finite and non-negative.")
    return energy, np.sqrt(energy / WINDOW_SIZE)


def local_organization_statistics(features: pd.DataFrame) -> pd.DataFrame:
    """Define D002's primary A_adj and secondary H_peak without using labels."""
    values = np.abs(features.loc[:, WINDOW_COLUMNS].to_numpy(dtype=float))
    if not np.isfinite(values).all():
        raise AssertionError("D002 local organization requires complete finite W values.")
    energy = np.square(values).sum(axis=1)
    adjacent = (values[:, :-1] * values[:, 1:]).sum(axis=1)
    peak = np.square(values).max(axis=1)
    a_adj = np.divide(adjacent, energy, out=np.zeros_like(energy), where=energy > 0.0)
    h_peak = np.divide(peak, energy, out=np.zeros_like(energy), where=energy > 0.0)
    result = pd.DataFrame({"E": energy, "I_recent": np.sqrt(energy / WINDOW_SIZE), "A_adj": a_adj, "H_peak": h_peak}, index=features.index)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise AssertionError("D002 local organization statistics must be finite.")
    return result


def match_neutral_to_directional_by_rms(frame: pd.DataFrame) -> MatchedPairs:
    """Match every neutral anchor to its nearest directional donor with replacement.

    Inputs must contain only one chronological block. Directional donors may
    recur, so all neutral anchors are retained. Ties resolve by lower donor RMS
    and then lower donor ID. Anchor ordering is RMS, then ID, for reproducible
    persisted pair order.
    """
    required = {"ID", "Z", "I_recent"}
    if not required.issubset(frame.columns):
        raise AssertionError("D002 RMS matching requires ID, Z, and I_recent.")
    if frame["ID"].duplicated().any() or not frame["I_recent"].ge(0.0).all() or not np.isfinite(frame["I_recent"]).all():
        raise AssertionError("D002 matching inputs must have unique IDs and finite non-negative RMS.")
    neutral = frame.loc[frame["Z"].eq(0), ["ID", "I_recent"]].sort_values(["I_recent", "ID"], kind="stable")
    directional = frame.loc[frame["Z"].eq(1), ["ID", "I_recent"]].sort_values(["I_recent", "ID"], kind="stable")
    if neutral.empty or directional.empty:
        raise AssertionError("D002 matching requires both Neutral and Directional rows in each block.")
    donor_rms = directional["I_recent"].to_numpy(dtype=float)
    donor_ids = directional["ID"].to_numpy()
    records: list[dict[str, float | int]] = []
    for anchor in neutral.itertuples(index=False):
        lower = int(np.searchsorted(donor_rms, anchor.I_recent, side="left")) - 1
        upper = int(np.searchsorted(donor_rms, anchor.I_recent, side="right"))
        boundary_positions = [position for position in (lower, upper) if 0 <= position < len(donor_rms)]
        # Rewind every candidate to the first identical-RMS donor, which has
        # the lowest ID because the donor frame was sorted by RMS then ID.
        candidates = [int(np.searchsorted(donor_rms, donor_rms[position], side="left")) for position in boundary_positions]
        # An exact RMS match lies between lower and upper and must be included.
        exact_start = int(np.searchsorted(donor_rms, anchor.I_recent, side="left"))
        exact_end = int(np.searchsorted(donor_rms, anchor.I_recent, side="right"))
        if exact_start < exact_end:
            candidates.append(exact_start)
        chosen = min(set(candidates), key=lambda position: (abs(donor_rms[position] - anchor.I_recent), donor_rms[position], donor_ids[position]))
        records.append(
            {
                "neutral_ID": int(anchor.ID),
                "directional_ID": int(donor_ids[chosen]),
                "neutral_I_recent": float(anchor.I_recent),
                "directional_I_recent": float(donor_rms[chosen]),
                "abs_rms_difference": float(abs(anchor.I_recent - donor_rms[chosen])),
            }
        )
    return MatchedPairs(pd.DataFrame(records), neutral_available=len(neutral), directional_available=len(directional))


def matching_quality_summary(matches: MatchedPairs, *, rms_scale_iqr: float, block: str, total_rows: int, zero_energy_rows: int) -> dict[str, float | int | str]:
    """Return the frozen target-control diagnostics only; no A_adj/H_peak values enter."""
    differences = matches.pairs["abs_rms_difference"]
    if rms_scale_iqr <= 0.0 or not np.isfinite(rms_scale_iqr):
        raise AssertionError("D002 normalized RMS differences require a positive finite block IQR.")
    unique_directional = int(matches.pairs["directional_ID"].nunique())
    return {
        "block": block,
        "n_complete_w_rows": total_rows,
        "n_zero_energy_rows": zero_energy_rows,
        "n_neutral_available": matches.neutral_available,
        "n_directional_available": matches.directional_available,
        "n_matched_pairs": int(len(matches.pairs)),
        "fraction_neutral_retained": float(len(matches.pairs) / matches.neutral_available),
        "n_unique_directional_matched": unique_directional,
        "fraction_directional_uniquely_retained": float(unique_directional / matches.directional_available),
        "exact_rms_match_fraction": float(differences.eq(0.0).mean()),
        "mean_abs_rms_difference": float(differences.mean()),
        "median_abs_rms_difference": float(differences.median()),
        "p90_abs_rms_difference": float(differences.quantile(0.90)),
        "p95_abs_rms_difference": float(differences.quantile(0.95)),
        "p99_abs_rms_difference": float(differences.quantile(0.99)),
        "max_abs_rms_difference": float(differences.max()),
        "rms_iqr_scale": float(rms_scale_iqr),
        "mean_normalized_abs_rms_difference": float((differences / rms_scale_iqr).mean()),
        "max_normalized_abs_rms_difference": float((differences / rms_scale_iqr).max()),
    }

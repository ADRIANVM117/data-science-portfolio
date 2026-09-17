"""Target-free same-day cross-sectional information audit.

This runner reads only the physically materialized Discovery *input*
partition. It never opens labels, constructs a target, fits a model, or reads
Internal Confirmation, E_holdout, or competition-test data.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.exp001_missingness import RETURN_COLUMNS  # noqa: E402


DISCOVERY_DIR = PROJECT_ROOT / "data" / "discovery"
INPUT_PATH = DISCOVERY_DIR / "discovery_input_training.csv"
MANIFEST_PATH = DISCOVERY_DIR / "discovery_partition_manifest.json"
RESULTS_DIR = PROJECT_ROOT / "discovery" / "results"


def _quantiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in (0.0, 0.05, 0.5, 0.95, 1.0)}


def _loo_median_difference(values: np.ndarray) -> tuple[float, float, int, int]:
    """Return exact inclusive-versus-leave-one-out median difference summaries.

    Values are observed finite returns from one day-position cross-section.
    The returned counts refer to observed rows; a missing row cannot change a
    statistic from which it is absent.
    """
    n = len(values)
    if n < 2:
        return np.nan, np.nan, 0, 0
    ordered = np.sort(values)
    ranks = np.searchsorted(ordered, values, side="left")
    if n % 2:
        k = n // 2
        inclusive = ordered[k]
        loo = np.where(
            ranks < k,
            (ordered[k] + ordered[k + 1]) / 2.0,
            np.where(ranks == k, (ordered[k - 1] + ordered[k + 1]) / 2.0, (ordered[k - 1] + ordered[k]) / 2.0),
        )
    else:
        k = n // 2
        inclusive = (ordered[k - 1] + ordered[k]) / 2.0
        loo = np.where(ranks <= k - 1, ordered[k], ordered[k - 1])
    difference = np.abs(inclusive - loo)
    return float(difference.sum()), float(difference.max()), int((difference > 0.0).sum()), n


def main() -> None:
    if not INPUT_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("The physical Discovery input partition and manifest are required.")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(INPUT_PATH)
    expected = {"ID", "day", "equity", *RETURN_COLUMNS}
    if set(frame.columns) != expected:
        raise AssertionError("Discovery input schema is not exactly ID/day/equity plus r0...r52.")
    if "reod" in frame.columns:
        raise AssertionError("Target data are prohibited in this target-free audit.")
    if len(frame) != int(manifest["row_count"]) or frame["ID"].nunique() != len(frame):
        raise AssertionError("Discovery input does not match its target-blind partition manifest.")
    if not frame["day"].between(0, 352).all() or frame["day"].nunique() != 353:
        raise AssertionError("Discovery audit may use only days 0 through 352.")
    if frame["equity"].nunique() != int(manifest["unique_equity_count"]):
        raise AssertionError("Discovery equity universe disagrees with the physical manifest.")

    returns = frame.loc[:, RETURN_COLUMNS]
    values = returns.to_numpy(dtype=float)
    observed = ~np.isnan(values)
    if not np.isfinite(values[observed]).all():
        raise AssertionError("Observed Discovery returns must be finite; non-finite values are not imputed.")
    day_size = frame.groupby("day", sort=True).size()
    observed_counts = returns.notna().groupby(frame["day"], sort=True).sum()
    coverage = observed_counts.div(day_size, axis=0)
    daily_medians = returns.groupby(frame["day"], sort=True).median()

    # MAD is computed separately for every same-day/position cross-section.
    median_by_row = daily_medians.reindex(frame["day"].to_numpy()).set_axis(frame.index)
    absolute_deviation = (returns - median_by_row).abs()
    daily_mad = absolute_deviation.groupby(frame["day"], sort=True).median()

    loo_records: list[dict[str, float | int | str]] = []
    total_difference = 0.0
    total_observed_for_loo = 0
    total_nonzero_difference = 0
    global_max_difference = 0.0
    for day, indices in frame.groupby("day", sort=True).indices.items():
        positions = np.asarray(indices, dtype=int)
        for column in RETURN_COLUMNS:
            raw = returns.iloc[positions][column].to_numpy(dtype=float)
            current = raw[~np.isnan(raw)]
            sum_difference, max_difference, n_nonzero, n_loo = _loo_median_difference(current)
            loo_records.append({
                "day": int(day), "position": column, "n_observed": int(len(current)),
                "loo_mean_abs_difference": np.nan if n_loo == 0 else sum_difference / n_loo,
                "loo_max_abs_difference": max_difference,
                "loo_nonzero_count": n_nonzero, "loo_defined_count": n_loo,
            })
            if n_loo:
                total_difference += sum_difference
                total_observed_for_loo += n_loo
                total_nonzero_difference += n_nonzero
                global_max_difference = max(global_max_difference, max_difference)

    loo = pd.DataFrame(loo_records)
    day_position = pd.DataFrame(
        {
            "day": np.repeat(coverage.index.to_numpy(dtype=int), len(RETURN_COLUMNS)),
            "position": np.tile(np.asarray(RETURN_COLUMNS), len(coverage)),
            "n_rows_day": np.repeat(day_size.to_numpy(dtype=int), len(RETURN_COLUMNS)),
            "n_observed": observed_counts.to_numpy(dtype=int).ravel(),
            "coverage": coverage.to_numpy(dtype=float).ravel(),
            "median": daily_medians.to_numpy(dtype=float).ravel(),
            "mad": daily_mad.to_numpy(dtype=float).ravel(),
        }
    ).merge(loo, on=["day", "position", "n_observed"], how="left", validate="one_to_one")

    position_summary = day_position.groupby("position", sort=False).agg(
        coverage_mean=("coverage", "mean"), coverage_median=("coverage", "median"),
        coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
        mad_mean=("mad", "mean"), mad_median=("mad", "median"),
        mad_min=("mad", "min"), mad_max=("mad", "max"),
        loo_mean_abs_difference_mean=("loo_mean_abs_difference", "mean"),
        loo_max_abs_difference_max=("loo_max_abs_difference", "max"),
    ).reset_index()

    coverage_values = day_position["coverage"]
    mad_values = day_position["mad"].dropna()
    audit = {
        "scope": {
            "input_only": True, "target_loaded": False, "days": "0-352", "equity_partition": "E_dev",
            "rows": int(len(frame)), "unique_days": int(frame["day"].nunique()),
            "unique_equities": int(frame["equity"].nunique()),
            "return_columns": list(RETURN_COLUMNS),
        },
        "day_row_count": _quantiles(day_size),
        "coverage_all_day_positions": _quantiles(coverage_values),
        "coverage_zero_day_positions": int((coverage_values == 0.0).sum()),
        "coverage_one_day_positions": int((coverage_values == 1.0).sum()),
        "rank_evaluable_day_positions_n_observed_at_least_2": int((day_position["n_observed"] >= 2).sum()),
        "rank_unevaluable_day_positions_n_observed_below_2": int((day_position["n_observed"] < 2).sum()),
        "mad_all_day_positions": _quantiles(mad_values),
        "mad_zero_day_positions": int((day_position["mad"] == 0.0).sum()),
        "leave_one_out": {
            "observed_rows_compared": total_observed_for_loo,
            "mean_abs_difference": total_difference / total_observed_for_loo,
            "max_abs_difference": global_max_difference,
            "nonzero_difference_fraction": total_nonzero_difference / total_observed_for_loo,
            "day_position_cells_with_loo_defined": int((day_position["loo_defined_count"] > 0).sum()),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    day_position.to_csv(RESULTS_DIR / "CROSS_SECTIONAL_TARGET_FREE_DAY_POSITION.csv", index=False)
    position_summary.to_csv(RESULTS_DIR / "CROSS_SECTIONAL_TARGET_FREE_POSITION_SUMMARY.csv", index=False)
    (RESULTS_DIR / "CROSS_SECTIONAL_TARGET_FREE_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

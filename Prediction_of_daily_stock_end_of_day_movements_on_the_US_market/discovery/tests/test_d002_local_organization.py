"""Synthetic integrity tests for D002 statistics and RMS matching only."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.d002_local_organization import (  # noqa: E402
    DISCOVERY_BLOCKS,
    WINDOW_COLUMNS,
    local_organization_statistics,
    match_neutral_to_directional_by_rms,
    matching_quality_summary,
)
from discovery.run_d002_matched_organization import summarize_delta  # noqa: E402


def _paths(rows: list[np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def test_matched_rms_synthetic_examples_and_zero_convention() -> None:
    distributed = np.ones(12)
    shock = np.array([np.sqrt(12), *([0.0] * 11)])
    block = np.array([0.0, 0.0, np.sqrt(3), np.sqrt(3), np.sqrt(3), np.sqrt(3), *([0.0] * 6)])
    alternating = np.array([np.sqrt(2) if index % 2 == 0 else 0.0 for index in range(12)])
    zeros = np.zeros(12)
    stats = local_organization_statistics(_paths([distributed, shock, block, alternating, zeros]))
    assert np.allclose(stats["I_recent"].iloc[:4], 1.0)
    assert np.isclose(stats["A_adj"].iloc[0], 11 / 12)
    assert np.isclose(stats["A_adj"].iloc[1], 0.0)
    assert np.isclose(stats["A_adj"].iloc[2], 3 / 4)
    assert np.isclose(stats["A_adj"].iloc[3], 0.0)
    assert np.isclose(stats["H_peak"].iloc[0], 1 / 12)
    assert np.isclose(stats["H_peak"].iloc[1], 1.0)
    assert np.isclose(stats["H_peak"].iloc[2], 1 / 4)
    assert np.isclose(stats["H_peak"].iloc[3], 1 / 6)
    assert stats.loc[4, ["E", "I_recent", "A_adj", "H_peak"]].eq(0.0).all()


def test_matching_is_deterministic_nearest_and_allows_directional_replacement() -> None:
    frame = pd.DataFrame({"ID": [30, 10, 20, 40, 50], "Z": [0, 0, 1, 1, 1], "I_recent": [1.0, 2.0, 1.5, 1.5, 4.0]})
    first = match_neutral_to_directional_by_rms(frame)
    second = match_neutral_to_directional_by_rms(frame.sample(frac=1.0, random_state=7))
    assert first.pairs.equals(second.pairs)
    assert first.pairs["neutral_ID"].tolist() == [30, 10]
    # Both 1.5 donors tie for neutral 1.0 and 2.0; lower donor ID wins and may repeat.
    assert first.pairs["directional_ID"].tolist() == [20, 20]
    quality = matching_quality_summary(first, rms_scale_iqr=1.0, block="synthetic", total_rows=5, zero_energy_rows=0)
    assert quality["n_matched_pairs"] == 2 and quality["n_unique_directional_matched"] == 1
    assert quality["fraction_neutral_retained"] == 1.0


def test_block_contract_and_no_scientific_target_use_in_runner() -> None:
    assert DISCOVERY_BLOCKS == (("initial_discovery", 0, 202), ("block_1", 203, 252), ("block_2", 253, 302), ("block_3", 303, 352))
    source = (PROJECT_ROOT / "discovery" / "run_d002_rms_matching_quality.py").read_text(encoding="utf-8")
    assert "local_organization_statistics" not in source
    assert "input_test" not in source and "output_test" not in source
    assert "A_adj" not in source and "H_peak" not in source


def test_paired_delta_summary_has_the_frozen_descriptive_fields() -> None:
    summary = summarize_delta(pd.Series([-0.5, 0.0, 0.5, 1.0]), block="block_1", statistic="Delta_A_adj")
    assert summary["n_matched_pairs"] == 4
    assert np.isclose(summary["mean"], 0.25) and np.isclose(summary["median"], 0.25)
    assert np.isclose(summary["fraction_positive"], 0.5)
    assert np.isclose(summary["fraction_zero"], 0.25)
    assert np.isclose(summary["fraction_negative"], 0.25)
    source = (PROJECT_ROOT / "discovery" / "run_d002_matched_organization.py").read_text(encoding="utf-8")
    assert "match_neutral_to_directional_by_rms" not in source
    assert "input_test" not in source and "output_test" not in source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_matched_rms_synthetic_examples_and_zero_convention,
        test_matching_is_deterministic_nearest_and_allows_directional_replacement,
        test_block_contract_and_no_scientific_target_use_in_runner,
        test_paired_delta_summary_has_the_frozen_descriptive_fields,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

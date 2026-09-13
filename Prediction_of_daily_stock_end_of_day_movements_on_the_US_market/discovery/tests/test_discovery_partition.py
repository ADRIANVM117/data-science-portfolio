"""Synthetic structural tests for controlled Discovery partitioning."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_discovery_partition import (  # noqa: E402
    _validate_partition_frames,
    authorized_membership,
)


def test_target_blind_membership_uses_only_id_day_equity() -> None:
    metadata = pd.DataFrame({"ID": [1, 2, 3], "day": [0, 352, 353], "equity": [10, 11, 10]})
    assert authorized_membership(metadata, {10, 11}).tolist() == [True, True, False]
    assert authorized_membership(metadata, {10}).tolist() == [True, False, False]


def test_input_order_label_alignment_and_structural_validation() -> None:
    input_rows = pd.DataFrame(
        {
            "ID": [20, 10, 30, 40],
            "day": [0, 352, 353, 1],
            "equity": [10, 11, 10, 1700],
            "r0": [9.0, 8.0, 7.0, 6.0],
        }
    )
    allowed = authorized_membership(input_rows[["ID", "day", "equity"]], {10, 11})
    selected_input = input_rows.loc[allowed].reset_index(drop=True)
    # Deliberately out of input order: labels must be realigned by selected IDs.
    raw_labels = pd.DataFrame({"ID": [10, 30, 20, 40], "reod": [0, 1, -1, 1]})
    selected_labels = raw_labels.set_index("ID").reindex(selected_input["ID"]).reset_index()
    assert selected_input["ID"].tolist() == [20, 10]
    assert selected_labels["ID"].tolist() == [20, 10]
    assert selected_labels["reod"].tolist() == [-1, 0]
    _validate_partition_frames(selected_input, selected_labels, {10, 11}, expected_rows=2)
    # Repeating the target-blind operation preserves the same IDs and order.
    repeated = input_rows.loc[authorized_membership(input_rows[["ID", "day", "equity"]], {10, 11})]
    assert repeated["ID"].tolist() == selected_input["ID"].tolist()


def test_source_contains_no_competition_or_feature_target_selection() -> None:
    source = (PROJECT_ROOT / "scripts" / "materialize_discovery_partition.py").read_text(encoding="utf-8")
    membership_body = source.split("def authorized_membership", 1)[1].split("def _atomic_csv", 1)[0]
    assert "reod" not in membership_body and "r0" not in membership_body
    assert "input_test" not in source and "output_test" not in source


if __name__ == "__main__":
    import unittest

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_target_blind_membership_uses_only_id_day_equity,
        test_input_order_label_alignment_and_structural_validation,
        test_source_contains_no_competition_or_feature_target_selection,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

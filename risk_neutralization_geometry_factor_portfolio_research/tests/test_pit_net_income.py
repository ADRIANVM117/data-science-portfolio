from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pit_net_income import build_quarters_as_of, ttm_net_income_as_of  # noqa: E402


def fact(start: str, end: str, value: float, form: str, accn: str, accepted: str) -> dict[str, object]:
    return {
        "issuer": "TEST",
        "cik": "0000000001",
        "tag": "NetIncomeLoss",
        "unit": "USD",
        "start": start,
        "end": end,
        "value": value,
        "form": form,
        "accn": accn,
        "acceptanceDateTime": accepted,
    }


def fiscal_year(prefix: str, accepted: str, q1: float = 10, q2: float = 20, q3: float = 30, q4: float = 40) -> list[dict[str, object]]:
    year = int(prefix[:4])
    return [
        fact(f"{year}-01-01", f"{year}-03-31", q1, "10-Q", f"{prefix}-q1", accepted),
        fact(f"{year}-01-01", f"{year}-06-30", q1 + q2, "10-Q", f"{prefix}-h1", accepted),
        fact(f"{year}-04-01", f"{year}-06-30", q2, "10-Q", f"{prefix}-q2", accepted),
        fact(f"{year}-01-01", f"{year}-09-30", q1 + q2 + q3, "10-Q", f"{prefix}-9m", accepted),
        fact(f"{year}-07-01", f"{year}-09-30", q3, "10-Q", f"{prefix}-q3", accepted),
        fact(f"{year}-01-01", f"{year}-12-31", q1 + q2 + q3 + q4, "10-K", f"{prefix}-fy", accepted),
    ]


def result_for(rows: list[dict[str, object]], decision: str):
    return ttm_net_income_as_of(pd.DataFrame(rows), "TEST", decision)


def test_four_valid_quarters_produce_ttm() -> None:
    result = result_for(fiscal_year("2024", "2025-01-02T12:00:00Z"), "2025-01-03T14:30:00Z")
    assert result.status == "VALID"
    assert result.value == 100
    assert len(result.quarters) == 4
    assert set(result.quarters["method"]) == {"direct", "reconstructed"}


def test_fewer_than_four_valid_quarters_produce_na() -> None:
    rows = fiscal_year("2024", "2025-01-02T12:00:00Z")[:-1]
    result = result_for(rows, "2025-01-03T14:30:00Z")
    assert result.value is None
    assert result.status == "NA_FEWER_THAN_FOUR_VALID_QUARTERS"


def test_duplicate_non_amendment_quarter_is_not_counted_twice() -> None:
    rows = fiscal_year("2024", "2025-01-02T12:00:00Z")
    rows.append(fact("2024-04-01", "2024-06-30", 20, "10-Q", "comparative-q2", "2025-02-02T12:00:00Z"))
    result = result_for(rows, "2025-02-03T14:30:00Z")
    assert result.status == "VALID"
    assert result.value == 100
    assert result.quarters["economic_quarter_end"].nunique() == 4


def test_future_filing_cannot_enter_earlier_decision_time() -> None:
    rows = fiscal_year("2024", "2025-01-02T15:00:00Z")
    result = result_for(rows, "2025-01-02T14:30:00Z")
    assert result.value is None
    assert result.status == "NA_FEWER_THAN_FOUR_VALID_QUARTERS"


def test_amendments_enter_only_prospectively() -> None:
    rows = fiscal_year("2024", "2025-01-02T12:00:00Z")
    amended = fiscal_year("2024", "2025-02-03T15:00:00Z", 12, 21, 31, 40)
    for row in amended:
        row["form"] = f"{row['form']}/A"
        row["accn"] = f"amended-{row['accn']}"
    rows.extend(amended)
    before = result_for(rows, "2025-02-03T14:30:00Z")
    after = result_for(rows, "2025-02-04T14:30:00Z")
    assert before.value == 100
    assert after.status == "VALID"
    assert after.value == 104


def test_ambiguous_direct_quarter_produces_na() -> None:
    rows = fiscal_year("2024", "2025-01-02T12:00:00Z")
    rows.append(fact("2024-01-01", "2024-03-31", 11, "10-Q", "conflicting-q1", "2025-01-02T12:00:00Z"))
    quarters = build_quarters_as_of(pd.DataFrame(rows), "TEST", "2025-01-03T14:30:00Z")
    q1 = quarters.loc[quarters["economic_quarter_end"].eq(pd.Timestamp("2024-03-31"))].iloc[0]
    assert q1["status"] == "NA_AMBIGUOUS_VINTAGE"
    assert result_for(rows, "2025-01-03T14:30:00Z").value is None


def test_ttm_changes_at_new_information_event_and_stays_constant_between_events() -> None:
    rows = fiscal_year("2024", "2025-01-02T12:00:00Z")
    current_year = fiscal_year("2025", "2025-10-31T10:00:00Z", 11, 21, 31, 47)
    for row in current_year[:-1]:
        row["acceptanceDateTime"] = "2025-09-30T12:00:00Z"
    rows.extend(current_year)
    before = result_for(rows, "2025-10-30T13:30:00Z")
    first_eligible = result_for(rows, "2025-10-31T13:30:00Z")
    later = result_for(rows, "2025-11-03T14:30:00Z")
    assert before.value == 103
    assert first_eligible.value == 110
    assert later.value == first_eligible.value

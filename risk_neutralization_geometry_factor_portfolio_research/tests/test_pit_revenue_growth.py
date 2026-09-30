from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pit_revenue_growth import revenue_growth_as_of, revenue_ttm_as_of  # noqa: E402


TAG = "Revenue"


def fact(start: str, end: str, value: float, form: str, accn: str, accepted: str, tag: str = TAG) -> dict[str, object]:
    return {"issuer": "TEST", "cik": "0000000001", "tag": tag, "unit": "USD", "start": start, "end": end,
            "value": value, "form": form, "accn": accn, "acceptanceDateTime": accepted}


def year(year: int, annual_value: float, accepted: str, tag: str = TAG) -> list[dict[str, object]]:
    q1, q2, q3 = annual_value * .20, annual_value * .25, annual_value * .25
    q4 = annual_value - q1 - q2 - q3
    return [
        fact(f"{year}-01-01", f"{year}-03-31", q1, "10-Q", f"{year}-q1", accepted, tag),
        fact(f"{year}-01-01", f"{year}-06-30", q1 + q2, "10-Q", f"{year}-h1", accepted, tag),
        fact(f"{year}-01-01", f"{year}-09-30", q1 + q2 + q3, "10-Q", f"{year}-9m", accepted, tag),
        fact(f"{year}-01-01", f"{year}-12-31", annual_value, "10-K", f"{year}-fy", accepted, tag),
    ]


def facts(values: list[float], tag: str = TAG) -> pd.DataFrame:
    rows = []
    for offset, value in enumerate(values):
        calendar_year = 2020 + offset
        rows.extend(year(calendar_year, value, f"{calendar_year + 1}-01-02T12:00:00Z", tag))
    return pd.DataFrame(rows)


def test_valid_revenue_ttm_and_three_year_cagr_with_full_persistence() -> None:
    frame = facts([100, 110, 121, 133.1])
    ttm = revenue_ttm_as_of(frame, "TEST", "2024-01-03T14:30:00Z", TAG)
    result = revenue_growth_as_of(frame, "TEST", "2024-01-03T14:30:00Z", TAG)
    assert ttm.status == "VALID" and ttm.value == 133.1
    assert result.status == "VALID"
    assert round(result.growth_magnitude, 10) == 0.1
    assert result.growth_persistence == 1.0


def test_partial_persistence_and_negative_magnitude_are_distinct() -> None:
    result = revenue_growth_as_of(facts([100, 110, 99, 90]), "TEST", "2024-01-03T14:30:00Z", TAG)
    assert result.status == "VALID"
    assert result.growth_persistence == 1 / 3
    assert result.growth_magnitude < 0


def test_insufficient_history_never_shortens_horizon() -> None:
    result = revenue_growth_as_of(facts([100, 110, 121]), "TEST", "2023-01-03T14:30:00Z", TAG)
    assert result.status == "NA_FEWER_THAN_FOUR_ANNUAL_TTMS"
    assert result.growth_magnitude is None and result.growth_persistence is None


def test_incompatible_tag_history_returns_na_instead_of_splicing() -> None:
    rows = facts([100, 110, 121, 133.1]).to_dict("records")
    for row in rows:
        if row["start"].startswith("2021"):
            row["tag"] = "AlternativeRevenue"
    result = revenue_growth_as_of(pd.DataFrame(rows), "TEST", "2024-01-03T14:30:00Z", TAG)
    assert result.status == "NA_FEWER_THAN_FOUR_ANNUAL_TTMS"


def test_future_filing_cannot_rewrite_prior_revenue_ttm() -> None:
    frame = facts([100, 110, 121, 133.1])
    frame.loc[frame["end"].eq("2023-12-31"), "acceptanceDateTime"] = "2024-01-03T15:00:00Z"
    before = revenue_ttm_as_of(frame, "TEST", "2024-01-03T14:30:00Z", TAG)
    after = revenue_ttm_as_of(frame, "TEST", "2024-01-04T14:30:00Z", TAG)
    assert before.value is not None and before.value != 133.1
    assert after.value == 133.1


def test_amendment_enters_prospectively_for_revenue_ttm() -> None:
    frame = facts([100, 110, 121, 133.1])
    amended = pd.DataFrame(year(2023, 140, "2024-02-03T15:00:00Z"))
    amended["form"] = amended["form"] + "/A"
    amended["accn"] = "amended-" + amended["accn"]
    combined = pd.concat([frame, amended])
    before = revenue_ttm_as_of(combined, "TEST", "2024-02-03T14:30:00Z", TAG)
    # The amendment was accepted on Saturday; frozen XNYS eligibility is Monday.
    after = revenue_ttm_as_of(combined, "TEST", "2024-02-05T14:30:00Z", TAG)
    assert before.value == 133.1
    assert after.value == 140


def test_manual_formula_matches_reusable_result() -> None:
    result = revenue_growth_as_of(facts([100, 90, 99, 108.9]), "TEST", "2024-01-03T14:30:00Z", TAG)
    manual_cagr = (108.9 / 100) ** (1 / 3) - 1
    manual_persistence = 2 / 3
    assert result.growth_magnitude == manual_cagr
    assert result.growth_persistence == manual_persistence

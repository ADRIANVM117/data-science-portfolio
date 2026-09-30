from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pit_profitability import average_assets_as_of, profitability_as_of  # noqa: E402


def income(start: str, end: str, value: float, form: str, accn: str, accepted: str = "2025-01-02T12:00:00Z") -> dict[str, object]:
    return {"issuer": "TEST", "cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD", "start": start, "end": end,
            "value": value, "form": form, "accn": accn, "acceptanceDateTime": accepted}


def valid_income() -> pd.DataFrame:
    return pd.DataFrame([
        income("2024-01-01", "2024-03-31", 10, "10-Q", "q1"),
        income("2024-01-01", "2024-06-30", 30, "10-Q", "h1"),
        income("2024-04-01", "2024-06-30", 20, "10-Q", "q2"),
        income("2024-01-01", "2024-09-30", 60, "10-Q", "y9"),
        income("2024-07-01", "2024-09-30", 30, "10-Q", "q3"),
        income("2024-01-01", "2024-12-31", 100, "10-K", "fy"),
    ])


def asset(date: str, value: float, accn: str, form: str = "10-K", accepted: str = "2025-01-02T12:00:00Z") -> dict[str, object]:
    return {"issuer": "TEST", "cik": "0000000001", "tag": "Assets", "unit": "USD", "reference_date": date,
            "value": value, "form": form, "accn": accn, "acceptanceDateTime": accepted}


def valid_assets() -> pd.DataFrame:
    return pd.DataFrame([asset("2023-12-31", 400, "begin"), asset("2024-12-31", 600, "end")])


DECISION = "2025-01-03T14:30:00Z"


def test_valid_average_assets_has_exact_ttm_boundary_alignment() -> None:
    result = average_assets_as_of(valid_income(), valid_assets(), "TEST", DECISION)
    assert result.status == "VALID"
    assert result.ttm_economic_start == pd.Timestamp("2024-01-01")
    assert result.ttm_economic_end == pd.Timestamp("2024-12-31")
    assert result.beginning_assets.reference_date == pd.Timestamp("2023-12-31")
    assert result.ending_assets.reference_date == pd.Timestamp("2024-12-31")
    assert result.average_assets == 500


def test_beginning_or_ending_boundary_unavailable_returns_explained_na() -> None:
    no_beginning = average_assets_as_of(valid_income(), valid_assets().iloc[1:], "TEST", DECISION)
    no_ending = average_assets_as_of(valid_income(), valid_assets().iloc[:1], "TEST", DECISION)
    assert no_beginning.average_assets is None and no_beginning.status == "INCOMPATIBLE_ASSET_BOUNDARY"
    assert no_ending.average_assets is None and no_ending.status == "INCOMPATIBLE_ASSET_BOUNDARY"
    assert "beginning" in no_beginning.reason and "ending" in no_ending.reason


def test_unresolved_duplicate_in_selected_vintage_returns_na() -> None:
    rows = valid_assets().to_dict("records")
    rows.append(asset("2024-12-31", 601, "conflict"))
    result = average_assets_as_of(valid_income(), pd.DataFrame(rows), "TEST", DECISION)
    assert result.average_assets is None
    assert result.status == "AMBIGUOUS_ASSET_VINTAGE"


def test_future_comparative_filing_cannot_rewrite_earlier_information() -> None:
    rows = valid_assets().to_dict("records")
    rows.append(asset("2024-12-31", 900, "future-comparative", accepted="2025-01-03T15:00:00Z"))
    before = average_assets_as_of(valid_income(), pd.DataFrame(rows), "TEST", DECISION)
    after = average_assets_as_of(valid_income(), pd.DataFrame(rows), "TEST", "2025-01-06T14:30:00Z")
    assert before.status == after.status == "VALID"
    assert before.ending_assets.value == after.ending_assets.value == 600
    assert before.ending_assets.source_accession == after.ending_assets.source_accession == "end"


def test_amendment_enters_only_prospectively() -> None:
    rows = valid_assets().to_dict("records")
    rows.append(asset("2024-12-31", 700, "end-amendment", "10-K/A", "2025-01-03T15:00:00Z"))
    before = average_assets_as_of(valid_income(), pd.DataFrame(rows), "TEST", DECISION)
    after = average_assets_as_of(valid_income(), pd.DataFrame(rows), "TEST", "2025-01-06T14:30:00Z")
    assert before.ending_assets.value == 600
    assert after.status == "VALID" and after.ending_assets.value == 700


def test_incompatible_boundary_never_substitutes_nearest_assets_date() -> None:
    assets = pd.DataFrame([asset("2023-12-30", 400, "near-begin"), asset("2024-12-31", 600, "end")])
    result = average_assets_as_of(valid_income(), assets, "TEST", DECISION)
    assert result.status == "INCOMPATIBLE_ASSET_BOUNDARY"
    assert result.average_assets is None


def test_valid_numerator_and_denominator_produce_profitability() -> None:
    result = profitability_as_of(valid_income(), valid_assets(), "TEST", DECISION)
    assert result.status == "VALID"
    assert result.net_income.value == 100
    assert result.average_assets.average_assets == 500
    assert result.profitability == 0.2


def test_invalid_numerator_or_denominator_produces_profitability_na() -> None:
    invalid_numerator = profitability_as_of(valid_income().iloc[:-1], valid_assets(), "TEST", DECISION)
    invalid_denominator = profitability_as_of(valid_income(), valid_assets().iloc[:1], "TEST", DECISION)
    assert invalid_numerator.profitability is None and invalid_numerator.status == "INVALID_TTM"
    assert invalid_denominator.profitability is None and invalid_denominator.status == "INCOMPATIBLE_ASSET_BOUNDARY"


def test_future_assets_do_not_enter_earlier_decision() -> None:
    assets = valid_assets().copy()
    assets.loc[assets["reference_date"].eq("2024-12-31"), "acceptanceDateTime"] = "2025-01-03T15:00:00Z"
    result = average_assets_as_of(valid_income(), assets, "TEST", DECISION)
    assert result.average_assets is None
    assert result.status == "INCOMPATIBLE_ASSET_BOUNDARY"

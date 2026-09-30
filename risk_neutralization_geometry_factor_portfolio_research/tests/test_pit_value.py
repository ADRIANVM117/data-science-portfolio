from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pit_value import earnings_yield_as_of, market_cap_as_of  # noqa: E402


def prices(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["price_date", "close", "split_coefficient"]).set_index("price_date")


def shares(rows: list[tuple[str, float, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["issuer", "reported_shares", "accn", "shares_observation_date", "acceptanceDateTime"])


def income_fact(start: str, end: str, value: float, form: str, accn: str) -> dict[str, object]:
    return {
        "issuer": "TEST", "cik": "0000000001", "tag": "NetIncomeLoss", "unit": "USD",
        "start": start, "end": end, "value": value, "form": form, "accn": accn,
        "acceptanceDateTime": "2025-01-02T12:00:00Z",
    }


def valid_income_facts() -> pd.DataFrame:
    return pd.DataFrame([
        income_fact("2024-01-01", "2024-03-31", 10, "10-Q", "q1"),
        income_fact("2024-01-01", "2024-06-30", 30, "10-Q", "h1"),
        income_fact("2024-04-01", "2024-06-30", 20, "10-Q", "q2"),
        income_fact("2024-01-01", "2024-09-30", 60, "10-Q", "y9"),
        income_fact("2024-07-01", "2024-09-30", 30, "10-Q", "q3"),
        income_fact("2024-01-01", "2024-12-31", 100, "10-K", "fy"),
    ])


def test_valid_market_cap_uses_raw_close_and_previous_session() -> None:
    result = market_cap_as_of(
        prices([("2025-01-02", 10.0, 1.0), ("2025-01-03", 999.0, 1.0)]),
        shares([("TEST", 100.0, "shares-1", "2024-12-31", "2025-01-01T12:00:00Z")]),
        "TEST", "2025-01-03T14:30:00Z",
    )
    assert result.status == "VALID"
    assert result.raw_price_date == pd.Timestamp("2025-01-02")
    assert result.raw_close == 10.0
    assert result.market_cap == 1000.0


def test_adjusted_close_is_not_required_or_used() -> None:
    frame = prices([("2025-01-02", 10.0, 1.0)])
    frame["adjusted_close"] = 0.01
    result = market_cap_as_of(
        frame,
        shares([("TEST", 100.0, "shares-1", "2024-12-31", "2025-01-01T12:00:00Z")]),
        "TEST", "2025-01-03T14:30:00Z",
    )
    assert result.status == "VALID"
    assert result.market_cap == 1000.0


def test_split_basis_mismatch_returns_na() -> None:
    result = market_cap_as_of(
        prices([("2025-01-02", 10.0, 1.0), ("2025-01-03", 2.5, 4.0)]),
        shares([("TEST", 100.0, "pre-split", "2024-12-31", "2025-01-01T12:00:00Z")]),
        "TEST", "2025-01-06T14:30:00Z",
    )
    assert result.status == "SPLIT_BASIS_MISMATCH"
    assert result.market_cap is None
    assert result.split_event_date == pd.Timestamp("2025-01-03")


def test_validity_resumes_after_post_split_shares_are_eligible() -> None:
    result = market_cap_as_of(
        prices([("2025-01-02", 10.0, 1.0), ("2025-01-03", 2.5, 4.0), ("2025-01-06", 2.6, 1.0)]),
        shares([
            ("TEST", 100.0, "pre-split", "2024-12-31", "2025-01-01T12:00:00Z"),
            ("TEST", 400.0, "post-split", "2025-01-04", "2025-01-05T12:00:00Z"),
        ]),
        "TEST", "2025-01-07T14:30:00Z",
    )
    assert result.status == "VALID"
    assert result.shares_source_accession == "post-split"
    assert result.market_cap == 1040.0


def test_missing_or_ambiguous_shares_return_explained_status() -> None:
    price = prices([("2025-01-02", 10.0, 1.0)])
    missing = market_cap_as_of(price, shares([],), "TEST", "2025-01-03T14:30:00Z")
    ambiguous = market_cap_as_of(
        price,
        shares([
            ("TEST", 100.0, "one", "2024-12-31", "2025-01-01T12:00:00Z"),
            ("TEST", 101.0, "two", "2024-12-31", "2025-01-01T12:00:00Z"),
        ]),
        "TEST", "2025-01-03T14:30:00Z",
    )
    assert missing.status == "MISSING_SHARES"
    assert ambiguous.status == "AMBIGUOUS_SHARES"


def test_earnings_yield_is_na_for_invalid_ttm() -> None:
    result = earnings_yield_as_of(
        valid_income_facts().iloc[:-1], prices([("2025-01-02", 10.0, 1.0)]),
        shares([("TEST", 100.0, "shares-1", "2024-12-31", "2025-01-01T12:00:00Z")]),
        "TEST", "2025-01-03T14:30:00Z",
    )
    assert result.status == "INVALID_TTM"
    assert result.earnings_yield is None


def test_valid_numerator_and_denominator_produce_earnings_yield() -> None:
    result = earnings_yield_as_of(
        valid_income_facts(), prices([("2025-01-02", 10.0, 1.0)]),
        shares([("TEST", 100.0, "shares-1", "2024-12-31", "2025-01-01T12:00:00Z")]),
        "TEST", "2025-01-03T14:30:00Z",
    )
    assert result.status == "VALID"
    assert result.net_income.value == 100.0
    assert result.market_cap.market_cap == 1000.0
    assert result.earnings_yield == 0.1


def test_future_shares_do_not_enter_earlier_decision() -> None:
    result = market_cap_as_of(
        prices([("2025-01-02", 10.0, 1.0)]),
        shares([("TEST", 100.0, "future", "2024-12-31", "2025-01-03T15:00:00Z")]),
        "TEST", "2025-01-03T14:30:00Z",
    )
    assert result.status == "MISSING_SHARES"

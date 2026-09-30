from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pit_momentum import XNYS, momentum_as_of  # noqa: E402


def opening(session: str) -> pd.Timestamp:
    return XNYS.session_open(pd.Timestamp(session))


def prices(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["price_date", "adjusted_close"])


def complete_prices(current: float = 120.0, one_month: float = 110.0, twelve_month: float = 100.0) -> pd.DataFrame:
    # Decision at 2025-01-03 open: cutoff 2025-01-02, 1M anchor 2024-12-02,
    # and 12M anchor 2024-01-02 under the frozen calendar/XNYS convention.
    return prices([("2024-01-02", twelve_month), ("2024-12-02", one_month), ("2025-01-02", current)])


def test_valid_12m_and_1m_manual_calculation() -> None:
    result = momentum_as_of(complete_prices(), "TEST", opening("2025-01-03"))
    assert result.status == "VALID"
    assert result.information_cutoff == pd.Timestamp("2025-01-02")
    assert result.twelve_month_reference_date == pd.Timestamp("2024-01-02")
    assert result.one_month_reference_date == pd.Timestamp("2024-12-02")
    assert result.momentum_12m == pytest.approx(120 / 100 - 1)
    assert result.momentum_1m == pytest.approx(120 / 110 - 1)


def test_positive_recent_continuation_and_negative_recent_deterioration() -> None:
    positive = momentum_as_of(complete_prices(current=120, one_month=110), "TEST", opening("2025-01-03"))
    negative = momentum_as_of(complete_prices(current=100, one_month=110), "TEST", opening("2025-01-03"))
    assert positive.momentum_1m > 0
    assert negative.momentum_1m < 0


def test_calendar_session_lookback_and_decision_open_cutoff() -> None:
    # Jan 1 is an XNYS holiday. The Jan 2 open must cut off at Dec 31.
    frame = prices([("2023-12-29", 100), ("2024-11-29", 110), ("2024-12-31", 120), ("2025-01-02", 999)])
    result = momentum_as_of(frame, "TEST", opening("2025-01-02"))
    assert result.status == "VALID"
    assert result.information_cutoff == pd.Timestamp("2024-12-31")
    assert result.current_adjusted_close == 120
    assert result.current_adjusted_close != 999


def test_insufficient_12m_history_returns_na() -> None:
    result = momentum_as_of(prices([("2024-12-02", 110), ("2025-01-02", 120)]), "TEST", opening("2025-01-03"))
    assert result.status == "NA_INSUFFICIENT_12M_HISTORY"
    assert result.momentum_12m is None and result.momentum_1m is None


def test_insufficient_1m_history_returns_na() -> None:
    result = momentum_as_of(prices([("2024-01-02", 100), ("2025-01-02", 120)]), "TEST", opening("2025-01-03"))
    assert result.status == "NA_INSUFFICIENT_1M_HISTORY"
    assert result.momentum_12m is None and result.momentum_1m is None


def test_no_same_day_or_future_information() -> None:
    frame = complete_prices()
    frame.loc[len(frame)] = ["2025-01-03", 9_999]
    frame.loc[len(frame)] = ["2025-02-03", 8_888]
    result = momentum_as_of(frame, "TEST", opening("2025-01-03"))
    assert result.current_price_date == pd.Timestamp("2025-01-02")
    assert result.current_adjusted_close == 120


def test_adjusted_close_split_sanity_has_no_mechanical_minus_75_percent_return() -> None:
    frame = prices([
        ("2019-08-30", 100.0), ("2020-07-31", 110.0),
        ("2020-08-28", 120.97), ("2020-08-31", 125.07),
    ])
    result = momentum_as_of(frame, "AAPL", opening("2020-09-01"))
    assert result.status == "VALID"
    assert result.momentum_1m == pytest.approx(125.07 / 110 - 1)
    assert result.momentum_12m > 0
    assert result.momentum_12m > -0.10


def test_requires_an_exact_timezone_aware_xnys_open() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        momentum_as_of(complete_prices(), "TEST", "2025-01-03")
    with pytest.raises(ValueError, match="regular-session opening"):
        momentum_as_of(complete_prices(), "TEST", "2025-01-03T15:00:00Z")

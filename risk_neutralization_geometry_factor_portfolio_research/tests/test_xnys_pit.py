from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xnys_pit import eligibility_from_acceptance  # noqa: E402


def session_date(acceptance_datetime: str) -> str:
    return str(eligibility_from_acceptance(acceptance_datetime).eligibility_session.date())


def test_pre_open_acceptance_uses_same_normal_weekday_session() -> None:
    assert session_date("2025-06-03T13:00:00Z") == "2025-06-03"


def test_in_session_acceptance_waits_until_next_session() -> None:
    assert session_date("2025-06-03T15:00:00Z") == "2025-06-04"


def test_acceptance_at_the_session_open_is_strictly_after_not_same_session() -> None:
    assert session_date("2025-06-03T13:30:00Z") == "2025-06-04"


def test_after_close_acceptance_waits_until_next_session() -> None:
    assert session_date("2025-06-03T21:30:00Z") == "2025-06-04"


def test_friday_after_close_and_weekend_wait_until_monday() -> None:
    assert session_date("2025-06-06T21:30:00Z") == "2025-06-09"
    assert session_date("2025-06-07T15:00:00Z") == "2025-06-09"


def test_holiday_boundary_skips_independence_day() -> None:
    assert session_date("2025-07-03T18:00:00Z") == "2025-07-07"


def test_scheduled_early_close_is_an_eligible_session() -> None:
    result = eligibility_from_acceptance("2025-07-03T12:00:00Z")
    assert str(result.eligibility_session.date()) == "2025-07-03"
    assert result.portfolio_decision_time == pd.Timestamp("2025-07-03T13:30:00Z")


def test_timezone_aware_input_is_required_and_normalized_to_utc() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        eligibility_from_acceptance("2025-06-03 09:00:00")

    result = eligibility_from_acceptance("2025-06-03T09:00:00-04:00")
    assert result.acceptance_datetime == pd.Timestamp("2025-06-03T13:00:00Z")

"""Deterministic SEC acceptance-to-XNYS eligibility mapping.

This module implements the frozen research convention only. It does not
assert when a filing first became visible on SEC.gov.
"""

from __future__ import annotations

from dataclasses import dataclass

import exchange_calendars as xcals
import pandas as pd


XNYS = xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class XNYSEligibility:
    """The distinct times used by the conservative daily PIT convention."""

    acceptance_datetime: pd.Timestamp
    eligibility_session: pd.Timestamp
    portfolio_decision_time: pd.Timestamp


def _as_utc_timestamp(acceptance_datetime: object) -> pd.Timestamp:
    """Parse a timezone-aware SEC acceptance time and normalize it to UTC."""

    timestamp = pd.Timestamp(acceptance_datetime)
    if timestamp.tz is None:
        raise ValueError("acceptance_datetime must be timezone-aware")
    return timestamp.tz_convert("UTC")


def eligibility_from_acceptance(acceptance_datetime: object) -> XNYSEligibility:
    """Return the first XNYS session opening strictly after SEC acceptance.

    Scheduled early-close sessions are retained because they have a regular
    session opening. The returned session label and its opening are separate
    from the SEC acceptance timestamp.
    """

    accepted = _as_utc_timestamp(acceptance_datetime)
    session_position = XNYS.opens.searchsorted(accepted, side="right")
    if session_position >= len(XNYS.sessions):
        raise ValueError("acceptance_datetime is after the available XNYS calendar")

    session = XNYS.sessions[session_position]
    return XNYSEligibility(
        acceptance_datetime=accepted,
        eligibility_session=session,
        portfolio_decision_time=XNYS.session_open(session),
    )

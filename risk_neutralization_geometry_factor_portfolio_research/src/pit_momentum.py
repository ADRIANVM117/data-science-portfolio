"""Frozen Stage 1C adjusted-close Momentum components.

This module returns the two raw components separately.  It does not make a
scalar Momentum score, cross-sectional transformation, or portfolio object.
"""

from __future__ import annotations

from dataclasses import dataclass

import exchange_calendars as xcals
import pandas as pd


XNYS = xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class PITMomentumResult:
    """Auditable adjusted-close Momentum inputs at one XNYS-open decision."""

    issuer: str
    decision_time: pd.Timestamp
    information_cutoff: pd.Timestamp | None
    current_price_date: pd.Timestamp | None
    current_adjusted_close: float | None
    twelve_month_reference_date: pd.Timestamp | None
    twelve_month_reference_adjusted_close: float | None
    momentum_12m: float | None
    one_month_reference_date: pd.Timestamp | None
    one_month_reference_adjusted_close: float | None
    momentum_1m: float | None
    status: str
    reason: str


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError("decision_time must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _empty(issuer: str, decision: pd.Timestamp, cutoff: pd.Timestamp | None, status: str, reason: str) -> PITMomentumResult:
    return PITMomentumResult(issuer, decision, cutoff, None, None, None, None, None, None, None, None, status, reason)


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    if "price_date" in frame.columns:
        frame = frame.set_index("price_date")
    if "adjusted_close" not in frame.columns:
        raise ValueError("prices requires adjusted_close")
    frame.index = pd.to_datetime(frame.index, errors="coerce").tz_localize(None).normalize()
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    return frame.loc[frame.index.notna()].sort_index()


def _decision_session(decision: pd.Timestamp) -> pd.Timestamp:
    positions = pd.DatetimeIndex(XNYS.opens.to_numpy()).get_indexer([decision])
    if positions[0] < 0:
        raise ValueError("decision_time must equal an XNYS regular-session opening")
    return XNYS.sessions[positions[0]]


def _session_on_or_before(calendar_date: pd.Timestamp) -> pd.Timestamp:
    position = XNYS.sessions.searchsorted(calendar_date.normalize(), side="right") - 1
    if position < 0:
        raise ValueError("lookback precedes available XNYS calendar")
    return XNYS.sessions[position].tz_localize(None).normalize()


def momentum_as_of(prices: pd.DataFrame, issuer: str, decision_time: object) -> PITMomentumResult:
    """Return frozen `[R_12M, R_1M]` using adjusted close through `t-1`.

    The 12M/1M anchors use calendar year/month offsets, mapped to the last
    eligible XNYS session on or before each anchor.  The caller must supply an
    exact XNYS session open so that no same-session close can enter.
    """

    decision = _utc(decision_time)
    session = _decision_session(decision)
    cutoff = XNYS.previous_session(session).tz_localize(None).normalize()
    frame = _prepare_prices(prices)
    reference_12m = _session_on_or_before(cutoff - pd.DateOffset(years=1))
    reference_1m = _session_on_or_before(cutoff - pd.DateOffset(months=1))

    if cutoff not in frame.index:
        return _empty(issuer, decision, cutoff, "NA_MISSING_CUTOFF_ADJUSTED_CLOSE", "no adjusted close for the prior eligible XNYS session")
    if reference_12m not in frame.index:
        return _empty(issuer, decision, cutoff, "NA_INSUFFICIENT_12M_HISTORY", "no adjusted close at the required 12M XNYS reference session")
    if reference_1m not in frame.index:
        return _empty(issuer, decision, cutoff, "NA_INSUFFICIENT_1M_HISTORY", "no adjusted close at the required 1M XNYS reference session")

    current = frame.at[cutoff, "adjusted_close"]
    twelve_month = frame.at[reference_12m, "adjusted_close"]
    one_month = frame.at[reference_1m, "adjusted_close"]
    if pd.isna(current) or pd.isna(twelve_month) or pd.isna(one_month):
        return _empty(issuer, decision, cutoff, "NA_MISSING_ADJUSTED_CLOSE", "one or more required adjusted-close observations are missing")
    if current <= 0 or twelve_month <= 0 or one_month <= 0:
        return _empty(issuer, decision, cutoff, "NA_NONPOSITIVE_ADJUSTED_CLOSE", "adjusted-close return inputs must be positive")

    return PITMomentumResult(
        issuer, decision, cutoff, cutoff, float(current), reference_12m, float(twelve_month),
        float(current / twelve_month - 1), reference_1m, float(one_month), float(current / one_month - 1),
        "VALID", "adjusted-close 12M magnitude and 1M recent movement through the prior XNYS close",
    )

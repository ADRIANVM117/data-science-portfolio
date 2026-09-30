"""Point-in-time Market Cap and raw Earnings Yield under frozen Stage 1C rules."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pit_net_income import PITTTMResult, ttm_net_income_as_of
from xnys_pit import XNYS, eligibility_from_acceptance


@dataclass(frozen=True)
class PITMarketCapResult:
    issuer: str
    decision_time: pd.Timestamp
    raw_price_date: pd.Timestamp | None
    raw_close: float | None
    shares_value: float | None
    shares_source_accession: str | None
    shares_reference_date: pd.Timestamp | None
    shares_pit_eligibility: pd.Timestamp | None
    split_event_date: pd.Timestamp | None
    split_coefficient: float | None
    market_cap: float | None
    status: str
    reason: str


@dataclass(frozen=True)
class PITEarningsYieldResult:
    issuer: str
    decision_time: pd.Timestamp
    net_income: PITTTMResult
    market_cap: PITMarketCapResult
    earnings_yield: float | None
    status: str
    reason: str


def _utc(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _empty_market_cap(issuer: str, decision: pd.Timestamp, status: str, reason: str) -> PITMarketCapResult:
    return PITMarketCapResult(issuer, decision, None, None, None, None, None, None, None, None, None, status, reason)


def _previous_price_date(decision: pd.Timestamp) -> pd.Timestamp:
    positions = pd.DatetimeIndex(XNYS.opens.to_numpy()).get_indexer([decision])
    if positions[0] < 0:
        raise ValueError("decision_time must equal an XNYS session opening")
    previous = XNYS.previous_session(XNYS.sessions[positions[0]])
    return previous.tz_localize(None).normalize()


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    if "price_date" in frame.columns:
        frame = frame.set_index("price_date")
    if "close" not in frame.columns or "split_coefficient" not in frame.columns:
        raise ValueError("prices requires close and split_coefficient columns")
    frame.index = pd.to_datetime(frame.index, errors="coerce").tz_localize(None).normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["split_coefficient"] = pd.to_numeric(frame["split_coefficient"], errors="coerce")
    return frame.loc[frame.index.notna()].sort_index()


def _prepare_shares(shares: pd.DataFrame, issuer: str) -> pd.DataFrame:
    required = {"issuer", "reported_shares", "accn", "shares_observation_date"}
    missing = required.difference(shares.columns)
    if missing:
        raise ValueError(f"shares is missing required columns: {sorted(missing)}")
    frame = shares.loc[shares["issuer"].eq(issuer)].copy()
    if frame.empty:
        frame["pit_eligibility"] = pd.Series(dtype="datetime64[ns, UTC]")
        frame["acceptanceDateTime"] = pd.Series(dtype="datetime64[ns, UTC]")
        return frame
    frame["reported_shares"] = pd.to_numeric(frame["reported_shares"], errors="coerce")
    frame["shares_observation_date"] = pd.to_datetime(frame["shares_observation_date"], errors="coerce")
    if "pit_eligibility" not in frame.columns:
        if "acceptanceDateTime" not in frame.columns:
            raise ValueError("shares requires pit_eligibility or acceptanceDateTime")
        frame["acceptanceDateTime"] = pd.to_datetime(frame["acceptanceDateTime"], utc=True, errors="coerce")
        frame["pit_eligibility"] = frame["acceptanceDateTime"].map(
            lambda accepted: eligibility_from_acceptance(accepted).portfolio_decision_time if pd.notna(accepted) else pd.NaT
        )
    else:
        frame["pit_eligibility"] = pd.to_datetime(frame["pit_eligibility"], utc=True, errors="coerce")
    if "acceptanceDateTime" not in frame.columns:
        frame["acceptanceDateTime"] = frame["pit_eligibility"]
    else:
        frame["acceptanceDateTime"] = pd.to_datetime(frame["acceptanceDateTime"], utc=True, errors="coerce")
    return frame.loc[
        frame["reported_shares"].notna() & frame["shares_observation_date"].notna() & frame["pit_eligibility"].notna()
    ].copy()


def _select_shares(shares: pd.DataFrame, decision: pd.Timestamp) -> tuple[pd.Series | None, str | None]:
    eligible = shares.loc[shares["pit_eligibility"] <= decision].copy()
    if eligible.empty:
        return None, "MISSING_SHARES"
    latest_eligibility = eligible["pit_eligibility"].max()
    latest = eligible.loc[eligible["pit_eligibility"].eq(latest_eligibility)]
    latest_acceptance = latest["acceptanceDateTime"].max()
    latest = latest.loc[latest["acceptanceDateTime"].eq(latest_acceptance)]
    latest_reference = latest["shares_observation_date"].max()
    latest = latest.loc[latest["shares_observation_date"].eq(latest_reference)]
    if latest["reported_shares"].nunique() != 1 or latest["accn"].nunique() != 1:
        return None, "AMBIGUOUS_SHARES"
    return latest.iloc[0], None


def market_cap_as_of(prices: pd.DataFrame, shares: pd.DataFrame, issuer: str, decision_time: object) -> PITMarketCapResult:
    """Return raw-close Market Cap only when SEC shares have a compatible split basis."""

    decision = _utc(decision_time, "decision_time")
    raw_price_date = _previous_price_date(decision)
    price_frame = _prepare_prices(prices)
    if raw_price_date not in price_frame.index or pd.isna(price_frame.loc[raw_price_date, "close"]):
        return _empty_market_cap(issuer, decision, "MISSING_PRICE", "previous eligible session has no raw close")
    raw_close = float(price_frame.loc[raw_price_date, "close"])

    selected_shares, shares_status = _select_shares(_prepare_shares(shares, issuer), decision)
    if selected_shares is None:
        return PITMarketCapResult(
            issuer, decision, raw_price_date, raw_close, None, None, None, None, None, None, None,
            shares_status or "MISSING_SHARES", "no deterministic PIT-eligible SEC shares observation",
        )

    split_events = price_frame.loc[
        (price_frame.index > selected_shares["shares_observation_date"])
        & (price_frame.index <= raw_price_date)
        & price_frame["split_coefficient"].ne(1.0)
    ]
    if not split_events.empty:
        event_date = split_events.index[0]
        coefficient = float(split_events.iloc[0]["split_coefficient"])
        return PITMarketCapResult(
            issuer, decision, raw_price_date, raw_close, float(selected_shares["reported_shares"]), selected_shares["accn"],
            selected_shares["shares_observation_date"], selected_shares["pit_eligibility"], event_date, coefficient,
            None, "SPLIT_BASIS_MISMATCH", "split occurred after selected SEC shares reference date",
        )
    market_cap = raw_close * float(selected_shares["reported_shares"])
    return PITMarketCapResult(
        issuer, decision, raw_price_date, raw_close, float(selected_shares["reported_shares"]), selected_shares["accn"],
        selected_shares["shares_observation_date"], selected_shares["pit_eligibility"], None, None,
        market_cap, "VALID", "raw close and PIT-reported shares have compatible observed split basis",
    )


def earnings_yield_as_of(
    facts: pd.DataFrame, prices: pd.DataFrame, shares: pd.DataFrame, issuer: str, decision_time: object
) -> PITEarningsYieldResult:
    """Return raw PIT Earnings Yield only when PIT TTM Net Income and Market Cap are valid."""

    decision = _utc(decision_time, "decision_time")
    net_income = ttm_net_income_as_of(facts, issuer, decision)
    market_cap = market_cap_as_of(prices, shares, issuer, decision)
    if net_income.status != "VALID":
        return PITEarningsYieldResult(issuer, decision, net_income, market_cap, None, "INVALID_TTM", net_income.status)
    if market_cap.status != "VALID":
        return PITEarningsYieldResult(issuer, decision, net_income, market_cap, None, market_cap.status, market_cap.reason)
    return PITEarningsYieldResult(
        issuer, decision, net_income, market_cap, net_income.value / market_cap.market_cap, "VALID", "PIT TTM Net Income divided by PIT Market Cap"
    )

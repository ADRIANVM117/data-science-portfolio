"""Offline price alignment, return calculation, and coverage accounting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import polars as pl


@dataclass(frozen=True)
class Coverage:
    symbol: str; observations: int; leading_nulls: int; interior_nulls: int; trailing_nulls: int; first_observation: date | None; last_observation: date | None; eligible: bool; reason: str


def price_matrix(series: dict[str, pl.DataFrame]) -> pl.DataFrame:
    matrix = None
    for symbol, frame in sorted(series.items()):
        part = frame.select("session_date", pl.col("adjusted_close").alias(symbol))
        matrix = part if matrix is None else matrix.join(part, on="session_date", how="full", coalesce=True)
    if matrix is None: raise ValueError("no approved series")
    return matrix.sort("session_date")


def simple_returns(prices: pl.DataFrame, expected_sessions: dict[str, set[date]]) -> pl.DataFrame:
    """Return only across adjacent expected sessions for each instrument."""
    rows = {"session_date": prices["session_date"].to_list()}
    dates = rows["session_date"]
    for col in prices.columns[1:]:
        if col not in expected_sessions: raise ValueError(f"missing calendar for {col}")
        allowed = expected_sessions[col]; values = prices[col].to_list(); output = []
        previous = None
        for current, value in zip(dates, values):
            if current not in allowed:
                output.append(None)
                continue
            consecutive = previous is not None and all(day not in allowed for day in _days_between(previous[0], current))
            output.append(value / previous[1] - 1 if consecutive and value is not None and previous[1] is not None else None)
            previous = (current, value)
        rows[col] = output
    return pl.DataFrame(rows, schema={"session_date": pl.Date, **{c: pl.Float64 for c in prices.columns[1:]}})


def _days_between(left: date, right: date):
    from datetime import timedelta
    day = left + timedelta(days=1)
    while day < right:
        yield day; day += timedelta(days=1)


def coverage(prices: pl.DataFrame, cutoff: str, minimum_observations: int, expected_sessions: dict[str, set[date]]) -> list[Coverage]:
    results = []
    for col in prices.columns[1:]:
        values = prices[col].to_list(); valid = [i for i, v in enumerate(values) if v is not None]
        if not valid: results.append(Coverage(col,0,0,0,len(values),None,None,False,"ALL_NULL")); continue
        if col not in expected_sessions: raise ValueError(f"missing calendar for {col}")
        first,last=valid[0],valid[-1]; expected=expected_sessions[col]; observed_dates=prices["session_date"].to_list(); observed_by_date=dict(zip(observed_dates, values)); first_date=observed_dates[first]
        unexpected_non_session = any(value is not None and day not in expected for day, value in zip(observed_dates, values))
        leading=sum(observed_by_date.get(day) is None for day in expected if day < first_date)
        interior=sum(observed_by_date.get(day) is None for day in expected if first_date <= day <= observed_dates[last])
        trailing=sum(observed_by_date.get(day) is None for day in expected if observed_dates[last] < day <= date.fromisoformat(cutoff))
        # Eligibility requires the approved common window in full. Leading gaps
        # are retained in diagnostic matrices but are insufficient history.
        if leading and interior:
            reason = "INSUFFICIENT_HISTORY_AND_INTERIOR_NULL"
        elif leading:
            reason = "INSUFFICIENT_HISTORY"
        elif interior:
            reason = "INTERIOR_NULL"
        elif unexpected_non_session:
            reason = "UNEXPECTED_NON_SESSION"
        elif trailing or str(prices["session_date"].max()) != cutoff:
            reason = "MISSING_CUTOFF"
        elif len(valid) < minimum_observations:
            reason = "INSUFFICIENT_HISTORY"
        else:
            reason = "ELIGIBLE"
        results.append(Coverage(col,len(valid),leading,interior,trailing,first_date,observed_dates[last],reason == "ELIGIBLE",reason))
    return results

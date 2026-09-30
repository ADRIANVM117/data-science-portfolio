"""PIT Revenue TTM and frozen two-component Revenue Growth.

This module returns separate magnitude (three-year Revenue CAGR) and
persistence (fraction of positive annual Revenue growth intervals).  It never
combines them into a scalar Growth score.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pit_net_income import PITTTMResult, ttm_net_income_as_of
from xnys_pit import eligibility_from_acceptance


REVENUE_TAGS = {
    "AAPL": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "JPM": "RevenuesNetOfInterestExpense",
    "XOM": "Revenues",
    "UNH": "Revenues",
    "WMT": "Revenues",
}
REQUIRED_REVENUE_COLUMNS = {
    "issuer", "cik", "tag", "unit", "start", "end", "value", "form", "accn", "acceptanceDateTime"
}
ANNUAL_FORMS = {"10-K", "10-K/A"}


@dataclass(frozen=True)
class PITRevenueTTMResult:
    """PIT Revenue TTM and four selected revenue quarters/provenance."""

    issuer: str
    decision_time: pd.Timestamp
    value: float | None
    status: str
    quarters: pd.DataFrame
    revenue_tag: str


@dataclass(frozen=True)
class PITRevenueGrowthResult:
    """Frozen raw Revenue Growth components, or an explained NA."""

    issuer: str
    decision_time: pd.Timestamp
    revenue_ttm_t3: PITRevenueTTMResult | None
    revenue_ttm_t2: PITRevenueTTMResult | None
    revenue_ttm_t1: PITRevenueTTMResult | None
    revenue_ttm_t0: PITRevenueTTMResult | None
    annual_growth_1: float | None
    annual_growth_2: float | None
    annual_growth_3: float | None
    growth_magnitude: float | None
    growth_persistence: float | None
    status: str
    reason: str


def _utc(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _tag_for(issuer: str, revenue_tag: str | None) -> str:
    if revenue_tag is not None:
        return revenue_tag
    if issuer not in REVENUE_TAGS:
        raise ValueError(f"no frozen Revenue tag configured for issuer {issuer!r}")
    return REVENUE_TAGS[issuer]


def revenue_ttm_as_of(
    facts: pd.DataFrame, issuer: str, decision_time: object, revenue_tag: str | None = None
) -> PITRevenueTTMResult:
    """Build PIT Revenue TTM using the frozen compatible-quarter policy.

    The underlying builder enforces duration classes, prospective amendments,
    duplicate handling, compatible cumulative arithmetic, and no look-ahead.
    The temporary technical label never mutates raw facts; output retains the
    selected Revenue taxonomy tag and original accession provenance.
    """

    missing = REQUIRED_REVENUE_COLUMNS.difference(facts.columns)
    if missing:
        raise ValueError(f"facts is missing required columns: {sorted(missing)}")
    tag = _tag_for(issuer, revenue_tag)
    revenue = facts.loc[facts["tag"].eq(tag)].copy()
    revenue["tag"] = "NetIncomeLoss"
    result = ttm_net_income_as_of(revenue, issuer, decision_time)
    quarters = result.quarters.copy()
    if not quarters.empty:
        quarters["revenue_tag"] = tag
    return PITRevenueTTMResult(result.issuer, result.decision_time, result.value, result.status, quarters, tag)


def _annual_anchors(
    facts: pd.DataFrame, issuer: str, decision: pd.Timestamp, tag: str
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp]], str | None]:
    """Return first-original PIT eligibility anchors, one per fiscal annual end."""

    annual = facts.loc[(facts["issuer"].eq(issuer)) & (facts["tag"].eq(tag))].copy()
    annual["start"] = pd.to_datetime(annual["start"], errors="coerce")
    annual["end"] = pd.to_datetime(annual["end"], errors="coerce")
    annual["acceptanceDateTime"] = pd.to_datetime(annual["acceptanceDateTime"], utc=True, errors="coerce")
    annual["value"] = pd.to_numeric(annual["value"], errors="coerce")
    annual = annual.loc[
        annual["form"].isin(ANNUAL_FORMS)
        & annual["start"].notna() & annual["end"].notna()
        & annual["acceptanceDateTime"].notna() & annual["value"].notna()
    ].copy()
    annual["duration_days"] = (annual["end"] - annual["start"]).dt.days
    annual = annual.loc[annual["duration_days"].between(301, 390)].copy()
    annual["pit_eligibility"] = annual["acceptanceDateTime"].map(
        lambda accepted: eligibility_from_acceptance(accepted).portfolio_decision_time
    )
    annual = annual.loc[annual["pit_eligibility"] <= decision].copy()
    anchors: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for end, rows in annual.loc[~annual["form"].str.endswith("/A")].groupby("end", sort=True):
        first_time = rows["acceptanceDateTime"].min()
        selected = rows.loc[rows["acceptanceDateTime"].eq(first_time)]
        if selected["value"].nunique() != 1 or selected["accn"].nunique() != 1 or selected["unit"].nunique() != 1:
            return [], "NA_AMBIGUOUS_ANNUAL_VINTAGE"
        anchors.append((end, selected.iloc[0]["pit_eligibility"]))
    if len(anchors) < 4:
        return anchors, "NA_FEWER_THAN_FOUR_ANNUAL_TTMS"
    return anchors, None


def _annual_consecutive(history: list[PITRevenueTTMResult]) -> bool:
    ends = [result.quarters["economic_quarter_end"].max() for result in history]
    return all(pd.Timedelta(days=300) <= later - earlier <= pd.Timedelta(days=400) for earlier, later in zip(ends, ends[1:]))


def _failure(
    issuer: str, decision: pd.Timestamp, history: list[PITRevenueTTMResult], status: str, reason: str
) -> PITRevenueGrowthResult:
    values = [None, None, None, None]
    for index, value in enumerate(history[-4:]):
        values[4 - len(history[-4:]) + index] = value
    return PITRevenueGrowthResult(issuer, decision, *values, None, None, None, None, None, status, reason)


def revenue_growth_as_of(
    facts: pd.DataFrame, issuer: str, decision_time: object, revenue_tag: str | None = None
) -> PITRevenueGrowthResult:
    """Return frozen three-year CAGR and positive-growth fraction from four PIT TTMs.

    Historical TTM points are each reconstructed at their own first-original
    annual filing eligibility time. Later comparative disclosures and later
    amendments therefore cannot retrospectively rewrite a prior point.
    """

    missing = REQUIRED_REVENUE_COLUMNS.difference(facts.columns)
    if missing:
        raise ValueError(f"facts is missing required columns: {sorted(missing)}")
    decision = _utc(decision_time, "decision_time")
    tag = _tag_for(issuer, revenue_tag)
    anchors, anchor_status = _annual_anchors(facts, issuer, decision, tag)
    history = [revenue_ttm_as_of(facts, issuer, anchor_time, tag) for _, anchor_time in anchors[-4:]]
    if anchor_status is not None:
        return _failure(issuer, decision, history, anchor_status, "four annual PIT Revenue TTM observations are required")
    if any(result.status != "VALID" for result in history):
        return _failure(issuer, decision, history, "INVALID_REVENUE_TTM_HISTORY", "at least one annual PIT Revenue TTM is invalid")
    if len({result.revenue_tag for result in history}) != 1:
        return _failure(issuer, decision, history, "INCOMPATIBLE_REVENUE_TAG_HISTORY", "Revenue tags cannot be spliced")
    if not _annual_consecutive(history):
        return _failure(issuer, decision, history, "NA_NONANNUAL_REVENUE_HISTORY", "annual Revenue TTM economic ends are not consecutive")
    values = [result.value for result in history]
    if any(value is None or value <= 0 for value in values):
        return _failure(issuer, decision, history, "INVALID_REVENUE_LEVEL", "Revenue CAGR requires positive annual TTM levels")
    g1, g2, g3 = (values[1] / values[0] - 1, values[2] / values[1] - 1, values[3] / values[2] - 1)
    magnitude = (values[3] / values[0]) ** (1 / 3) - 1
    persistence = sum(growth > 0 for growth in (g1, g2, g3)) / 3
    return PITRevenueGrowthResult(
        issuer, decision, *history, g1, g2, g3, magnitude, persistence, "VALID",
        "three-year PIT Revenue CAGR and positive annual growth fraction",
    )

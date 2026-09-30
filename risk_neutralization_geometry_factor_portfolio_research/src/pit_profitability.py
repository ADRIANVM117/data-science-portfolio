"""PIT ROA-like Profitability under the frozen Stage 1C policy.

This module combines a valid PIT TTM ``us-gaap:NetIncomeLoss`` flow with
two exact, PIT-eligible ``us-gaap:Assets`` instants.  It deliberately does
not impute a missing balance-sheet boundary, construct a Quality composite,
or perform cross-sectional transformations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pit_net_income import PITTTMResult, ttm_net_income_as_of
from xnys_pit import eligibility_from_acceptance


ASSET_REQUIRED_COLUMNS = {
    "issuer", "cik", "tag", "unit", "reference_date", "value", "form", "accn", "acceptanceDateTime"
}


@dataclass(frozen=True)
class PITAssetSnapshot:
    """One selected, reported balance-sheet state and its filing provenance."""

    issuer: str
    cik: str | None
    value: float | None
    reference_date: pd.Timestamp | None
    unit: str | None
    source_accession: str | None
    source_form: str | None
    acceptance_datetime: pd.Timestamp | None
    pit_eligibility: pd.Timestamp | None
    status: str
    reason: str


@dataclass(frozen=True)
class PITAverageAssetsResult:
    """Exact-boundary Assets denominator aligned to a PIT Net Income TTM."""

    issuer: str
    decision_time: pd.Timestamp
    net_income: PITTTMResult
    ttm_economic_start: pd.Timestamp | None
    ttm_economic_end: pd.Timestamp | None
    beginning_assets: PITAssetSnapshot
    ending_assets: PITAssetSnapshot
    average_assets: float | None
    status: str
    reason: str


@dataclass(frozen=True)
class PITProfitabilityResult:
    """Raw Profitability = PIT TTM Net Income / PIT Average Assets."""

    issuer: str
    decision_time: pd.Timestamp
    net_income: PITTTMResult
    average_assets: PITAverageAssetsResult
    profitability: float | None
    status: str
    reason: str


def _utc(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _empty_snapshot(issuer: str, status: str, reason: str) -> PITAssetSnapshot:
    return PITAssetSnapshot(issuer, None, None, None, None, None, None, None, None, status, reason)


def _prepare_assets(assets: pd.DataFrame, issuer: str, decision: pd.Timestamp) -> pd.DataFrame:
    missing = ASSET_REQUIRED_COLUMNS.difference(assets.columns)
    if missing:
        raise ValueError(f"assets is missing required columns: {sorted(missing)}")
    frame = assets.loc[assets["issuer"].eq(issuer)].copy()
    frame["reference_date"] = pd.to_datetime(frame["reference_date"], errors="coerce").dt.normalize()
    frame["acceptanceDateTime"] = pd.to_datetime(frame["acceptanceDateTime"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[
        frame["tag"].eq("Assets")
        & frame["reference_date"].notna()
        & frame["acceptanceDateTime"].notna()
        & frame["value"].notna()
    ].copy()
    frame["pit_eligibility"] = frame["acceptanceDateTime"].map(
        lambda accepted: eligibility_from_acceptance(accepted).portfolio_decision_time
    )
    return frame.loc[frame["pit_eligibility"] <= decision].copy()


def _snapshot_from_row(row: pd.Series) -> PITAssetSnapshot:
    return PITAssetSnapshot(
        issuer=str(row["issuer"]), cik=str(row["cik"]), value=float(row["value"]),
        reference_date=row["reference_date"], unit=str(row["unit"]), source_accession=str(row["accn"]),
        source_form=str(row["form"]), acceptance_datetime=row["acceptanceDateTime"],
        pit_eligibility=row["pit_eligibility"], status="VALID",
        reason="exact Assets boundary selected under prospective-vintage policy",
    )


def asset_snapshot_as_of(
    assets: pd.DataFrame, issuer: str, reference_date: object, decision_time: object
) -> PITAssetSnapshot:
    """Select one exact PIT Assets instant, or return an auditable failure.

    The first eligible original disclosure is retained.  A later eligible
    amendment supersedes it only prospectively.  A later non-amendment
    comparative disclosure cannot rewrite the original.  Competing facts in
    the selected vintage are deliberately unresolved rather than guessed.
    """

    decision = _utc(decision_time, "decision_time")
    boundary = pd.Timestamp(reference_date).tz_localize(None).normalize()
    frame = _prepare_assets(assets, issuer, decision)
    matching = frame.loc[frame["reference_date"].eq(boundary)].copy()
    if matching.empty:
        return _empty_snapshot(
            issuer, "MISSING_ASSET_BOUNDARY",
            f"no PIT-eligible us-gaap:Assets fact for exact boundary {boundary.date()}",
        )
    if matching[["issuer", "cik", "tag", "unit"]].nunique().gt(1).any():
        return _empty_snapshot(issuer, "INCOMPATIBLE_ASSET_CONTEXT", "issuer/CIK/tag/unit are not mutually compatible")

    matching["is_amendment"] = matching["form"].str.endswith("/A")
    amendments = matching.loc[matching["is_amendment"]]
    pool = amendments if not amendments.empty else matching.loc[~matching["is_amendment"]]
    ordered = pool.sort_values(["acceptanceDateTime", "accn"])
    selected_time = ordered["acceptanceDateTime"].iloc[-1] if not amendments.empty else ordered["acceptanceDateTime"].iloc[0]
    selected = pool.loc[pool["acceptanceDateTime"].eq(selected_time)]
    if selected["value"].nunique() != 1 or selected["accn"].nunique() != 1:
        return _empty_snapshot(
            issuer, "AMBIGUOUS_ASSET_VINTAGE",
            "competing Assets facts exist in the selected filing vintage",
        )
    return _snapshot_from_row(selected.iloc[0])


def _average_failure(
    issuer: str, decision: pd.Timestamp, net_income: PITTTMResult, start: pd.Timestamp | None,
    end: pd.Timestamp | None, beginning: PITAssetSnapshot, ending: PITAssetSnapshot, status: str, reason: str,
) -> PITAverageAssetsResult:
    return PITAverageAssetsResult(issuer, decision, net_income, start, end, beginning, ending, None, status, reason)


def average_assets_as_of(
    net_income_facts: pd.DataFrame, assets: pd.DataFrame, issuer: str, decision_time: object
) -> PITAverageAssetsResult:
    """Return average Assets only for the two exact TTM economic boundaries."""

    decision = _utc(decision_time, "decision_time")
    net_income = ttm_net_income_as_of(net_income_facts, issuer, decision)
    if net_income.status != "VALID":
        empty = _empty_snapshot(issuer, "NOT_EVALUATED", "TTM Net Income is not valid")
        return _average_failure(issuer, decision, net_income, None, None, empty, empty, "INVALID_TTM", net_income.status)

    start = net_income.quarters["economic_quarter_start"].min()
    end = net_income.quarters["economic_quarter_end"].max()
    beginning_boundary = start - pd.Timedelta(days=1)
    beginning = asset_snapshot_as_of(assets, issuer, beginning_boundary, decision)
    ending = asset_snapshot_as_of(assets, issuer, end, decision)

    if beginning.status == "MISSING_ASSET_BOUNDARY" or ending.status == "MISSING_ASSET_BOUNDARY":
        missing = "beginning" if beginning.status == "MISSING_ASSET_BOUNDARY" else "ending"
        return _average_failure(
            issuer, decision, net_income, start, end, beginning, ending, "INCOMPATIBLE_ASSET_BOUNDARY",
            f"{missing} exact economic boundary is unavailable; nearest Assets date is not substituted",
        )
    if beginning.status != "VALID":
        return _average_failure(issuer, decision, net_income, start, end, beginning, ending, beginning.status, beginning.reason)
    if ending.status != "VALID":
        return _average_failure(issuer, decision, net_income, start, end, beginning, ending, ending.status, ending.reason)
    if beginning.cik != ending.cik or beginning.unit != ending.unit:
        return _average_failure(
            issuer, decision, net_income, start, end, beginning, ending, "INCOMPATIBLE_ASSET_CONTEXT",
            "beginning and ending Assets snapshots have different CIK or unit",
        )
    average = (beginning.value + ending.value) / 2
    return PITAverageAssetsResult(
        issuer, decision, net_income, start, end, beginning, ending, average, "VALID",
        "arithmetic mean of exact, compatible PIT Assets boundaries",
    )


def profitability_as_of(
    net_income_facts: pd.DataFrame, assets: pd.DataFrame, issuer: str, decision_time: object
) -> PITProfitabilityResult:
    """Return the frozen raw ROA-like Profitability component, or an explained NA."""

    decision = _utc(decision_time, "decision_time")
    average_assets = average_assets_as_of(net_income_facts, assets, issuer, decision)
    if average_assets.net_income.status != "VALID":
        return PITProfitabilityResult(
            issuer, decision, average_assets.net_income, average_assets, None, "INVALID_TTM", average_assets.net_income.status
        )
    if average_assets.status != "VALID" or average_assets.average_assets is None or average_assets.average_assets == 0:
        status = "INVALID_AVERAGE_ASSETS" if average_assets.status == "VALID" else average_assets.status
        reason = "Average Assets is zero" if average_assets.status == "VALID" else average_assets.reason
        return PITProfitabilityResult(issuer, decision, average_assets.net_income, average_assets, None, status, reason)
    return PITProfitabilityResult(
        issuer, decision, average_assets.net_income, average_assets,
        average_assets.net_income.value / average_assets.average_assets, "VALID",
        "PIT TTM Net Income divided by PIT Average Assets",
    )

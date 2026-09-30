"""Point-in-time standalone-quarter and TTM Net Income construction.

The functions here implement the frozen Stage 1C information policy for
``us-gaap:NetIncomeLoss``. They do not calculate Earnings Yield, market
capitalization, or a Value characteristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from xnys_pit import eligibility_from_acceptance


REQUIRED_COLUMNS = {
    "issuer", "cik", "tag", "unit", "start", "end", "value", "form", "accn", "acceptanceDateTime"
}
DIRECT_FORMS = {"10-Q", "10-Q/A"}
ANNUAL_FORMS = {"10-K", "10-K/A"}


@dataclass(frozen=True)
class PITTTMResult:
    """TTM value and the four selected PIT-eligible quarter records."""

    issuer: str
    decision_time: pd.Timestamp
    value: float | None
    status: str
    quarters: pd.DataFrame


def _utc(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _prepare_facts(facts: pd.DataFrame, issuer: str, decision_time: object) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(facts.columns)
    if missing:
        raise ValueError(f"facts is missing required columns: {sorted(missing)}")

    decision = _utc(decision_time, "decision_time")
    frame = facts.loc[facts["issuer"].eq(issuer)].copy()
    frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    frame["acceptanceDateTime"] = pd.to_datetime(frame["acceptanceDateTime"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.loc[
        frame["tag"].eq("NetIncomeLoss")
        & frame["start"].notna()
        & frame["end"].notna()
        & frame["acceptanceDateTime"].notna()
        & frame["value"].notna()
    ].copy()
    frame["pit_eligibility"] = frame["acceptanceDateTime"].map(
        lambda accepted: eligibility_from_acceptance(accepted).portfolio_decision_time
    )
    frame = frame.loc[frame["pit_eligibility"] <= decision].copy()
    frame["duration_days"] = (frame["end"] - frame["start"]).dt.days
    return frame


def _select_vintage(rows: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    """Select an original or prospective amendment for one exact context.

    The first accepted non-amendment filing is the deterministic original
    disclosure. A later amendment supersedes it only from its own eligibility.
    Conflicting rows within the selected accession/time or incompatible units
    remain unresolved and return ``None``.
    """

    if rows.empty:
        return None, "NA_MISSING"
    if rows[["issuer", "cik", "tag", "unit"]].nunique().gt(1).any():
        return None, "NA_INCOMPATIBLE_CONTEXT"

    candidates = rows.copy()
    candidates["is_amendment"] = candidates["form"].str.endswith("/A")
    amendments = candidates.loc[candidates["is_amendment"]]
    pool = amendments if not amendments.empty else candidates.loc[~candidates["is_amendment"]]
    ordering = pool.sort_values(["acceptanceDateTime", "accn"])
    selected_time = ordering["acceptanceDateTime"].iloc[-1] if not amendments.empty else ordering["acceptanceDateTime"].iloc[0]
    selected = pool.loc[pool["acceptanceDateTime"].eq(selected_time)]
    if selected["value"].nunique() != 1 or selected["accn"].nunique() != 1:
        return None, "NA_AMBIGUOUS_VINTAGE"
    return selected.iloc[0], None


def _record(
    issuer: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    value: float | None,
    status: str,
    method: str,
    sources: list[pd.Series],
    direct_value: float | None = None,
    reconstructed_value: float | None = None,
    issue: str | None = None,
) -> dict[str, Any]:
    return {
        "issuer": issuer,
        "cik": None if not sources else sources[0]["cik"],
        "economic_quarter_start": start,
        "economic_quarter_end": end,
        "value": value,
        "status": status,
        "method": method,
        "source_accessions": tuple(source["accn"] for source in sources),
        "source_acceptance_datetimes": tuple(source["acceptanceDateTime"] for source in sources),
        "pit_eligibility": tuple(source["pit_eligibility"] for source in sources),
        "source_forms": tuple(source["form"] for source in sources),
        "direct_value": direct_value,
        "reconstructed_value": reconstructed_value,
        "issue": issue,
    }


def build_quarters_as_of(facts: pd.DataFrame, issuer: str, decision_time: object) -> pd.DataFrame:
    """Build auditable PIT standalone Net Income quarters for one issuer/time.

    Direct Q1/Q2/Q3 candidates come from 10-Q-family facts with 70--115 day
    durations. Compatible cumulative candidates may construct Q2, Q3, and Q4
    using only the three frozen identities. The result has at most one record
    per economic quarter; invalid records explicitly carry an ``NA_*`` status.
    """

    frame = _prepare_facts(facts, issuer, decision_time)
    if frame.empty:
        return pd.DataFrame(columns=[
            "issuer", "cik", "economic_quarter_start", "economic_quarter_end", "value", "status", "method",
            "source_accessions", "source_acceptance_datetimes", "pit_eligibility", "source_forms",
            "direct_value", "reconstructed_value", "issue",
        ])

    q_forms = frame["form"].isin(DIRECT_FORMS)
    k_forms = frame["form"].isin(ANNUAL_FORMS)
    direct_rows = frame.loc[q_forms & frame["duration_days"].between(70, 115)]
    h1_rows = frame.loc[q_forms & frame["duration_days"].between(116, 200)]
    y9_rows = frame.loc[q_forms & frame["duration_days"].between(201, 300)]
    fy_rows = frame.loc[k_forms & frame["duration_days"].between(301, 390)]

    records: dict[tuple[pd.Timestamp, pd.Timestamp], dict[str, Any]] = {}
    direct: dict[tuple[pd.Timestamp, pd.Timestamp], pd.Series] = {}

    for key, rows in direct_rows.groupby(["start", "end"], sort=True):
        selected, issue = _select_vintage(rows)
        if selected is None:
            records[key] = _record(issuer, key[0], key[1], None, issue or "NA_AMBIGUOUS", "direct", [rows.iloc[i] for i in range(len(rows))], issue=issue)
        else:
            direct[key] = selected
            records[key] = _record(issuer, key[0], key[1], float(selected["value"]), "VALID", "direct", [selected], direct_value=float(selected["value"]))

    def selected_groups(rows: pd.DataFrame) -> list[pd.Series]:
        result: list[pd.Series] = []
        for _, group in rows.groupby(["start", "end"], sort=True):
            selected, _ = _select_vintage(group)
            if selected is not None:
                result.append(selected)
        return result

    h1 = selected_groups(h1_rows)
    y9 = selected_groups(y9_rows)
    annual = selected_groups(fy_rows)

    def merge_reconstruction(
        start: pd.Timestamp, end: pd.Timestamp, reconstructed: float, sources: list[pd.Series], label: str
    ) -> None:
        key = (start, end)
        existing = records.get(key)
        if existing is None:
            records[key] = _record(issuer, start, end, reconstructed, "VALID", "reconstructed", sources, reconstructed_value=reconstructed)
            return
        if existing["status"] != "VALID":
            return
        if existing["cik"] != sources[0]["cik"]:
            records[key] = _record(
                issuer, start, end, None, "NA_INCOMPATIBLE_CONTEXT", "ambiguous", sources,
                direct_value=existing["direct_value"], reconstructed_value=reconstructed,
                issue=f"{label} and direct-quarter fact have different CIK provenance",
            )
            return
        if float(existing["value"]) != float(reconstructed):
            records[key] = _record(
                issuer, start, end, None, "NA_DIRECT_RECONSTRUCTED_MISMATCH", "ambiguous",
                [*sources, *([direct[key]] if key in direct else [])], direct_value=existing["direct_value"],
                reconstructed_value=reconstructed, issue=f"{label} disagrees with direct-quarter fact",
            )
            return
        existing["reconstructed_value"] = reconstructed
        existing["issue"] = f"direct/reconstructed check passed: {label}"

    for cumulative in h1:
        q1_candidates = [
            fact for (start, end), fact in direct.items()
            if start == cumulative["start"] and end < cumulative["end"]
            and fact["unit"] == cumulative["unit"] and fact["cik"] == cumulative["cik"]
        ]
        if len(q1_candidates) != 1:
            continue
        q1 = q1_candidates[0]
        q2_start = q1["end"] + pd.Timedelta(days=1)
        merge_reconstruction(q2_start, cumulative["end"], float(cumulative["value"] - q1["value"]), [cumulative, q1], "H1_YTD - Q1")

    for cumulative in y9:
        h1_candidates = [
            fact for fact in h1
            if fact["start"] == cumulative["start"] and fact["end"] < cumulative["end"]
            and fact["unit"] == cumulative["unit"] and fact["cik"] == cumulative["cik"]
        ]
        if len(h1_candidates) != 1:
            continue
        h1_fact = h1_candidates[0]
        q3_start = h1_fact["end"] + pd.Timedelta(days=1)
        merge_reconstruction(q3_start, cumulative["end"], float(cumulative["value"] - h1_fact["value"]), [cumulative, h1_fact], "9M_YTD - H1_YTD")

    for fy in annual:
        y9_candidates = [
            fact for fact in y9
            if fact["start"] == fy["start"] and fact["end"] < fy["end"]
            and fact["unit"] == fy["unit"] and fact["cik"] == fy["cik"]
        ]
        if len(y9_candidates) != 1:
            continue
        y9_fact = y9_candidates[0]
        q4_start = y9_fact["end"] + pd.Timedelta(days=1)
        merge_reconstruction(q4_start, fy["end"], float(fy["value"] - y9_fact["value"]), [fy, y9_fact], "FY - 9M_YTD")

    return pd.DataFrame(records.values()).sort_values(["economic_quarter_end", "economic_quarter_start"]).reset_index(drop=True)


def _are_consecutive(quarters: pd.DataFrame) -> bool:
    ordered = quarters.sort_values("economic_quarter_start")
    previous_end: pd.Timestamp | None = None
    for row in ordered.itertuples():
        if previous_end is not None and not (pd.Timedelta(days=0) < row.economic_quarter_start - previous_end <= pd.Timedelta(days=7)):
            return False
        previous_end = row.economic_quarter_end
    return True


def ttm_net_income_as_of(facts: pd.DataFrame, issuer: str, decision_time: object) -> PITTTMResult:
    """Return the four latest valid, distinct, consecutive PIT quarters and TTM."""

    decision = _utc(decision_time, "decision_time")
    quarters = build_quarters_as_of(facts, issuer, decision)
    valid = quarters.loc[quarters["status"].eq("VALID")].sort_values("economic_quarter_end", ascending=False).head(4).copy()
    if len(valid) != 4:
        return PITTTMResult(issuer, decision, None, "NA_FEWER_THAN_FOUR_VALID_QUARTERS", valid)
    if valid[["economic_quarter_start", "economic_quarter_end"]].duplicated().any():
        return PITTTMResult(issuer, decision, None, "NA_DUPLICATE_ECONOMIC_QUARTER", valid)
    if not _are_consecutive(valid):
        return PITTTMResult(issuer, decision, None, "NA_NONCONSECUTIVE_QUARTERS", valid)
    return PITTTMResult(issuer, decision, float(valid["value"].sum()), "VALID", valid.sort_values("economic_quarter_end").reset_index(drop=True))

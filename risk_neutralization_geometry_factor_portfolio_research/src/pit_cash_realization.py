"""PIT Operating Cash Flow TTM and frozen Cash Realization.

Cash Realization is defined only for positive PIT TTM Net Income.  Its cap is
upper-only: negative operating cash flow remains negative rather than being
floored at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pit_net_income import PITTTMResult, ttm_net_income_as_of


OCF_TAG = "NetCashProvidedByUsedInOperatingActivities"
REQUIRED_OCF_COLUMNS = {
    "issuer", "cik", "tag", "unit", "start", "end", "value", "form", "accn", "acceptanceDateTime"
}


@dataclass(frozen=True)
class PITOperatingCashFlowResult:
    """PIT OCF TTM plus the four selected cash-flow quarters and provenance."""

    issuer: str
    decision_time: pd.Timestamp
    value: float | None
    status: str
    quarters: pd.DataFrame
    accounting_tag: str = OCF_TAG


@dataclass(frozen=True)
class PITCashRealizationResult:
    """Frozen upper-capped OCF/NI representation, or an explained NA."""

    issuer: str
    decision_time: pd.Timestamp
    operating_cash_flow: PITOperatingCashFlowResult
    net_income: PITTTMResult
    raw_ratio: float | None
    cash_realization: float | None
    status: str
    reason: str


def ttm_operating_cash_flow_as_of(
    facts: pd.DataFrame, issuer: str, decision_time: object
) -> PITOperatingCashFlowResult:
    """Reconstruct PIT OCF TTM using the frozen compatible-quarter identities.

    The existing quarterly builder already implements the frozen duration,
    vintage, amendment, duplicate, direct-vs-reconstructed, and no-look-ahead
    policy. This narrow adapter verifies the OCF taxonomy tag, then uses those
    mechanics on a copy solely to obtain the auditable quarterly construction.
    Provenance columns and selected source accessions remain unchanged.
    """

    missing = REQUIRED_OCF_COLUMNS.difference(facts.columns)
    if missing:
        raise ValueError(f"facts is missing required columns: {sorted(missing)}")
    ocf_facts = facts.copy()
    ocf_facts = ocf_facts.loc[ocf_facts["tag"].eq(OCF_TAG)].copy()
    # ``pit_net_income`` is a validated generic duration/vintage engine; the
    # temporary label is technical only and never changes raw evidence.
    ocf_facts["tag"] = "NetIncomeLoss"
    result = ttm_net_income_as_of(ocf_facts, issuer, decision_time)
    quarters = result.quarters.copy()
    if not quarters.empty:
        quarters["accounting_tag"] = OCF_TAG
    return PITOperatingCashFlowResult(
        result.issuer, result.decision_time, result.value, result.status, quarters
    )


def cash_realization_as_of(
    net_income_facts: pd.DataFrame, operating_cash_flow_facts: pd.DataFrame,
    issuer: str, decision_time: object,
) -> PITCashRealizationResult:
    """Return frozen upper-capped Cash Realization for one issuer/time.

    ``NI <= 0`` is not a ratio case: it is outside this component's economic
    meaning and returns ``NA``.  For positive NI, ``min(OCF / NI, 1.0)`` is
    applied without any lower cap.
    """

    net_income = ttm_net_income_as_of(net_income_facts, issuer, decision_time)
    operating_cash_flow = ttm_operating_cash_flow_as_of(operating_cash_flow_facts, issuer, decision_time)
    decision = net_income.decision_time
    if net_income.status != "VALID":
        return PITCashRealizationResult(
            issuer, decision, operating_cash_flow, net_income, None, None,
            "INVALID_TTM_NET_INCOME", net_income.status,
        )
    if operating_cash_flow.status != "VALID":
        return PITCashRealizationResult(
            issuer, decision, operating_cash_flow, net_income, None, None,
            "INVALID_TTM_OCF", operating_cash_flow.status,
        )
    if net_income.value is None or net_income.value <= 0:
        return PITCashRealizationResult(
            issuer, decision, operating_cash_flow, net_income, None, None,
            "NONPOSITIVE_NET_INCOME", "Cash Realization is defined only for positive PIT TTM Net Income",
        )
    raw_ratio = operating_cash_flow.value / net_income.value
    return PITCashRealizationResult(
        issuer, decision, operating_cash_flow, net_income, raw_ratio, min(raw_ratio, 1.0), "VALID",
        "upper-capped PIT OCF TTM divided by positive PIT Net Income TTM",
    )

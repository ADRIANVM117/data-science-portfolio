"""Eligibility only: no signals, ranks, allocations, or recommendations."""
from __future__ import annotations


def etf_eligible(record: dict[str, str]) -> tuple[bool, str]:
    if record.get("etf_structure_state") != "CONFIRMED_DIRECT_UNLEVERAGED": return False, "ETF_STRUCTURE_REVIEW"
    if record.get("direction") not in {"DIRECT", "BULL"} or record.get("leverage_multiple") != "1.0": return False, "ETF_NOT_DIRECT_UNLEVERAGED"
    return True, "ELIGIBLE"


def eligible(record: dict[str, str], coverage_ok: bool) -> tuple[bool, str]:
    if record.get("identity_verification_state") != "VERIFIED": return False, "IDENTITY_REVIEW"
    if record.get("listing_currency") != "USD" or not record.get("listing_market"): return False, "MARKET_OR_CURRENCY_REVIEW"
    if not coverage_ok: return False, "COVERAGE_OR_QUALITY_FAILURE"
    return etf_eligible(record) if record.get("asset_class") == "etf" else (True, "ELIGIBLE")

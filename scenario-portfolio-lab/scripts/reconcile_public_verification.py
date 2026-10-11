"""Reconcile a closed Alpha Vantage verification run with public SEC evidence.

This script never calls Alpha Vantage and never reads `.env`.  It records the
official SEC issuer/exchange dataset used for equity identity corroboration,
then creates a point-in-time, one-row-per-original-record reconciliation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests


SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_USER_AGENT = "scenario-portfolio-lab public-catalog research contact research@example.invalid"


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def latest_provider_events(path: Path) -> tuple[dict[str, dict], int]:
    latest: dict[str, dict] = {}; count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line); latest[event["instrument_id"]] = event; count += 1
    return latest, count


def fetch_sec(destination: Path) -> tuple[dict[str, list], str, str]:
    response = requests.get(SEC_URL, headers={"User-Agent": SEC_USER_AGENT}, timeout=(8, 30))
    response.raise_for_status()
    payload = response.content
    destination.write_bytes(payload)
    checked = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {row[2].upper(): row for row in response.json()["data"] if len(row) >= 4}, hashlib.sha256(payload).hexdigest(), checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("config/universe.csv"))
    parser.add_argument("--provider-events", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--apply-catalog", action="store_true")
    args = parser.parse_args()

    rows, fields = read_csv(args.catalog)
    latest, event_count = latest_provider_events(args.provider_events)
    evidence_dir = args.evidence_dir; evidence_dir.mkdir(parents=True, exist_ok=True)
    sec, sec_hash, checked = fetch_sec(evidence_dir / "sec_company_tickers_exchange.json")

    provider_summary = []
    reconciliation = []
    confirmed_equity_ids: set[str] = set()
    for instrument_id, event in sorted(latest.items()):
        match = event.get("match") or {}
        provider_summary.append({
            "instrument_id": instrument_id,
            "provider_status": event.get("provider_status", ""),
            "http_status": str(event.get("http_status", "")),
            "classification": event.get("classification", ""),
            "provider_symbol": match.get("provider_symbol", ""),
            "provider_name": match.get("provider_name", ""),
            "provider_type": match.get("provider_type", ""),
            "provider_region": match.get("provider_region", ""),
            "provider_currency": match.get("provider_currency", ""),
            "match_score": str(match.get("match_score", "")),
        })
    write_csv(evidence_dir / "provider_symbol_consolidated.csv", provider_summary, list(provider_summary[0]))

    for row in rows:
        event = latest.get(row["instrument_id"]); match = (event or {}).get("match") or {}
        provider_symbol = match.get("provider_symbol", "")
        sec_row = sec.get(provider_symbol.upper()) if provider_symbol else None
        state = "REVIEW"; reason = ""
        source = ""; identity = ""; market = ""; currency = ""
        # An ADR is a different legal instrument and requires the user's explicit decision.
        if row["mapping_type"] == "ADR_PROXY":
            reason = "PROPOSED_ADR_OR_DIFFERENT_SHARE_CLASS_REQUIRES_EXPLICIT_DECISION"
        elif row["asset_class"] == "equity" and event and sec_row and match.get("provider_type") == "Equity" and match.get("provider_currency") == "USD":
            state = "CONFIRMED"; confirmed_equity_ids.add(row["instrument_id"])
            source = SEC_URL
            identity = f"SEC ticker {provider_symbol} corroborates provider identity: {sec_row[1]}"
            market = f"SEC exchange: {sec_row[3]}"
            currency = "USD reported in the captured Alpha Vantage symbol response"
            reason = "SEC_IDENTITY_AND_EXCHANGE_PLUS_PROVIDER_EXACT_USD_SYMBOL"
        elif row["asset_class"] == "etf" and row["public_evidence_status"] == "CONFIRMED":
            state = "CONFIRMED"; reason = "OFFICIAL_ISSUER_EVIDENCE_PLUS_PROVIDER_EXACT_USD_SYMBOL"
        elif row["asset_class"] == "etf" and row["public_evidence_status"] == "CONFIRMED_STRUCTURAL_EXCLUSION":
            state = "EXCLUDED"; reason = "OFFICIAL_ISSUER_EVIDENCE_CONFIRMS_LEVERAGED_OR_INVERSE_ETF"
        elif not row["provider_symbol_candidate"]:
            reason = "NO_CANDIDATE_MAPPING; LOCAL_OR_AMBIGUOUS_INSTRUMENT_WAS_NOT_SUBSTITUTED_WITH_ADR_OR_OTHER_CLASS"
        elif row["mapping_type"] == "SYMBOL_COLLISION":
            reason = "SYMBOL_COLLISION_REQUIRES_INSTRUMENT_IDENTITY_EVIDENCE"
        elif row["asset_class"] == "etf":
            reason = "OFFICIAL_ISSUER_STRUCTURE_AND_USD_LISTING_EVIDENCE_NOT_YET_SUFFICIENT; PROVIDER_MATCH_ALONE_IS_NOT_PROMOTION"
        else:
            reason = "PRIMARY_IDENTITY_OR_LISTING_EVIDENCE_INSUFFICIENT_FOR_THIS_MAPPING"
        reconciliation.append({
            "instrument_id": row["instrument_id"], "reference_symbol": row["reference_symbol"],
            "asset_class": row["asset_class"], "provider_symbol_candidate": row["provider_symbol_candidate"],
            "mapping_type": row["mapping_type"], "final_verification_state": state,
            "state_reason": reason, "public_source_url": source,
            "public_checked_at_utc": checked if source else "",
            "identity_evidence": identity, "market_evidence": market, "usd_evidence": currency,
            "provider_status": (event or {}).get("provider_status", "NOT_QUERIED"),
            "provider_symbol": provider_symbol, "provider_type": match.get("provider_type", ""),
            "provider_currency": match.get("provider_currency", ""),
        })
    write_csv(evidence_dir / "catalog_reconciliation.csv", reconciliation, list(reconciliation[0]))
    summary = {
        "created_at_utc": checked, "catalog_records": len(rows), "provider_event_count": event_count,
        "provider_unique_instruments": len(latest), "provider_duplicate_events": event_count - len(latest),
        "sec_source_url": SEC_URL, "sec_sha256": sec_hash,
        "states_by_asset_class": {f"{asset}:{state}": n for (asset, state), n in sorted(Counter((r["asset_class"], r["final_verification_state"]) for r in reconciliation).items())},
        "rules": {"confirmed_equity": "SEC ticker/exchange + exact captured provider Equity/USD match, excluding ADR proxies", "confirmed_etf": "existing official issuer evidence + exact captured provider ETF/USD match", "review": "not silently substituted or promoted", "excluded": "official structural ETF exclusion"},
    }
    (evidence_dir / "reconciliation_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.apply_catalog:
        by_id = {r["instrument_id"]: r for r in reconciliation}
        for row in rows:
            result = by_id[row["instrument_id"]]
            if result["final_verification_state"] == "CONFIRMED":
                row["provider_mapping_state"] = "VERIFIED"
                row["identity_verification_state"] = "VERIFIED"
                row["listing_currency"] = "USD"
                row["public_evidence_status"] = "CONFIRMED"
                row["public_checked_at_utc"] = result["public_checked_at_utc"] or row["public_checked_at_utc"]
                row["public_verification_conclusion"] = result["state_reason"]
                if row["asset_class"] == "equity":
                    row["listing_market"] = result["market_evidence"].replace("SEC exchange: ", "")
                    row["identity_source_url"] = SEC_URL
                    row["listing_source_url"] = SEC_URL
            elif result["final_verification_state"] == "EXCLUDED":
                row["public_verification_conclusion"] = result["state_reason"]
            else:
                row["review_reason"] = result["state_reason"]
                row["public_verification_conclusion"] = result["state_reason"]
        write_csv(args.catalog, rows, fields)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Seal the approved entry universe and materialize it from immutable raw files.

No provider calls, credentials, or network access are used by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import exchange_calendars as xcals

from scenario_portfolio_lab.catalog import read_catalog
from scenario_portfolio_lab.ingest import parse_daily_adjusted
from scenario_portfolio_lab.pipeline import build_from_raw


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["symbol"])
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=Path, default=Path("config/universe.csv"))
    p.add_argument("--capture-manifest", type=Path, default=Path("docs/capture_manifest_consolidated_2026-10-09.json"))
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.snapshot.exists() or args.output.exists():
        raise SystemExit("snapshot and output must both be new directories")
    rows = read_catalog(args.catalog)
    included = [r for r in rows if r["public_evidence_status"] == "CONFIRMED"]
    if len(included) != 138:
        raise SystemExit(f"expected 138 approved entry records, found {len(included)}")
    manifest = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    by_symbol = {r["provider_symbol_candidate"].upper(): r for r in included}
    entries = {}
    for entry in manifest["entries"]:
        instrument_id = entry.get("instrument_id") or by_symbol.get(entry["symbol"].upper(), {}).get("instrument_id")
        if not instrument_id:
            raise SystemExit(f"capture manifest cannot be reconciled to catalog: {entry.get('symbol')}")
        entries[instrument_id] = {**entry, "instrument_id": instrument_id}
    if set(r["instrument_id"] for r in included) != set(entries):
        raise SystemExit("entry catalog and capture manifest do not identify the same instruments")
    raw_payloads: dict[str, bytes] = {}; records: dict[str, dict[str, str]] = {}
    sessions = {}; calendar_metadata = {}; session_report = []
    for row in included:
        symbol = row["provider_symbol_candidate"].upper(); entry = entries[row["instrument_id"]]; raw_path = Path(entry["path"])
        raw = raw_path.read_bytes(); actual = hashlib.sha256(raw).hexdigest()
        if actual != entry["sha256"]: raise SystemExit(f"raw hash mismatch: {symbol}")
        # XNYS is the versioned common US equity session calendar. exchange_calendars
        # resolves XNAS to the same session schedule in this supported release.
        cal = xcals.get_calendar("XNYS")
        # Use the calendar's versioned session index directly.  This avoids the
        # pandas-3 parsing regression in sessions_in_range observed in this venv.
        expected = {stamp.date() for stamp in cal.sessions if date(2018, 1, 2) <= stamp.date() <= date(2026, 10, 9)}
        frame = parse_daily_adjusted(json.loads(raw.decode("utf-8")), symbol)
        observed = set(frame["session_date"].to_list())
        raw_payloads[symbol] = raw; records[symbol] = row; sessions[symbol] = expected
        calendar_metadata[symbol] = {"source": "exchange_calendars:US_EQUITY_COMMON(XNYS=XNAS=ARCX)", "version": xcals.__version__, "market_recorded": row["listing_market"]}
        session_report.append({"instrument_id":row["instrument_id"],"symbol":symbol,"asset_class":row["asset_class"],"expected_sessions":len(expected),"observed_sessions":len(observed),"missing_expected_sessions":len(expected-observed),"extra_dates":len(observed-expected),"duplicate_dates":frame.height-frame["session_date"].n_unique(),"zero_volume_rows":frame.filter(frame["volume"]==0).height,"null_volume_rows":frame["volume"].null_count(),"dividend_rows":frame.filter(frame["dividend_amount"]>0).height,"split_rows":frame.filter(frame["split_coefficient"]!=1).height})
    args.snapshot.mkdir(parents=True)
    write_csv(args.snapshot / "entry_universe.csv", included)
    sealed = {"status":"SEALED_ENTRY_UNIVERSE","sealed_at_utc":datetime.now(timezone.utc).isoformat(),"decision":"Approved entry universe; not an eligible universe or economic portfolio selection.","period":{"start":"2018-01-02","end":"2026-10-09","frequency":"daily","currency":"USD"},"counts":{"included":138,"equities":sum(r["asset_class"]=="equity" for r in included),"etfs":sum(r["asset_class"]=="etf" for r in included),"structural_exclusions":14,"out_of_scope_pending":50},"entries":[{**entries[r["instrument_id"]],"provider_symbol":r["provider_symbol_candidate"],"listing_currency":r["listing_currency"],"evidence":{"identity":r["identity_source_url"],"listing":r["listing_source_url"],"etf_structure":r["etf_structure_source_url"]}} for r in included]}
    (args.snapshot / "entry_universe_manifest.json").write_text(json.dumps(sealed,indent=2,sort_keys=True),encoding="utf-8")
    write_csv(args.snapshot / "session_observation_report.csv", session_report)
    result = build_from_raw(raw_payloads=raw_payloads,records=records,cutoff="2026-10-09",output=args.output,expected_sessions=sessions,minimum_observations=2,calendar_metadata=calendar_metadata)
    print(json.dumps({"snapshot":str(args.snapshot),"output":str(args.output),"status":result["status"],"gold":sum(x["status"]=="GOLD_APPROVED" for x in result["quality"].values()),"eligible":sum(x["eligible"] for x in result["eligibility"].values())},sort_keys=True))

if __name__ == "__main__":
    main()

"""Capture independently confirmed instruments into a resumable raw batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from scenario_portfolio_lab.capture import ProviderRequestError, capture_plan, fetch_daily_adjusted
from scenario_portfolio_lab.catalog import read_catalog
from scenario_portfolio_lab.ingest import validate_cutoff, write_raw_once


def stable_hash(plan: list[dict[str, str]], cutoff: str) -> str:
    payload = [{"instrument_id": r["instrument_id"], "symbol": r["provider_symbol_candidate"], "asset_class": r["asset_class"]} for r in plan]
    return hashlib.sha256(json.dumps({"cutoff": cutoff, "plan": payload}, sort_keys=True).encode()).hexdigest()


def append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def prior_symbols(manifests: list[Path]) -> set[str]:
    symbols: set[str] = set()
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("status") != "COMPLETED":
            raise SystemExit(f"Prior manifest is not closed: {manifest}")
        symbols.update(entry["symbol"].upper() for entry in data.get("entries", []))
    return symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture verified daily raw responses")
    parser.add_argument("--catalog", type=Path, default=Path("config/universe.csv"))
    parser.add_argument("--cutoff", required=True, help="ISO date; must be a verified market session")
    parser.add_argument("--output", type=Path, required=True, help="A new batch directory")
    parser.add_argument("--prior-manifest", type=Path, action="append", default=[])
    parser.add_argument("--execute", action="store_true", help="Required to permit network requests")
    parser.add_argument("--resume", action="store_true", help="Resume this exact interrupted batch")
    parser.add_argument("--limit", type=int, help="Controlled-run cap; omitted captures the full plan")
    args = parser.parse_args(); validate_cutoff(args.cutoff)
    plan = capture_plan(read_catalog(args.catalog))
    already_closed = prior_symbols(args.prior_manifest)
    plan = [row for row in plan if row["provider_symbol_candidate"].upper() not in already_closed]
    if args.limit is not None: plan = plan[:args.limit]
    plan_hash = stable_hash(plan, args.cutoff)
    print(f"PLAN count={len(plan)} cutoff={args.cutoff} prior_closed={len(already_closed)}")
    if not args.execute or not plan: return
    manifest_path = args.output / "capture_manifest.json"; events_path = args.output / "capture_events.jsonl"; lock = args.output / ".capture.lock"
    args.output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if not args.resume: raise SystemExit("Batch directory already has a manifest; use --resume only for the same interrupted batch.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "RUNNING" or manifest.get("plan_sha256") != plan_hash: raise SystemExit("Batch is closed or incompatible with the current plan.")
    elif args.resume:
        raise SystemExit("No interrupted batch exists to resume.")
    else:
        manifest = {"status": "RUNNING", "cutoff": args.cutoff, "created_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_hash, "planned_symbols": [r["provider_symbol_candidate"].upper() for r in plan], "prior_closed_manifests": [str(p) for p in args.prior_manifest], "entries": [], "failures": []}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit("Capture batch is locked by another process; no request was sent.")
    try:
        done = set()
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("status") == "CAPTURED": done.add(json.loads(line)["symbol"])
        entries = {entry["symbol"]: entry for entry in manifest.get("entries", [])}; failures = {entry["symbol"]: entry for entry in manifest.get("failures", [])}
        for row in plan:
            symbol = row["provider_symbol_candidate"].upper()
            if symbol in done: continue
            started = time.monotonic(); event = {"instrument_id": row["instrument_id"], "symbol": symbol, "started_at_utc": datetime.now(timezone.utc).isoformat()}
            try:
                raw = fetch_daily_adjusted(symbol); artifact = write_raw_once(raw, args.output / "raw" / f"{symbol}.json")
                event.update({"status": "CAPTURED", "finished_at_utc": datetime.now(timezone.utc).isoformat(), **artifact}); entries[symbol] = {"instrument_id": row["instrument_id"], "asset_class": row["asset_class"], "symbol": symbol, **artifact}; failures.pop(symbol, None)
            except ProviderRequestError as exc:
                event.update({"status": "FAILED", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc)}); failures[symbol] = {"instrument_id": row["instrument_id"], "symbol": symbol, "error_type": type(exc).__name__, "error": str(exc)}
            append_event(events_path, event); manifest["entries"] = [entries[k] for k in sorted(entries)]; manifest["failures"] = [failures[k] for k in sorted(failures)]; manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(); manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            time.sleep(max(0.0, 12.0 - (time.monotonic() - started)))
        manifest["status"] = "COMPLETED" if len(entries) + len(failures) == len(plan) else "RUNNING"; manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat(); manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        lock.rmdir()


if __name__ == "__main__": main()

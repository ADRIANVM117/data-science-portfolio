"""Resumable serial Alpha Vantage verification; no automatic catalog promotion."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path
from scenario_portfolio_lab.capture import ProviderRequestError, search_symbol_with_telemetry
from scenario_portfolio_lab.catalog import read_catalog

def utc_now() -> str: return datetime.now(timezone.utc).isoformat()
def atomic_json(path: Path, value: dict) -> None:
    temp=path.with_suffix(".tmp"); temp.write_text(json.dumps(value,indent=2,sort_keys=True),encoding="utf-8"); os.replace(temp,path)
def append_event(path: Path, value: dict) -> None:
    with path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(value,sort_keys=True)+"\n"); handle.flush(); os.fsync(handle.fileno())
def events(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
def successful_ids(path: Path) -> set[str]: return {event["instrument_id"] for event in events(path) if event.get("provider_status") in {"MATCHED","NOT_FOUND"}}

def main() -> None:
    parser=argparse.ArgumentParser(description="Serial provider symbol verification")
    parser.add_argument("--catalog",type=Path,default=Path("config/universe.csv")); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true"); parser.add_argument("--min-interval-seconds",type=float,default=12.0); parser.add_argument("--limit",type=int,default=None)
    args=parser.parse_args(); catalog_bytes=args.catalog.read_bytes(); catalog_hash=hashlib.sha256(catalog_bytes).hexdigest(); rows=read_catalog(args.catalog)
    candidates=[row for row in rows if row["provider_symbol_candidate"].strip()]; print(f"CANDIDATE_RECORDS={len(candidates)}")
    if not args.execute: return
    if args.min_interval_seconds < 12: raise SystemExit("minimum interval is 12 seconds (5 requests/minute)")
    args.output.mkdir(parents=True,exist_ok=True); manifest_path=args.output/"provider_symbol_manifest.json"; events_path=args.output/"provider_symbol_evidence.jsonl"; lock_path=args.output/".verifier.lock"
    config={"catalog_sha256":catalog_hash,"candidate_records":len(candidates),"requests_serial":True,"minimum_interval_seconds":args.min_interval_seconds,"connect_timeout_seconds":8,"read_timeout_seconds":15,"max_attempts":3,"max_operation_seconds":45}
    if lock_path.exists(): raise SystemExit("verification output is locked by another verifier")
    lock_path.mkdir()
    try:
     if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if {key:manifest.get(key) for key in config} != config: raise SystemExit("incompatible existing verification state")
        if manifest.get("status") in {"INTERRUPTED_DUPLICATE_PROCESSES", "BLOCKED"}:
            manifest["status"]="RUNNING"; manifest.pop("blocked_by",None); manifest["resumed_at_utc"]=utc_now(); atomic_json(manifest_path,manifest)
     else:
        manifest={"status":"RUNNING","started_at_utc":utc_now(),**config,"completed":0,"failed":0,"report":"provider_symbol_evidence.jsonl"}; atomic_json(manifest_path,manifest)
     successful=successful_ids(events_path)
     selected=[row for row in candidates if row["instrument_id"] not in successful]
     if args.limit is not None: selected=selected[:args.limit]
     for row in selected:
        started_utc,started=utc_now(),time.monotonic(); event={"instrument_id":row["instrument_id"],"candidate":row["provider_symbol_candidate"],"started_at_utc":started_utc}
        try:
            matches,t=search_symbol_with_telemetry(row["provider_symbol_candidate"].strip()); match=next((item for item in matches if item["provider_symbol"].upper()==row["provider_symbol_candidate"].upper()),None)
            event.update({"provider_status":"MATCHED" if match else "NOT_FOUND","http_status":t.http_status,"stage":t.stage,"classification":t.classification,"attempts":t.attempts,"duration_seconds":round(t.duration_seconds,3),"match":match})
        except ProviderRequestError as error:
            t=error.telemetry; event.update({"provider_status":"ERROR","http_status":t.http_status,"stage":t.stage,"classification":t.classification,"attempts":t.attempts,"duration_seconds":round(t.duration_seconds,3)})
        event["finished_at_utc"]=utc_now(); append_event(events_path,event); all_events=events(events_path)
        manifest.update({"completed":len(successful_ids(events_path)),"failed":sum(item.get("provider_status")=="ERROR" for item in all_events),"updated_at_utc":utc_now()}); atomic_json(manifest_path,manifest)
        if event["provider_status"]=="ERROR" and event["classification"] in {"CONNECT_TIMEOUT","READ_TIMEOUT","CONNECTION_ERROR","PROXY_ERROR","TLS_ERROR","HTTP_CLIENT_ERROR","PROVIDER_THROTTLE_OR_INFO"}:
            manifest.update({"status":"BLOCKED","blocked_by":event["classification"]}); atomic_json(manifest_path,manifest); break
        time.sleep(max(0,args.min_interval_seconds-(time.monotonic()-started)))
     if manifest.get("status")=="RUNNING" and args.limit is None and manifest["completed"]==len(candidates): manifest.update({"status":"COMPLETED","finished_at_utc":utc_now()}); atomic_json(manifest_path,manifest)
    finally:
     lock_path.rmdir()

if __name__ == "__main__": main()

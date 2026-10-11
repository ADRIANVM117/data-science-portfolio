"""Offline-only materialization of the public data boundary from supplied raw bytes."""
from __future__ import annotations

import hashlib
import json
import csv
import os
import uuid
from datetime import date
from pathlib import Path

import polars as pl

from .eligibility import eligible
from .ingest import END_DATE, START_DATE, decode_raw, parse_daily_adjusted, validate_cutoff, write_raw_once
from .matrices import coverage, price_matrix, simple_returns
from .quality import assess_silver, write_parquet_atomic


def _write_csv_atomic(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["symbol", "status"]
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp.csv")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        with temporary.open(encoding="utf-8", newline="") as handle:
            if csv.DictReader(handle).fieldnames != fields: raise IOError("CSV reread mismatch")
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def build_from_raw(*, raw_payloads: dict[str, bytes], records: dict[str, dict[str, str]], cutoff: str, output: Path, expected_sessions: dict[str, set[date]], minimum_observations: int = 2, calendar_metadata: dict[str, dict[str, str]] | None = None) -> dict:
    """Build raw → Silver → Gold → matrices → eligibility without network access."""
    if output.exists(): raise FileExistsError("run output must be a new dated directory")
    cutoff_date = validate_cutoff(cutoff)
    if set(raw_payloads) != set(records): raise ValueError("raw payloads and records must have identical symbols")
    if set(raw_payloads) - set(expected_sessions): raise ValueError("missing instrument calendar")
    if calendar_metadata is None or set(raw_payloads) - set(calendar_metadata):
        raise ValueError("calendar source and version are required per instrument")
    for symbol in raw_payloads:
        metadata = calendar_metadata[symbol]
        if not metadata.get("source") or not metadata.get("version"):
            raise ValueError("calendar metadata must contain source and version")
    gold: dict[str, pl.DataFrame] = {}; manifest = {"status":"PARTIAL", "cutoff":cutoff, "approved_period":{"start":str(START_DATE),"end":str(END_DATE)}, "raw":{}, "quality":{}, "eligibility":{}, "calendars":calendar_metadata}
    try:
        for symbol, raw in raw_payloads.items():
            record = records[symbol]
            raw_meta = write_raw_once(raw, output / "raw" / f"{symbol}.json"); manifest["raw"][symbol] = raw_meta
            silver = parse_daily_adjusted(decode_raw(raw), symbol)
            write_parquet_atomic(silver, output / "silver" / record["asset_class"] / f"{symbol}.parquet")
            result = assess_silver(silver, cutoff); manifest["quality"][symbol] = {"status":result.status,"warnings":list(result.warnings)}
            if result.status == "GOLD_APPROVED":
                gold[symbol] = silver; write_parquet_atomic(silver, output / "gold" / "series" / record["asset_class"] / f"{symbol}.parquet")
        prices = price_matrix(gold); returns = simple_returns(prices, expected_sessions)
        write_parquet_atomic(prices, output / "gold" / "adjusted_close_matrix.parquet")
        write_parquet_atomic(returns, output / "gold" / "simple_total_return_matrix.parquet")
        coverage_rows = []
        eligibility_rows = []
        for item in coverage(prices, cutoff, minimum_observations, expected_sessions):
            record = records[item.symbol]; ok, reason = eligible(record, item.eligible)
            manifest["eligibility"][item.symbol] = {"eligible":ok,"reason":reason,"coverage_reason":item.reason,"observations":item.observations}
            coverage_rows.append({"symbol": item.symbol, "observations": item.observations, "leading_nulls": item.leading_nulls, "interior_nulls": item.interior_nulls, "trailing_nulls": item.trailing_nulls, "first_observation": item.first_observation, "last_observation": item.last_observation, "leading_gap_interpretation": "NONE" if item.leading_nulls == 0 else "UNVERIFIED_CAUSE_NOT_AUTOMATICALLY_PRE_LISTING", "coverage_status": item.reason})
            eligibility_rows.append({"symbol": item.symbol, "asset_class": record["asset_class"], "universe_status": "ELIGIBLE" if ok else "REVIEW_OR_EXCLUDED", "reason": reason, "coverage_reason": item.reason})
        quality_rows = [{"symbol": symbol, "status": value["status"], "warnings": "|".join(value["warnings"])} for symbol, value in manifest["quality"].items()]
        _write_csv_atomic(quality_rows, output / "reports" / "quality_report.csv")
        _write_csv_atomic(coverage_rows, output / "reports" / "coverage_report.csv")
        _write_csv_atomic(eligibility_rows, output / "reports" / "eligible_universe.csv")
        eligible_symbols = [row["symbol"] for row in eligibility_rows if row["universe_status"] == "ELIGIBLE"]
        write_parquet_atomic(returns.select(["session_date", *eligible_symbols]), output / "gold" / "eligible_simple_total_return_matrix.parquet")
        manifest["status"] = "COMPLETED"
    finally:
        output.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(manifest, indent=2, sort_keys=True).encode()
        (output / "run_manifest.json").write_bytes(encoded)
        manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return manifest

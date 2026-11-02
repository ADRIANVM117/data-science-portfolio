"""Build the offline VIX state-alignment audit for the sealed return matrix.

This script does not call Alpha Vantage, load credentials, alter sealed inputs, or
construct probabilities, states, simulations, or FFR.  It maps each usable return
date x to the VIX close on the *published equity-calendar* session x-2.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "data" / "published" / "cut_2026-10-09_run_2026-10-11_full_window_final"
PRICE_MATRIX = RUN_ROOT / "gold" / "adjusted_close_matrix.parquet"
RETURN_MATRIX = RUN_ROOT / "gold" / "eligible_simple_total_return_matrix.parquet"
VIX_ROOT = ROOT / "data" / "raw" / "vix_index_access_2026-10-29"
VIX_RAW = VIX_ROOT / "raw" / "VIX_INDEX_DATA_daily.json"
VIX_MANIFEST = VIX_ROOT / "manifest.json"
OUTPUT_ROOT = ROOT / "data" / "published" / "vix_state_alignment_2026-10-31"
WINDOW_START = date(2018, 1, 2)
WINDOW_END = date(2026, 10, 9)

SUPPORT_EVIDENCE = (
    "It's the official VIX closing value from CBOE (we are CBOE-licensed).\n"
    "The value is normally available by 10PM US/Eastern.\n"
    "We are not aware of any revisions from CBOE in the past 2 years."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_vix_closes() -> tuple[dict[date, float], dict[str, object], str]:
    manifest = json.loads(VIX_MANIFEST.read_text(encoding="utf-8"))
    raw_hash = sha256(VIX_RAW)
    expected_hash = manifest.get("raw_sha256")
    if not isinstance(expected_hash, str) or raw_hash != expected_hash:
        raise ValueError("VIX raw SHA-256 does not match its capture manifest")

    payload = json.loads(VIX_RAW.read_text(encoding="utf-8"))
    if payload.get("symbol") != "VIX" or payload.get("interval") != "daily":
        raise ValueError("VIX raw identity or interval does not match the captured INDEX_DATA contract")
    records = payload.get("data")
    if not isinstance(records, list):
        raise ValueError("VIX raw data array is absent")

    closes: dict[date, float] = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("VIX raw contains a non-object observation")
        stamp = date.fromisoformat(str(item["date"]))
        close = float(item["close"])
        if stamp in closes:
            raise ValueError("VIX raw contains duplicate dates")
        if not math.isfinite(close) or close <= 0:
            raise ValueError("VIX raw contains a non-finite or non-positive close")
        closes[stamp] = close
    return closes, manifest, raw_hash


def build() -> dict[str, object]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {relative(OUTPUT_ROOT)}")

    vix_closes, vix_manifest, vix_raw_hash = load_vix_closes()
    prices = pl.read_parquet(PRICE_MATRIX)
    returns = pl.read_parquet(RETURN_MATRIX)
    if "session_date" not in prices.columns or "session_date" not in returns.columns:
        raise ValueError("published price or return matrix lacks session_date")

    price_dates = prices.get_column("session_date").to_list()
    return_dates_all = returns.get_column("session_date").to_list()
    symbol_columns = [column for column in returns.columns if column != "session_date"]
    if len(price_dates) != 2_205 or len(return_dates_all) != 2_205 or len(symbol_columns) != 125:
        raise ValueError("sealed matrix dimensions differ from the 2,205-price / 2,204-return contract")
    if not set(symbol_columns).issubset(set(prices.columns)):
        raise ValueError("eligible return symbols are not a subset of published price symbols")
    if price_dates != return_dates_all or price_dates != sorted(price_dates) or len(set(price_dates)) != len(price_dates):
        raise ValueError("published price calendar is not identical, unique, and increasing across matrices")
    if price_dates[0] != WINDOW_START or price_dates[-1] != WINDOW_END:
        raise ValueError("published price calendar does not span the approved cut")

    first_return_nulls = returns.head(1).select(pl.exclude("session_date").null_count()).row(0)
    subsequent_nulls = returns.slice(1).select(pl.exclude("session_date").null_count()).row(0)
    if any(count != 1 for count in first_return_nulls) or any(count != 0 for count in subsequent_nulls):
        raise ValueError("return matrix does not have exactly one initial null per instrument")
    usable_returns_are_finite = all(
        bool(returns.get_column(symbol).slice(1).is_finite().all()) for symbol in symbol_columns
    )
    if not usable_returns_are_finite:
        raise ValueError("usable rows of the sealed return matrix contain a non-finite value")

    rows: list[dict[str, object]] = []
    for calendar_position in range(1, len(price_dates)):
        return_date = price_dates[calendar_position]
        return_start_date = price_dates[calendar_position - 1]
        if calendar_position < 2:
            rows.append(
                {
                    "return_row_id": calendar_position - 1,
                    "return_date": return_date,
                    "return_start_date": return_start_date,
                    "vix_source_date": None,
                    "vix_close": None,
                    "alignment_status": "UNALIGNABLE_PRE_PUBLISHED_CALENDAR_START",
                }
            )
            continue
        vix_source_date = price_dates[calendar_position - 2]
        vix_close = vix_closes.get(vix_source_date)
        status = "ALIGNED" if vix_close is not None else "VIX_CLOSE_MISSING_FOR_PUBLISHED_SOURCE_SESSION"
        rows.append(
            {
                "return_row_id": calendar_position - 1,
                "return_date": return_date,
                "return_start_date": return_start_date,
                "vix_source_date": vix_source_date,
                "vix_close": vix_close,
                "alignment_status": status,
            }
        )

    audit = pl.DataFrame(rows).with_columns(
        pl.col("return_date").cast(pl.Date),
        pl.col("return_start_date").cast(pl.Date),
        pl.col("vix_source_date").cast(pl.Date),
        pl.col("vix_close").cast(pl.Float64),
    )
    aligned = audit.filter(pl.col("alignment_status") == "ALIGNED")
    source_dates = aligned.get_column("vix_source_date").to_list()
    alignment_checks = {
        "return_row_count_is_2204": audit.height == 2_204,
        "return_dates_match_usable_return_matrix": audit.get_column("return_date").to_list() == price_dates[1:],
        "return_dates_unique_and_increasing": audit.get_column("return_date").n_unique() == audit.height and audit.get_column("return_date").is_sorted(),
        "all_aligned_dates_follow_published_calendar_positions": all(
            row["return_start_date"] == price_dates[int(row["return_row_id"])]
            and row["vix_source_date"] == price_dates[int(row["return_row_id"]) - 1]
            for row in aligned.to_dicts()
        ),
        "aligned_date_order_is_strict": all(
            row["vix_source_date"] < row["return_start_date"] < row["return_date"] for row in aligned.to_dicts()
        ),
        "aligned_vix_matches_raw_exact_date": all(vix_closes[row["vix_source_date"]] == row["vix_close"] for row in aligned.to_dicts()),
        "aligned_vix_closes_are_finite_and_positive": all(math.isfinite(value) and value > 0 for value in aligned.get_column("vix_close").to_list()),
        "no_used_vix_date_after_cutoff": all(stamp <= WINDOW_END for stamp in source_dates),
        "only_published_calendar_dates_are_used": set(source_dates).issubset(set(price_dates)),
        "first_return_is_retained_and_marked_unalignable": audit.row(0, named=True)["alignment_status"] == "UNALIGNABLE_PRE_PUBLISHED_CALENDAR_START",
        "input_vix_hash_matches_capture_manifest": vix_raw_hash == vix_manifest["raw_sha256"],
        "usable_return_matrix_values_are_finite": usable_returns_are_finite,
    }
    if not all(alignment_checks.values()):
        failures = [name for name, passed in alignment_checks.items() if not passed]
        raise AssertionError(f"alignment controls failed: {failures}")

    OUTPUT_ROOT.mkdir(parents=True)
    table_path = OUTPUT_ROOT / "vix_return_alignment.parquet"
    report_path = OUTPUT_ROOT / "control_report.json"
    method_path = OUTPUT_ROOT / "methodology.json"
    manifest_path = OUTPUT_ROOT / "alignment_manifest.json"
    audit.write_parquet(table_path)

    status_counts = {row["alignment_status"]: row["len"] for row in audit.group_by("alignment_status").len().to_dicts()}
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "controls": alignment_checks,
        "all_controls_passed": all(alignment_checks.values()),
        "counts": {"audit_rows": audit.height, "aligned_rows": aligned.height, "not_aligned_rows": audit.height - aligned.height, "status_counts": status_counts},
        "first_five_rows": [
            {key: value.isoformat() if isinstance(value, date) else value for key, value in row.items()} for row in audit.head(5).to_dicts()
        ],
        "last_five_rows": [
            {key: value.isoformat() if isinstance(value, date) else value for key, value in row.items()} for row in audit.tail(5).to_dicts()
        ],
        "not_aligned_rows": [
            {key: value.isoformat() if isinstance(value, date) else value for key, value in row.items()} for row in audit.filter(pl.col("alignment_status") != "ALIGNED").to_dicts()
        ],
        "vix_only_dates_within_approved_window_not_used": len(
            {stamp for stamp in vix_closes if WINDOW_START <= stamp <= WINDOW_END} - set(price_dates)
        ),
        "published_calendar_dates_not_used_as_vix_source": len(set(price_dates) - set(source_dates)),
    }
    methodology = {
        "purpose": "offline audit alignment between the sealed 2,204-row log-return sample and VIX daily closes; not FFR, state construction, probabilities, or simulation",
        "return_definition": "A return dated x runs from the close of published equity-calendar session x-1 to the close of session x.",
        "alignment_rule": "For each return date x, use VIX close dated at published equity-calendar session x-2. Session offsets are computed only on the complete sealed price calendar.",
        "first_return_policy": "Retain the first return audit row. If x-2 precedes the available published calendar, set vix_source_date and vix_close null and status UNALIGNABLE_PRE_PUBLISHED_CALENDAR_START; do not extend the calendar or remove the row.",
        "join_rule": "Exact-date lookup into captured VIX raw data. No VIX-row shift, forward fill, backfill, interpolation, or use of VIX-only calendar dates.",
        "availability_convention": "The two-session lag incorporates a publication margin. It is not point-in-time verification and does not eliminate possible historical revisions.",
        "user_provided_alpha_vantage_support_evidence_verbatim": SUPPORT_EVIDENCE,
        "evidence_limitations": "No date of the support message, SLA, historical availability guarantee, or guarantee against future/past revisions is asserted.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    method_path.write_text(json.dumps(methodology, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "published-equity-calendar positional lag of two sessions; exact VIX-date lookup",
        "inputs": {
            "price_calendar": {"path": relative(PRICE_MATRIX), "sha256_computed_now": sha256(PRICE_MATRIX), "historical_reference_hash_available": False},
            "simple_return_matrix": {"path": relative(RETURN_MATRIX), "sha256_computed_now": sha256(RETURN_MATRIX), "historical_reference_hash_available": False},
            "vix_raw": {"path": relative(VIX_RAW), "sha256_capture_manifest_reference": vix_manifest["raw_sha256"], "sha256_verified_now": vix_raw_hash, "reference_matches": True},
        },
        "outputs": {
            "alignment_table": {"path": relative(table_path), "sha256": sha256(table_path)},
            "control_report": {"path": relative(report_path), "sha256": sha256(report_path)},
            "methodology": {"path": relative(method_path), "sha256": sha256(method_path)},
        },
        "counts": report["counts"],
        "scope": "No modification of sealed inputs; no probabilities, states, simulations, FFR, notebooks, or pipeline execution.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"output_root": relative(OUTPUT_ROOT), "report": report, "manifest": manifest}


if __name__ == "__main__":
    result = build()
    print(json.dumps({"output_root": result["output_root"], "counts": result["report"]["counts"], "controls_passed": result["report"]["all_controls_passed"]}, sort_keys=True))

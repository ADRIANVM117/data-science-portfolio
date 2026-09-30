"""Read-only structural inspection for preserved Stage 0B SEC evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta, timezone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "stage_0b_sec"
FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
FIXED_US_HOLIDAYS = {(1, 1), (7, 4), (12, 25)}  # Deliberately not a trading calendar.
CANDIDATE_TAGS = {
    "Assets", "Liabilities", "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
    "NetIncomeLoss", "OperatingIncomeLoss", "GrossProfit",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def acceptance_to_eastern(value: str) -> datetime:
    """Normalize only for clock-time diagnostics; raw timestamp is never replaced."""
    if "T" in value:
        utc_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        year = utc_value.year
        march_first = datetime(year, 3, 1)
        dst_start_day = 1 + (6 - march_first.weekday()) % 7 + 7  # second Sunday in March
        november_first = datetime(year, 11, 1)
        dst_end_day = 1 + (6 - november_first.weekday()) % 7  # first Sunday in November
        dst_start_utc = datetime(year, 3, dst_start_day, 7, tzinfo=timezone.utc)
        dst_end_utc = datetime(year, 11, dst_end_day, 6, tzinfo=timezone.utc)
        offset = -4 if dst_start_utc <= utc_value < dst_end_utc else -5
        return (utc_value + timedelta(hours=offset)).replace(tzinfo=None)
    return datetime.strptime(value, "%Y%m%d%H%M%S")


def main() -> int:
    manifest = load_json(RAW_ROOT / "manifest.json")
    successful_all = [entry for entry in manifest if entry["request_status"] == "HTTP_200"]
    successful_by_hash = {entry["content_sha256"]: entry for entry in successful_all}
    successful = list(successful_by_hash.values())
    print(f"successful_raw_records={len(successful_all)}")
    print(f"unique_successful_payloads={len(successful)}")
    print("endpoint_counts=" + repr(dict(Counter(item["endpoint"] for item in successful))))

    resolved_ciks = {
        entry["issuer"]: entry["resolved_cik"]
        for entry in manifest
        if entry["endpoint"] == "identity_resolution"
    }
    submissions_by_issuer: dict[str, list[dict]] = {}
    for entry in successful:
        if entry["endpoint"] not in {"submissions", "submissions_history"}:
            continue
        payload = load_json(PROJECT_ROOT / entry["local_raw_file"])
        resolved_cik = resolved_ciks.get(entry["issuer"])
        if resolved_cik:
            payload_cik = str(payload.get("cik", "")).zfill(10)
            path_matches_cik = resolved_cik in entry["local_raw_file"]
            if entry["endpoint"] == "submissions" and payload_cik != resolved_cik:
                continue
            if entry["endpoint"] == "submissions_history" and not path_matches_cik:
                continue
        inventory = payload.get("filings", {}).get("recent", payload)
        rows = [dict(zip(inventory, values)) for values in zip(*inventory.values())]
        submissions_by_issuer.setdefault(entry["issuer"], []).extend(rows)

    for issuer, rows in sorted(submissions_by_issuer.items()):
        relevant = [row for row in rows if row.get("form") in FORMS]
        amendments = [row for row in relevant if row["form"].endswith("/A")]
        missing_acceptance = sum(not row.get("acceptanceDateTime") for row in relevant)
        period_counts = Counter((row.get("reportDate"), row["form"].removesuffix("/A")) for row in relevant)
        competing_periods = sum(count > 1 for count in period_counts.values())
        accepted = [acceptance_to_eastern(row["acceptanceDateTime"]) for row in relevant]
        pre_open = sum(item.time().hour < 9 or (item.time().hour == 9 and item.time().minute < 30) for item in accepted)
        regular_hours = sum(
            (item.time().hour > 9 or (item.time().hour == 9 and item.time().minute >= 30))
            and item.time().hour < 16
            for item in accepted
        )
        after_close = sum(item.time().hour >= 16 for item in accepted)
        weekends = sum(item.weekday() >= 5 for item in accepted)
        fixed_holidays = sum((item.month, item.day) in FIXED_US_HOLIDAYS for item in accepted)
        print(
            f"{issuer}: filings_in_scope={len(relevant)} amendments={len(amendments)} "
            f"missing_acceptance={missing_acceptance} competing_periods={competing_periods} "
            f"pre_open={pre_open} regular_hours={regular_hours} after_close={after_close} "
            f"weekends={weekends} fixed_us_holidays={fixed_holidays}"
        )

    for entry in successful:
        if entry["endpoint"] != "companyfacts":
            continue
        payload = load_json(PROJECT_ROOT / entry["local_raw_file"])
        resolved_cik = resolved_ciks.get(entry["issuer"])
        if resolved_cik and str(payload.get("cik", "")).zfill(10) != resolved_cik:
            continue
        facts = payload.get("facts", {}).get("us-gaap", {})
        present = sorted(CANDIDATE_TAGS.intersection(facts))
        fact_rows = [
            (tag, unit, row)
            for tag in present
            for unit, unit_rows in facts[tag].get("units", {}).items()
            for row in unit_rows
            if row.get("form") in FORMS
        ]
        provenance = Counter((row.get("accn"), row.get("filed"), row.get("form")) for _, _, row in fact_rows)
        submission_accessions = {row.get("accessionNumber") for row in submissions_by_issuer.get(entry["issuer"], [])}
        missing_submission_link = sum(row.get("accn") not in submission_accessions for _, _, row in fact_rows)
        duplicate_periods = Counter(
            (tag, unit, row.get("start"), row.get("end"), row.get("fy"), row.get("fp"))
            for tag, unit, row in fact_rows
        )
        print(
            f"{entry['issuer']}: tags_present={len(present)} facts_in_scope={len(fact_rows)} "
            f"accession_provenance={len(provenance)} duplicate_period_keys="
            f"{sum(count > 1 for count in duplicate_periods.values())} "
            f"facts_without_submission_link={missing_submission_link}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

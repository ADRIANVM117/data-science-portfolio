"""Read-only lower-bound contamination summary for preserved LISTING_STATUS CSVs."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "listing_status_audit"
CONFIRMED_NON_COMMON_PATTERNS = {
    "warrant": r"\bwarrants?\b",
    "unit": r"\bunits\b|\bunit\s+(?:\d|\()",
    "right": r"\bcontingent value rights?\b|\btradeable rights?\b",
    "preferred": r"\bpreferred\b",
    "depositary": r"\bdepositary\b",
    "adr_ads": r"\badr\b|\bads\b|american depositary",
    "etf_etn": r"\betf\b|\betn\b",
    "closed_end_fund": r"closed[ -]end fund",
}
UNKNOWN_FLAG_PATTERNS = {
    "spac_like": r"acquisition corp|acquisition company|blank check",
    "fund_like": r"\bfund\b",
    "trust_like": r"\btrust\b",
    "foreign_labelled": r"\bforeign\b",
}


def matching_labels(name: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items() if re.search(pattern, name, flags=re.IGNORECASE)]


def main() -> None:
    manifest = json.loads((RAW_ROOT / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in manifest
        if entry["request_status"] == "HTTP_200" and entry["parameters"]["state"] == "active"
    ]
    results: list[dict[str, object]] = []
    examples: dict[str, list[dict[str, str]]] = {
        **{key: [] for key in CONFIRMED_NON_COMMON_PATTERNS},
        **{key: [] for key in UNKNOWN_FLAG_PATTERNS},
    }
    for entry in sorted(entries, key=lambda item: item["parameters"]["date"]):
        with (PROJECT_ROOT / entry["local_raw_file"]).open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        stocks = [row for row in rows if row["assetType"] == "Stock"]
        symbol_counts = Counter(row["symbol"] for row in rows)
        full_row_counts = Counter(
            tuple(row[column] for column in ("symbol", "name", "exchange", "assetType", "ipoDate", "delistingDate", "status"))
            for row in rows
        )
        confirmed = 0
        unknown_flag_counts: Counter[str] = Counter()
        confirmed_label_counts: Counter[str] = Counter()
        for row in stocks:
            labels = matching_labels(row["name"], CONFIRMED_NON_COMMON_PATTERNS)
            unknown_labels = matching_labels(row["name"], UNKNOWN_FLAG_PATTERNS)
            if labels:
                confirmed += 1
                confirmed_label_counts.update(labels)
            unknown_flag_counts.update(unknown_labels)
            for label in [*labels, *unknown_labels]:
                if len(examples[label]) < 5:
                    examples[label].append(
                        {"date": entry["parameters"]["date"], "symbol": row["symbol"], "name": row["name"], "exchange": row["exchange"]}
                    )
        stock_count = len(stocks)
        survivors = stock_count - confirmed
        results.append(
            {
                "date": entry["parameters"]["date"],
                "active_total": len(rows),
                "stock_labelled": stock_count,
                "etf_labelled": sum(row["assetType"] == "ETF" for row in rows),
                "symbol_duplicate_groups": sum(count > 1 for count in symbol_counts.values()),
                "exact_duplicate_extra_rows": sum(count - 1 for count in full_row_counts.values() if count > 1),
                "confirmed_non_common": confirmed,
                "apparently_eligible": survivors,
                "unknown": survivors,
                "contamination_rate_lower_bound": confirmed / stock_count if stock_count else None,
                "unknown_rate": survivors / stock_count if stock_count else None,
                "exchanges": sorted({row["exchange"] for row in rows}),
                "confirmed_label_counts": dict(confirmed_label_counts),
                "unknown_flag_counts": dict(unknown_flag_counts),
            }
        )
    print(json.dumps({"snapshots": results, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()

"""Public catalog validation; this module never calls a provider."""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REQUIRED = {"instrument_id", "reference_symbol", "asset_class", "provider_symbol_candidate", "mapping_type", "provider_mapping_state", "listing_market", "listing_currency", "identity_verification_state", "etf_structure_state", "direction", "leverage_multiple", "public_evidence_status", "review_reason"}


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogReport:
    records: int
    equities: int
    etfs: int
    unique_reference_symbols: int
    duplicate_provider_candidates: tuple[str, ...]
    review_records: int


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise CatalogError("catalog schema is incomplete")
    ids = [r["instrument_id"] for r in rows]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise CatalogError("instrument_id must be unique and non-empty")
    if any(r["asset_class"] not in {"equity", "etf"} for r in rows):
        raise CatalogError("unsupported asset_class")
    return rows


def report_catalog(rows: list[dict[str, str]]) -> CatalogReport:
    providers = [r["provider_symbol_candidate"].strip().upper() for r in rows if r["provider_symbol_candidate"].strip()]
    duplicate = tuple(sorted(k for k, n in Counter(providers).items() if n > 1))
    return CatalogReport(
        records=len(rows), equities=sum(r["asset_class"] == "equity" for r in rows),
        etfs=sum(r["asset_class"] == "etf" for r in rows),
        unique_reference_symbols=len({r["reference_symbol"] for r in rows}),
        duplicate_provider_candidates=duplicate,
        review_records=sum(r["provider_mapping_state"] != "VERIFIED" for r in rows),
    )


def is_downloadable(row: dict[str, str]) -> bool:
    """Only independently verified USD instruments may be downloaded."""
    if row["provider_mapping_state"] != "VERIFIED" or row["identity_verification_state"] != "VERIFIED":
        return False
    if row["listing_currency"] != "USD" or not row["listing_market"] or not row["provider_symbol_candidate"]:
        return False
    return row["asset_class"] != "etf" or (
        row["etf_structure_state"] == "CONFIRMED_DIRECT_UNLEVERAGED"
        and row["direction"] in {"DIRECT", "BULL"}
        and row["leverage_multiple"] == "1.0"
    )

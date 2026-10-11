"""Silver quality gate and atomic Gold promotion."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from .ingest import validate_silver


@dataclass(frozen=True)
class QualityResult:
    status: str
    warnings: tuple[str, ...]


def assess_silver(frame: pl.DataFrame, cutoff: str) -> QualityResult:
    validate_silver(frame)
    warnings = []
    if frame.filter(pl.col("volume") == 0).height: warnings.append("VALID_ZERO_VOLUME")
    if str(frame["session_date"].max()) != cutoff: return QualityResult("REJECTED", tuple(warnings + ["MISSING_CUTOFF_PRICE"]))
    return QualityResult("GOLD_APPROVED", tuple(warnings))


def write_parquet_atomic(frame: pl.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True); tmp = destination.with_name(f".{uuid.uuid4().hex}.tmp.parquet")
    try:
        frame.write_parquet(tmp)
        if not pl.read_parquet(tmp).equals(frame): raise IOError("parquet reread mismatch")
        os.replace(tmp, destination)
    finally:
        if tmp.exists(): tmp.unlink()

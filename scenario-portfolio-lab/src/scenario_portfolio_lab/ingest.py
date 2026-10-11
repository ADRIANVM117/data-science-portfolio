"""Alpha Vantage daily-response parsing and immutable raw persistence."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

SCHEMA = {"symbol": pl.String, "session_date": pl.Date, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64, "adjusted_close": pl.Float64, "volume": pl.Int64, "dividend_amount": pl.Float64, "split_coefficient": pl.Float64, "provider_timezone": pl.String, "source": pl.String, "provider_last_refreshed": pl.String, "ingested_at_utc": pl.Datetime(time_zone="UTC")}
START_DATE = date(2018, 1, 2)
END_DATE = date(2026, 10, 9)


class ProviderResponseError(ValueError):
    pass


def validate_cutoff(cutoff: str) -> date:
    """Parse and constrain every materialization/capture cut-off to the approved period."""
    value = date.fromisoformat(cutoff)
    if not START_DATE <= value <= END_DATE:
        raise ValueError("cutoff is outside the approved period")
    return value


def api_key_from_environment() -> str:
    value = os.getenv("ALPHAVANTAGE_API_KEY")
    if not value:
        # Deliberately load only this project's explicitly named local file.
        local_env = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(dotenv_path=local_env, override=False)
        value = os.getenv("ALPHAVANTAGE_API_KEY")
    if not value:
        raise ProviderResponseError("ALPHAVANTAGE_API_KEY is required from process environment or local project .env")
    return value


def parse_daily_adjusted(payload: dict, symbol: str) -> pl.DataFrame:
    if not isinstance(payload, dict) or any(k in payload for k in ("Error Message", "Information", "Note")):
        raise ProviderResponseError("provider returned an unusable response")
    meta, series = payload.get("Meta Data"), payload.get("Time Series (Daily)")
    expected = symbol.strip().upper()
    if not isinstance(meta, dict) or not isinstance(series, dict) or str(meta.get("2. Symbol", "")).upper() != expected:
        raise ProviderResponseError("provider metadata or identity mismatch")
    now = datetime.now(timezone.utc); rows = []
    fields = {"1. open":"open", "2. high":"high", "3. low":"low", "4. close":"close", "5. adjusted close":"adjusted_close", "6. volume":"volume", "7. dividend amount":"dividend_amount", "8. split coefficient":"split_coefficient"}
    for stamp, values in series.items():
        try:
            row = {name: (int(values[key]) if name == "volume" else float(values[key])) for key, name in fields.items()}
            row |= {"symbol":expected, "session_date":datetime.strptime(stamp, "%Y-%m-%d").date(), "provider_timezone":str(meta.get("5. Time Zone", "")), "source":"ALPHA_VANTAGE", "provider_last_refreshed":str(meta.get("3. Last Refreshed", "")), "ingested_at_utc":now}
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError(f"invalid daily bar {stamp}") from exc
        rows.append(row)
    frame = pl.DataFrame(rows, schema=SCHEMA, strict=True).filter(pl.col("session_date").is_between(START_DATE, END_DATE)).sort("session_date")
    validate_silver(frame)
    return frame


def validate_silver(frame: pl.DataFrame) -> None:
    if frame.is_empty() or frame.schema != pl.Schema(SCHEMA) or frame["session_date"].n_unique() != frame.height:
        raise ProviderResponseError("invalid schema, empty series, or duplicate sessions")
    for col in ("open", "high", "low", "close", "adjusted_close", "dividend_amount", "split_coefficient"):
        values = frame[col].to_list()
        if any(v is None or not math.isfinite(float(v)) for v in values): raise ProviderResponseError(f"non-finite {col}")
    if frame.select(((pl.col("open")<=0)|(pl.col("high")<=0)|(pl.col("low")<=0)|(pl.col("close")<=0)|(pl.col("adjusted_close")<=0)|(pl.col("dividend_amount")<0)|(pl.col("split_coefficient")<=0)).any()).item(): raise ProviderResponseError("invalid economic values")
    if frame.select(((pl.col("high") < pl.max_horizontal("open","close","low")) | (pl.col("low") > pl.min_horizontal("open","close","high"))).any()).item(): raise ProviderResponseError("inconsistent OHLC")
    if frame["volume"].null_count() or frame.select((pl.col("volume") < 0).any()).item(): raise ProviderResponseError("invalid volume")


def write_raw_once(raw: bytes, path: Path) -> dict[str, str]:
    if path.exists(): raise FileExistsError(f"raw destination occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(raw)
        if tmp.read_bytes() != raw: raise IOError("raw reread mismatch")
        os.replace(tmp, path)
    finally:
        if tmp.exists(): tmp.unlink()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": str(len(raw))}


def decode_raw(raw: bytes) -> dict:
    try: return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise ProviderResponseError("raw is not JSON") from exc

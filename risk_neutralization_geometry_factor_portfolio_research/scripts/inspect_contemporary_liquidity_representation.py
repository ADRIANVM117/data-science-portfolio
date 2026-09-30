"""Read-only audit of the proposed MedianDollarVolume_60Sessions representation.

This is a diagnostic script, not a universe gate or a production liquidity
implementation. It never selects a threshold and retains partial-history
results separately from full-history values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from xnys_pit import XNYS  # noqa: E402


RAW_ROOTS = (
    PROJECT_ROOT / "data" / "raw" / "contemporary_measurement_pilot",
    PROJECT_ROOT / "data" / "raw" / "contemporary_liquidity_stress",
)
T0 = pd.Timestamp("2026-09-25")
FULL_SESSIONS = 60
PARTIAL_MINIMUM = 20


def main() -> None:
    manifest = [
        entry
        for root in RAW_ROOTS
        for entry in json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ]
    latest: dict[str, dict[str, object]] = {}
    for entry in manifest:
        if entry["endpoint"] == "TIME_SERIES_DAILY_ADJUSTED" and entry["request_status"] == "HTTP_200":
            latest[str(entry["symbol"])] = entry
    sessions = pd.DatetimeIndex(XNYS.sessions[XNYS.sessions < T0])[-FULL_SESSIONS:].tz_localize(None).normalize()
    sessions20 = sessions[-20:]
    observations: list[dict[str, object]] = []
    for symbol, entry in sorted(latest.items()):
        payload = json.loads((PROJECT_ROOT / str(entry["local_raw_file"])).read_text())
        rows = []
        for date, row in payload.get("Time Series (Daily)", {}).items():
            rows.append({"date": pd.Timestamp(date), "close": row.get("4. close"), "volume": row.get("6. volume"), "split": row.get("8. split coefficient")})
        if len(rows) != len({row["date"] for row in rows}):
            raise ValueError(f"{symbol} raw provider response contains duplicate date rows")
        raw = pd.DataFrame(rows).set_index("date")
        frame = raw.reindex(sessions)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame["split"] = pd.to_numeric(frame["split"], errors="coerce")
        present = frame.index.isin(raw.index)
        valid_positive = frame["close"].gt(0) & frame["volume"].gt(0)
        valid_zero = frame["close"].gt(0) & frame["volume"].eq(0)
        valid = valid_positive | valid_zero
        invalid = present & ~valid
        dollar_volume = (frame.loc[valid, "close"] * frame.loc[valid, "volume"])
        n_valid = int(valid.sum())
        status = "FULL_HISTORY" if n_valid >= FULL_SESSIONS else "PARTIAL_HISTORY" if n_valid >= PARTIAL_MINIMUM else "UNKNOWN"
        window20_valid = valid.loc[sessions20]
        dollar_volume20 = (frame.loc[sessions20].loc[window20_valid, "close"] * frame.loc[sessions20].loc[window20_valid, "volume"])
        observations.append({
            "symbol": symbol, "window_start": str(sessions[0].date()), "window_end": str(sessions[-1].date()),
            "n_xnys_sessions": FULL_SESSIONS, "n_valid": n_valid,
            "valid_positive_volume": int(valid_positive.sum()), "valid_zero_volume": int(valid_zero.sum()),
            "missing_observations": int((~present).sum()), "invalid_observations": int(invalid.sum()),
            "split_sessions": int((frame.loc[valid, "split"] != 1).sum()), "history_status": status,
            "median_dollar_volume": float(dollar_volume.median()) if n_valid else None,
            "mean_dollar_volume": float(dollar_volume.mean()) if n_valid else None,
            "max_dollar_volume": float(dollar_volume.max()) if n_valid else None,
            "last_dollar_volume": float(dollar_volume.iloc[-1]) if n_valid else None,
            "n_valid_20": int(window20_valid.sum()),
            "median_dollar_volume_20": float(dollar_volume20.median()) if len(dollar_volume20) else None,
            "mean_dollar_volume_20": float(dollar_volume20.mean()) if len(dollar_volume20) else None,
        })
    print(json.dumps({"t0": str(T0.date()), "window": [str(sessions[0].date()), str(sessions[-1].date())], "policy": {"full": FULL_SESSIONS, "partial_minimum": PARTIAL_MINIMUM}, "observations": observations}, indent=2))


if __name__ == "__main__":
    main()

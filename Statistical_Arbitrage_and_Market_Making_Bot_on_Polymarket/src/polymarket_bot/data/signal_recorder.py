from __future__ import annotations

import csv
from pathlib import Path
from time import time

from signal_state import SignalTracker


class SignalRecorder:
    def __init__(self, output_path: str = "logs/persistent_signals.csv") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "timestamp",
            "asset_id",
            "signal",
            "count",
            "age_seconds",
            "mid_price",
            "microprice",
            "edge",
            "imbalance",
            "spread",
            "depth",
        ]

        if not self.output_path.exists():
            with self.output_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def record(self, tracker: SignalTracker) -> None:
        signal = tracker.last_signal

        if signal is None:
            return

        row = {
            "timestamp": time(),
            "asset_id": tracker.asset_id,
            "signal": tracker.current_signal.value,
            "count": tracker.consecutive_count,
            "age_seconds": round(tracker.age_seconds, 3),
            "mid_price": str(signal.mid_price),
            "microprice": str(signal.microprice),
            "edge": str(signal.microprice_edge),
            "imbalance": str(signal.imbalance),
            "spread": str(signal.spread),
            "depth": str(signal.depth),
        }

        with self.output_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)
from __future__ import annotations

import csv
from pathlib import Path
from time import time

from features import BookFeatures


class MarketSnapshotRecorder:
    def __init__(self, output_path: str = "logs/market_snapshots.csv") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "timestamp",
            "asset_id",
            "best_bid",
            "best_ask",
            "mid_price",
            "microprice",
            "edge",
            "spread",
            "imbalance",
            "depth",
        ]

        if not self.output_path.exists():
            with self.output_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def record(self, features: BookFeatures) -> None:
        row = {
            "timestamp": time(),
            "asset_id": features.asset_id,
            "best_bid": str(features.best_bid),
            "best_ask": str(features.best_ask),
            "mid_price": str(features.mid_price),
            "microprice": str(features.microprice),
            "edge": str(features.microprice - features.mid_price),
            "spread": str(features.spread),
            "imbalance": str(features.book_imbalance),
            "depth": str(features.top_depth),
        }

        with self.output_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(row)
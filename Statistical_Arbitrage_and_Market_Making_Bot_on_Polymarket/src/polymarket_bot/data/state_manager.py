from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from time import time
from typing import Optional

from features import BookFeatures
from filters import is_tradable

@dataclass
class MarketState:
    """
    Stores the latest computed features per asset_id.
    This is the in-memory state layer between market data and signals.
    """

    features_by_asset: dict[str, BookFeatures] = field(default_factory=dict)
    last_update_ts: dict[str, float] = field(default_factory=dict)

    def update(self, features: BookFeatures) -> None:
        """Update latest features for one asset."""
        self.features_by_asset[features.asset_id] = features
        self.last_update_ts[features.asset_id] = time()

    def get(self, asset_id: str) -> Optional[BookFeatures]:
        """Get latest features for one asset."""
        return self.features_by_asset.get(asset_id)

    def all_features(self) -> list[BookFeatures]:
        """Return all latest features."""
        return list(self.features_by_asset.values())

    def top_spreads(self, n: int = 10) -> list[BookFeatures]:
        """Assets with widest spreads."""
        return sorted(
            self.all_features(),
            key=lambda x: x.spread,
            reverse=True,
        )[:n]

    def top_buy_imbalance(self, n: int = 10) -> list[BookFeatures]:
        """Assets with strongest bid-side pressure."""
        return sorted(
            self.all_features(),
            key=lambda x: x.book_imbalance,
            reverse=True,
        )[:n]

    def top_sell_imbalance(self, n: int = 10) -> list[BookFeatures]:
        """Assets with strongest ask-side pressure."""
        return sorted(
            self.all_features(),
            key=lambda x: x.book_imbalance,
        )[:n]

    def top_depth(self, n: int = 10) -> list[BookFeatures]:
        """Assets with largest top-of-book depth."""
        return sorted(
            self.all_features(),
            key=lambda x: x.top_depth,
            reverse=True,
        )[:n]
    
    def tradable_features(self) -> list[BookFeatures]:
        """Return only features that pass the tradable-universe filter."""
        return [f for f in self.all_features() if is_tradable(f)]

    def top_tradable_spreads(self, n: int = 10) -> list[BookFeatures]:
        """Tradable assets with widest spreads."""
        return sorted(self.tradable_features(), key=lambda x: x.spread,reverse=True,)[:n]

    def top_tradable_buy_imbalance(self, n: int = 10) -> list[BookFeatures]:
        """Tradable assets with strongest bid-side pressure."""
        return sorted(self.tradable_features(),key=lambda x: x.book_imbalance,reverse=True,
                      )[:n]

    def top_tradable_sell_imbalance(self, n: int = 10) -> list[BookFeatures]:
        """Tradable assets with strongest ask-side pressure."""
        return sorted(self.tradable_features(),key=lambda x: x.book_imbalance,)[:n]

    def top_tradable_depth(self, n: int = 10) -> list[BookFeatures]:
        """Tradable assets with largest top-of-book depth."""
        return sorted(self.tradable_features(),key=lambda x: x.top_depth,reverse=True,)[:n]
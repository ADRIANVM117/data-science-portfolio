from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


@dataclass(frozen=True)
class Level:
    price: Decimal
    size: Decimal


@dataclass
class OrderBook:
    """
    In-memory L2 order book for one Polymarket asset_id.
    """

    asset_id: str
    market_id: Optional[str] = None
    bids: list[Level] = field(default_factory=list)
    asks: list[Level] = field(default_factory=list)
    timestamp_ms: Optional[int] = None
    book_hash: Optional[str] = None

    def update_from_snapshot(self, message: dict) -> None:
        """Replace the full book using a Polymarket 'book' snapshot."""
        self.market_id = message.get("market")
        self.book_hash = message.get("hash")

        timestamp = message.get("timestamp")
        self.timestamp_ms = int(timestamp) if timestamp is not None else None

        self.bids = sorted(
            [
                Level(
                    price=Decimal(str(x["price"])),
                    size=Decimal(str(x["size"])),
                )
                for x in message.get("bids", [])
            ],
            key=lambda x: x.price,
            reverse=True,
        )

        self.asks = sorted(
            [
                Level(
                    price=Decimal(str(x["price"])),
                    size=Decimal(str(x["size"])),
                )
                for x in message.get("asks", [])
            ],
            key=lambda x: x.price,
        )
    
    def apply_price_change(self, change: dict) -> None:
        """Aplica una actualización delta L2 (Price Level) de Polymarket."""
        side = change.get("side", "").upper()
        # En Polymarket el precio y tamaño vienen como strings
        price = Decimal(str(change["price"]))
        size = Decimal(str(change["size"]))
        if side == "BUY":
            self.bids = self._upsert_level(
                levels=self.bids,
                price=price,
                size=size,
                reverse=True,)
        
        elif side == "SELL":
            self.asks = self._upsert_level(
                levels=self.asks,
                price=price,
                size=size,
                reverse=False,)
        else:
            raise ValueError(f"Unknown side: {side}")
    
    


    @staticmethod
    def _upsert_level(
        levels: list[Level],
        price: Decimal,
        size: Decimal,
        reverse: bool,
    ) -> list[Level]:
        filtered = [lvl for lvl in levels if lvl.price != price]

        if size > 0:
            filtered.append(Level(price=price, size=size))

        return sorted(filtered, key=lambda x: x.price, reverse=reverse)

    def top_bids(self, n: int = 3) -> list[Level]:
        return self.bids[:n]

    def top_asks(self, n: int = 3) -> list[Level]:
        return self.asks[:n]

    @property
    def best_bid(self) -> Optional[Level]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[Level]:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[Decimal]:
        if self.best_bid is None or self.best_ask is None:
            return None

        return Decimal("0.5") * (self.best_bid.price + self.best_ask.price)

    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid is None or self.best_ask is None:
            return None

        return self.best_ask.price - self.best_bid.price

    @property
    def is_crossed(self) -> bool:
        if self.best_bid is None or self.best_ask is None:
            return False

        return self.best_bid.price >= self.best_ask.price

    @property
    def is_valid(self) -> bool:
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and self.spread is not None
            and self.spread > 0
        )
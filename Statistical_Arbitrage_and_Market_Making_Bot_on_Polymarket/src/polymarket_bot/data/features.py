from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from order_book import OrderBook


@dataclass(frozen=True)
class BookFeatures:
    asset_id: str
    best_bid: Decimal
    best_ask: Decimal
    bid_size: Decimal
    ask_size: Decimal
    spread: Decimal
    mid_price: Decimal
    microprice: Decimal
    book_imbalance: Decimal
    top_depth: Decimal


def compute_book_features(book: OrderBook) -> Optional[BookFeatures]:
    """
    Computes basic market microstructure features from a live L2 order book.
    Returns None if the book is not usable.
    """

    if book.best_bid is None or book.best_ask is None:
        return None

    bid = book.best_bid
    ask = book.best_ask

    bid_size = bid.size
    ask_size = ask.size

    total_size = bid_size + ask_size

    if total_size <= 0:
        return None

    spread = ask.price - bid.price
    mid_price = Decimal("0.5") * (bid.price + ask.price)

    book_imbalance = bid_size / total_size

    microprice = (
        ask.price * bid_size + bid.price * ask_size
    ) / total_size

    top_depth = bid_size + ask_size

    return BookFeatures(
        asset_id=book.asset_id,
        best_bid=bid.price,
        best_ask=ask.price,
        bid_size=bid_size,
        ask_size=ask_size,
        spread=spread,
        mid_price=mid_price,
        microprice=microprice,
        book_imbalance=book_imbalance,
        top_depth=top_depth,
    )
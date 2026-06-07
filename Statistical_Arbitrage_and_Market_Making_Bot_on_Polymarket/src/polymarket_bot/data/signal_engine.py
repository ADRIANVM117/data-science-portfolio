from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from features import BookFeatures


class SignalType(str, Enum):
    BUY_PRESSURE = "BUY_PRESSURE"
    SELL_PRESSURE = "SELL_PRESSURE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class MarketSignal:
    asset_id: str
    signal: SignalType
    mid_price: Decimal
    microprice: Decimal
    microprice_edge: Decimal
    imbalance: Decimal
    spread: Decimal
    depth: Decimal


def generate_signal(
    features: BookFeatures,
    min_edge: Decimal = Decimal("0.005"),
    buy_imbalance_threshold: Decimal = Decimal("0.65"),
    sell_imbalance_threshold: Decimal = Decimal("0.35"),
) -> MarketSignal:
    """
    Generate a simple microstructure signal.

    Logic:
    - BUY_PRESSURE if microprice is above mid and bid-side imbalance is strong.
    - SELL_PRESSURE if microprice is below mid and ask-side imbalance is strong.
    - NEUTRAL otherwise.
    """

    microprice_edge = features.microprice - features.mid_price

    if (microprice_edge >= min_edge and features.book_imbalance >= buy_imbalance_threshold):
        signal = SignalType.BUY_PRESSURE

    elif (microprice_edge <= -min_edge and features.book_imbalance <= sell_imbalance_threshold):
        signal = SignalType.SELL_PRESSURE

    else:
        signal = SignalType.NEUTRAL

    return MarketSignal(
        asset_id=features.asset_id,
        signal=signal,
        mid_price=features.mid_price,
        microprice=features.microprice,
        microprice_edge=microprice_edge,
        imbalance=features.book_imbalance,
        spread=features.spread,
        depth=features.top_depth,)
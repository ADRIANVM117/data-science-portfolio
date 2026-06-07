from __future__ import annotations

from decimal import Decimal

from features import BookFeatures


def is_tradable(
    features: BookFeatures,
    max_spread: Decimal = Decimal("0.05"),
    min_depth: Decimal = Decimal("100"),
    min_mid: Decimal = Decimal("0.05"),
    max_mid: Decimal = Decimal("0.95"),
) -> bool:
    """
    Basic tradable-universe filter.

    Keeps assets with:
    - reasonable spread
    - enough top-of-book depth
    - mid price away from extreme boundaries
    """

    return (
        features.spread <= max_spread
        and features.top_depth >= min_depth
        and min_mid <= features.mid_price <= max_mid
    )
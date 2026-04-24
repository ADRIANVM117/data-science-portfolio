import numpy as np 
def implementation_shortfall(
    execution_prices: np.ndarray,
    shares_per_slice: np.ndarray,
    S0: float
):
    """
    Compute implementation shortfall.

    Returns:
        total_cost
        average_execution_price
        IS_per_share
    """

    total_shares = shares_per_slice.sum()

    total_cost = np.sum(shares_per_slice * (execution_prices - S0))

    avg_execution_price = np.sum(shares_per_slice * execution_prices) / total_shares

    is_per_share = avg_execution_price - S0

    return {
        "total_cost": total_cost,
        "avg_price": avg_execution_price,
        "is_per_share": is_per_share
    }
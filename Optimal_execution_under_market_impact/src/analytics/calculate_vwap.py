import numpy as np
def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """
    Calculate Volume-Weighted Average Price.

    Parameters
    ----------
    prices : np.ndarray
        Array of prices for each interval
    volumes : np.ndarray
        Array of volumes for each interval

    Returns
    -------
    float
        Volume-weighted average price
    """
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")
    if np.sum(volumes) == 0:
        raise ValueError("Total volume cannot be zero")

    return np.sum(prices * volumes) / np.sum(volumes)
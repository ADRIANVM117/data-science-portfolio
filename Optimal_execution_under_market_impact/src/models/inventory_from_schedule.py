import numpy as np
def inventory_from_schedule(Q: float, shares_per_slice: np.ndarray):
    """
    Convert execution schedule into inventory trajectory.

    x_k = remaining shares after k trades
    """
    cumulative = np.cumsum(shares_per_slice)
    inventory = np.concatenate(([Q], Q - cumulative))
    return inventory
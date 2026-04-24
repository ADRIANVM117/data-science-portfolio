import numpy as np
def simulate_brownian_price_path(
    S0: float,
    sigma: float,
    T: float,
    N: int,
    seed: int = None
):
    """
    Simulate arithmetic Brownian motion price path.

    S_{k+1} = S_k + sigma * sqrt(tau) * epsilon_k
    """
    if seed is not None:
        np.random.seed(seed)

    tau = T / N

    # Gaussian shocks
    epsilon = np.random.normal(0, 1, N)

    # Price increments
    dS = sigma * np.sqrt(tau) * epsilon

    # Build price path
    prices = np.empty(N + 1)
    prices[0] = S0
    prices[1:] = S0 + np.cumsum(dS)

    t_points = np.arange(N + 1) * tau

    return {
        "t_points": t_points,
        "prices": prices,
        "increments": dS,
        "shocks": epsilon
    }


######################3

def simulate_multiple_price_paths(
    S0: float,
    sigma: float,
    T: float,
    N: int,
    n_paths: int
):
    """
    Simulate multiple Brownian price paths.
    """
    tau = T / N

    shocks = np.random.normal(0, 1, (n_paths, N))
    increments = sigma * np.sqrt(tau) * shocks

    prices = np.zeros((n_paths, N + 1))
    prices[:, 0] = S0
    prices[:, 1:] = S0 + np.cumsum(increments, axis=1)

    return prices


def simulate_execution_prices(
    mid_prices: np.ndarray,
    shares_per_slice: np.ndarray,
    gamma: float,
    eta: float,
    tau: float
):
    """
    Compute execution prices with market impact.

    Parameters
    ----------
    mid_prices : np.ndarray
        Array of mid prices S_k (length N+1)
    shares_per_slice : np.ndarray
        Shares executed n_k (length N)
    gamma : float
        Permanent impact coefficient
    eta : float
        Temporary impact coefficient
    tau : float
        Time interval length

    Returns
    -------
    dict with:
        - execution_prices
        - permanent_impact
        - temporary_impact
        - cumulative_volume
    """

    N = len(shares_per_slice)

    # Cumulative volume BEFORE each trade
    cumulative_volume = np.zeros(N)
    cumulative_volume[1:] = np.cumsum(shares_per_slice[:-1])

    # Permanent impact
    permanent_impact = gamma * cumulative_volume

    # Temporary impact
    temporary_impact = eta * (shares_per_slice / tau)

    # Execution prices (IMPORTANT: use S_k, not S_{k+1})
    execution_prices = mid_prices[:-1] + permanent_impact + temporary_impact

    return {
        "execution_prices": execution_prices,
        "permanent_impact": permanent_impact,
        "temporary_impact": temporary_impact,
        "cumulative_volume": cumulative_volume
    }
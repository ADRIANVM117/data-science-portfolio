import numpy as np

def almgren_chriss_schedule(
    Q: float,
    T: float,
    N: int,
    sigma: float,
    eta: float,
    lambda_risk: float
):
    """
    Generate Almgren-Chriss optimal execution schedule.

    Returns
    -------
    dict with:
        - t_points
        - inventory_path
        - shares_per_slice
        - kappa
    """
    tau = T / N
    t_points = np.arange(N + 1) * tau

    if lambda_risk < 1e-12:
        kappa = 0.0
        inventory_path = Q * (T - t_points) / T
    else:
        kappa = np.sqrt(lambda_risk * sigma**2 / eta)
        inventory_path = Q * np.sinh(kappa * (T - t_points)) / np.sinh(kappa * T)

    shares_per_slice = -np.diff(inventory_path)

    return {
        "t_points": t_points,
        "inventory_path": inventory_path,
        "shares_per_slice": shares_per_slice,
        "kappa": kappa
    }
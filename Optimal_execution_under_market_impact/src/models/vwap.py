
import numpy as np 
def generate_vwap_schedule(total_shares: float, volume_profile: np.ndarray) -> np.ndarray:
    """
    Generate a VWAP-style execution schedule.

    Allocates shares proportionally to expected volume in each period.

    Parameters
    ----------
    total_shares : float
        Total shares to execute
    volume_profile : np.ndarray
        Expected relative volume in each period

    Returns
    -------
    np.ndarray
        Shares to execute in each period
    """
    volume_profile = np.asarray(volume_profile, dtype=float)

    if np.any(volume_profile < 0):
        raise ValueError("Volume profile must be non-negative")
    if np.sum(volume_profile) == 0:
        raise ValueError("Volume profile cannot sum to zero")

    weights = volume_profile / np.sum(volume_profile)
    return total_shares * weights
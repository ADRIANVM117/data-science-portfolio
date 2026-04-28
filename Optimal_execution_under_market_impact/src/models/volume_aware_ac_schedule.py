import numpy as np
def volume_aware_ac_schedule(
    ac_shares: np.ndarray,
    vwap_shares: np.ndarray,
    alpha: float
):
    """
    Construct a volume-aware Almgren-Chriss schedule.

    n_k = alpha * n_k_AC + (1 - alpha) * n_k_VWAP

    Parameters
    ----------
    ac_shares : np.ndarray
        Almgren-Chriss shares per slice.
    vwap_shares : np.ndarray
        VWAP shares per slice based on real volume.
    alpha : float
        Weight assigned to Almgren-Chriss schedule.

    Returns
    -------
    np.ndarray
        Volume-aware Almgren-Chriss schedule.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    if len(ac_shares) != len(vwap_shares):
        raise ValueError("AC and VWAP schedules must have the same length")

    blended = alpha * ac_shares + (1 - alpha) * vwap_shares

    # Numerical correction to ensure total shares exactly equals Q
    blended *= ac_shares.sum() / blended.sum()

    return blended
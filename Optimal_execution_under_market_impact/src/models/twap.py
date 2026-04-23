import  numpy as np 

def twap_schedule(Q, T, N):
    """
      Parameters
    ----------
    Q : total shares
    T : execution horizon
    N : number of slices
    """
    tau = T / N
    t_points = np.arange(N + 1) * tau

    shares_per_slice = np.full(N, Q / N)
    inventory_path = Q - np.concatenate(([0], np.cumsum(shares_per_slice)))

    return {
        "t_points": t_points,
        "shares_per_slice": shares_per_slice,
        "inventory_path": inventory_path
    }
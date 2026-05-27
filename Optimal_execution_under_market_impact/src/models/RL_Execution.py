import numpy as np 
class StateBuilder:
    def __init__(self, market_data, feature_cols):
        self.market_data = market_data.reset_index(drop=True).copy()
        self.feature_cols = feature_cols

    def build_state(self, t, inventory_remaining, order_size):
        """
        Build the RL state vector at timestep t.

        State:
        [
            inventory_remaining / order_size,
            time_remaining / total_steps,
            rel_volume_t,
            rolling_vol_t,
            spread_proxy_pct_t,
            momentum_t
        ]
        """

        total_steps = len(self.market_data)

        inventory_ratio = inventory_remaining / order_size
        time_remaining = (total_steps - t - 1) / total_steps

        market_features = (
            self.market_data
            .loc[t, self.feature_cols]
            .values
            .astype(np.float32)
        )

        state = np.concatenate([
            np.array(
                [inventory_ratio, time_remaining],
                dtype=np.float32
            ),
            market_features
        ])

        return state
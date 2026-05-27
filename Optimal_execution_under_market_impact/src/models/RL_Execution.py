import numpy as np 
import pandas as pd
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
    

# Execution Environment
class ExecutionEnv:
    def __init__(
        self,
        market_data,
        state_builder,
        order_size=100_000,
        action_space=None,
        eta=0.05,
        inventory_penalty=1e-10,
        participation_penalty=1e-2,
        terminal_penalty=1e-6
    ):
        self.market_data = market_data.reset_index(drop=True).copy()
        self.state_builder = state_builder
        self.order_size = order_size

        self.action_space = action_space or  np.linspace(0.00,0.05,51).tolist()

        self.eta = eta
        self.inventory_penalty = inventory_penalty
        self.participation_penalty = participation_penalty
        self.terminal_penalty = terminal_penalty

        self.total_steps = len(self.market_data)

        self.reset()

    def reset(self):
        self.t = 0
        self.inventory_remaining = self.order_size
        self.arrival_price = float(self.market_data.loc[0, "Close"])

        self.cumulative_cost = 0.0
        self.execution_history = []

        state = self.state_builder.build_state(
            t=self.t,
            inventory_remaining=self.inventory_remaining,
            order_size=self.order_size
        )

        return state

    def step(self, action_idx):
        action_fraction = self.action_space[action_idx]

        price = float(self.market_data.loc[self.t, "Close"])
        volume = float(self.market_data.loc[self.t, "Volume"])

        volume = max(volume, 1.0)

        # Force liquidation at final step
        is_last_step = self.t == self.total_steps - 1

        if is_last_step:
            shares_to_trade = self.inventory_remaining
        else:
            shares_to_trade = action_fraction * self.order_size

        shares_to_trade = min(shares_to_trade, self.inventory_remaining)

        participation_rate = shares_to_trade / volume

        # Temporary market impact model
        execution_price = price + self.eta * participation_rate

        # Implementation shortfall-style cost
        step_cost = shares_to_trade * (execution_price - self.arrival_price)

        self.cumulative_cost += step_cost

        # Update inventory
        self.inventory_remaining -= shares_to_trade

        # Reward as negative penalty
        reward = -step_cost
        reward -= self.inventory_penalty * (self.inventory_remaining ** 2)
        reward -= self.participation_penalty * (participation_rate ** 2)

        # Terminal penalty if inventory remains
        if is_last_step and self.inventory_remaining > 0:
            reward -= self.terminal_penalty * (self.inventory_remaining ** 2)

        self.execution_history.append({
            "t": self.t,
            "price": price,
            "volume": volume,
            "action_fraction": action_fraction,
            "shares_traded": shares_to_trade,
            "inventory_remaining": self.inventory_remaining,
            "participation_rate": participation_rate,
            "execution_price": execution_price,
            "step_cost": step_cost,
            "cumulative_cost": self.cumulative_cost,
            "reward": reward
        })

        self.t += 1

        done = (
            self.t >= self.total_steps
            or self.inventory_remaining <= 0
        )

        if done:
            next_state = np.zeros_like(
                self.state_builder.build_state(
                    t=0,
                    inventory_remaining=self.order_size,
                    order_size=self.order_size
                )
            )
        else:
            next_state = self.state_builder.build_state(
                t=self.t,
                inventory_remaining=self.inventory_remaining,
                order_size=self.order_size
            )

        info = {
            "t": self.t,
            "inventory_remaining": self.inventory_remaining,
            "cumulative_cost": self.cumulative_cost,
            "participation_rate": participation_rate,
            "shares_traded": shares_to_trade,
            "execution_price": execution_price
        }

        return next_state, reward, done, info

    def get_execution_history(self):
        return pd.DataFrame(self.execution_history)
    
# --------------------------------------------------------


def run_random_policy(env, seed=42):
    np.random.seed(seed)
    
    state = env.reset()
    done = False
    
    while not done:
        action_idx = np.random.choice(len(env.action_space))
        next_state, reward, done, info = env.step(action_idx)
        state = next_state
    
    history = env.get_execution_history()
    history["policy"] = "Random"
    
    return history


def run_twap_policy(env):
    state = env.reset()
    done = False
    
    while not done:
        inventory_remaining = env.inventory_remaining
        steps_remaining = env.total_steps - env.t
        
        target_shares = inventory_remaining / steps_remaining
        
        # action is fraction of total parent order
        action_fraction = target_shares / env.order_size
        
        action_idx = np.argmin(
            np.abs(np.array(env.action_space) - action_fraction)
        )
        
        next_state, reward, done, info = env.step(action_idx)
        state = next_state
    
    history = env.get_execution_history()
    history["policy"] = "TWAP"
    
    return history


def run_vwap_like_policy(env):
    state = env.reset()
    done = False
    
    while not done:
        inventory_remaining = env.inventory_remaining
        
        volume_now = env.market_data.loc[env.t, "Volume"]
        volume_remaining = env.market_data.loc[env.t:, "Volume"].sum()
        
        target_shares = inventory_remaining * (
            volume_now / volume_remaining
        )
        
        # action is fraction of total parent order
        action_fraction = target_shares / env.order_size
        
        action_idx = np.argmin(
            np.abs(np.array(env.action_space) - action_fraction)
        )
        
        next_state, reward, done, info = env.step(action_idx)
        state = next_state
    
    history = env.get_execution_history()
    history["policy"] = "VWAP-like"
    
    return history


def run_volume_aware_ac_policy(
    env,
    risk_aversion=1.0,
    volume_weight=0.7
):
    """
    Volume-Aware Almgren-Chriss inspired policy.

    Combines:
    - time urgency
    - remaining inventory
    - relative liquidity

    The policy increases execution when volume is high
    and slows down when volume is low.
    """

    state = env.reset()
    done = False

    while not done:
        inventory_remaining = env.inventory_remaining
        steps_remaining = env.total_steps - env.t

        volume_now = env.market_data.loc[env.t, "Volume"]
        volume_remaining = env.market_data.loc[env.t:, "Volume"].sum()

        # TWAP component
        twap_target = inventory_remaining / steps_remaining

        # VWAP liquidity component
        vwap_target = inventory_remaining * (
            volume_now / volume_remaining
        )

        # Volume-aware AC-style blend
        target_shares = (
            (1 - volume_weight) * twap_target
            + volume_weight * vwap_target
        )

        # Risk aversion adjustment
        urgency_multiplier = 1 + risk_aversion * (
            1 / steps_remaining
        )

        target_shares = target_shares * urgency_multiplier

        # Convert target shares to action fraction of total parent order
        action_fraction = target_shares / env.order_size

        action_idx = np.argmin(
            np.abs(np.array(env.action_space) - action_fraction)
        )

        next_state, reward, done, info = env.step(action_idx)
        state = next_state

    history = env.get_execution_history()
    history["policy"] = "VA-AC"

    return history
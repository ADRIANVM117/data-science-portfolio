"""
Strategy Backtest Engine

Purpose
-------
Simulate the intraday pair strategy at trade level.

This module starts with a single pair-day backtest function.
The engine respects:
- SMA15 warm-up from 09:15
- Trading only from 09:30 to 15:55
- No overnight positions
- Fixed notional per trade
- Transaction costs per share
"""

from pathlib import Path
import pandas as pd


def load_pair_day_prices(leader,follower,trading_day,data_dir,):
    data_dir = Path(data_dir)
    df_leader = pd.read_parquet(data_dir / f"{leader}_backtest_1min_warmup.parquet")
    df_follower = pd.read_parquet(data_dir / f"{follower}_backtest_1min_warmup.parquet")
    trading_day = pd.to_datetime(trading_day).date()
    df_leader = df_leader.loc[df_leader["date"].dt.date == trading_day,["date", "close"]].rename(columns={"close": "leader_price"})
    df_follower = df_follower.loc[df_follower["date"].dt.date == trading_day,["date", "close"]].rename(columns={"close": "follower_price"})
    
    df = (df_leader.merge(df_follower, on="date", how="inner").sort_values("date").reset_index(drop=True))

    return df
#########################################################################
def prepare_pair_day_signals(df,leader_edge=0.006,follower_edge=0.002,):
    df = df.copy()
    df["leader_sma15"] = df["leader_price"].rolling(15).mean()
    df["follower_sma15"] = df["follower_price"].rolling(15).mean()
    df["leader_long_sma"] = df["leader_sma15"] * (1 + leader_edge)
    df["follower_long_sma"] = df["follower_sma15"] * (1 + follower_edge)
    df["leader_short_sma"] = df["leader_sma15"] * (1 - leader_edge)
    df["follower_short_sma"] = df["follower_sma15"] * (1 - follower_edge)
    df["time"] = df["date"].dt.time

    return df
###########################################################################
def run_pair_day_backtest(leader,follower,trading_day,data_dir,notional=100_000,cost_per_share=0.0035,leader_edge=0.006,follower_edge=0.002,min_price=10.0,):
    df = load_pair_day_prices(leader=leader,follower=follower,trading_day=trading_day,data_dir=data_dir,)

    if df.empty:
        return pd.DataFrame()

    df = prepare_pair_day_signals(df,leader_edge=leader_edge,follower_edge=follower_edge,)

    trade_start = pd.to_datetime("09:30").time()
    trade_end = pd.to_datetime("15:55").time()

    trades = []
    position = None
    trade_completed = False  # max 1 completed trade per pair per day

    for _, row in df.iterrows():
        current_time = row["time"]

        if current_time < trade_start:
            continue

        if current_time > trade_end:
            break

        leader_price = row["leader_price"]
        follower_price = row["follower_price"]

        if follower_price <= min_price:
            continue

        if position is None and not trade_completed:
            long_entry = (leader_price > row["leader_long_sma"] and follower_price < row["follower_long_sma"])

            short_entry = (leader_price < row["leader_short_sma"] and follower_price < row["follower_short_sma"])

            if long_entry:
                shares = notional / follower_price
                position = {"side": "long", "entry_time": row["date"], "entry_price": follower_price,"shares": shares,}

            elif short_entry:
                shares = notional / follower_price
                position = {"side": "short", "entry_time": row["date"], "entry_price": follower_price, "shares": shares,}

        elif position is not None:
            exit_signal = False
            exit_reason = None

            if position["side"] == "long":
                if leader_price < row["leader_sma15"]:
                    exit_signal = True
                    exit_reason = "leader_below_base_sma"

            elif position["side"] == "short":
                if leader_price > row["leader_sma15"]:
                    exit_signal = True
                    exit_reason = "leader_above_base_sma"

            if current_time >= trade_end:
                exit_signal = True
                exit_reason = "forced_eod_exit"

            if exit_signal:
                exit_price = follower_price
                shares = position["shares"]

                if position["side"] == "long":
                    gross_pnl = shares * (exit_price - position["entry_price"])
                else:
                    gross_pnl = shares * (position["entry_price"] - exit_price)

                transaction_cost = 2 * shares * cost_per_share
                net_pnl = gross_pnl - transaction_cost

                trades.append({
                    "trading_day": trading_day,
                    "leader": leader,
                    "follower": follower,
                    "side": position["side"],
                    "entry_time": position["entry_time"],
                    "exit_time": row["date"],
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "shares": shares,
                    "gross_pnl": gross_pnl,
                    "transaction_cost": transaction_cost,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                })

                position = None
                trade_completed = True

    return pd.DataFrame(trades)

#####################################################

def run_single_day_backtest(daily_universe,trading_day,data_dir,notional=100_000,cost_per_share=0.0035,
                            leader_edge=0.006,follower_edge=0.002,min_price=10.0,):
    pairs_today = daily_universe.loc[daily_universe["trading_day"] == trading_day].copy()
    trades = []
    for _, row in pairs_today.iterrows():
        pair_trades = run_pair_day_backtest(leader=row["leader"],
                                            follower=row["follower"],
                                            trading_day=trading_day,
                                            data_dir=data_dir,
                                            notional=notional,
                                            cost_per_share=cost_per_share,
                                            leader_edge=leader_edge,
                                            follower_edge=follower_edge,
                                            min_price=min_price,)

        if not pair_trades.empty:
            trades.append(pair_trades)

    if not trades:
        return pd.DataFrame()

    return pd.concat(trades, ignore_index=True)

#####################################################

def run_full_backtest(daily_universe,data_dir,notional=100_000,cost_per_share=0.0035,leader_edge=0.006,follower_edge=0.002,min_price=10.0,):
    trading_days = sorted(daily_universe["trading_day"].unique())
    all_trades = []
    for i, trading_day in enumerate(trading_days, start=1):
        print(f"{i}/{len(trading_days)} - {trading_day}")
        day_trades = run_single_day_backtest(
            daily_universe=daily_universe,
            trading_day=trading_day,
            data_dir=data_dir,
            notional=notional,
            cost_per_share=cost_per_share,
            leader_edge=leader_edge,
            follower_edge=follower_edge,
            min_price=min_price,
        )

        if not day_trades.empty:
            all_trades.append(day_trades)

    if not all_trades:
        return pd.DataFrame()

    return pd.concat(all_trades, ignore_index=True)
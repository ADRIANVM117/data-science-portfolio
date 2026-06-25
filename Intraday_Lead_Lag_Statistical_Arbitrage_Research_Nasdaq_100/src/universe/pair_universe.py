"""
Pair Universe Construction

Build the daily tradable universe used by the backtest.

Pipeline

1. Read symbol-level metadata from backtest parquet files.
2. Apply manual sector corrections.
3. Generate all possible stock pairs within the same sector.
4. Compute rolling 60-day correlations using 1-minute RTH log returns.
5. Shift correlations one trading day forward to avoid look-ahead bias.
6. Rank pairs by rolling correlation for each trading day.
7. Select the Top50 most correlated pairs.
8. Assign leader and follower using market capitalization:
    - Leader = larger market-cap stock
    - Follower = smaller market-cap stock

Output
------
Daily tradable universe:

    trading_day
    stock_a
    stock_b
    sector
    rolling_corr_60d
    leader
    follower

Bias Controls
-------------
- Same-sector pairs only.
- Correlations use only information available up to t-1.
- Top50 selection is shifted forward one trading day.
- No future information is used in pair ranking.
- Leader/follower assignment uses static universe market capitalization.

Notes
-----
The resulting universe is the only set of pairs allowed to trade
during each trading day in the backtest engine.
"""


import pandas as pd
import numpy as np
from pathlib import Path
import sys
import polars as pl
import pyarrow
import gc
from itertools import combinations
sys.path.append(str(Path("..")))

def build_universe_metadata_from_symbol_parquets(data_dir):
    records = []

    for file in Path(data_dir).glob("*_backtest_1min_warmup.parquet"):
        df = pd.read_parquet(file, columns=["symbol", "market_cap", "sic_description", "company_name"])
        records.append({
            "symbol": df["symbol"].iloc[0],
            "market_cap": df["market_cap"].iloc[0],
            "sector": df["sic_description"].iloc[0],
            "company_name": df["company_name"].iloc[0],
        })

    universe_df = pd.DataFrame(records).sort_values("symbol").reset_index(drop=True)
    return universe_df
#################################
def apply_sector_fixes(universe_df, sector_fixes):
    universe_df = universe_df.copy()
    for symbol, sector in sector_fixes.items():
        universe_df.loc[universe_df["symbol"] == symbol,"sector"] = sector
    return universe_df

#####################################
def generate_candidate_pairs(universe_df):
    assert universe_df["sector"].isna().sum() == 0, "There are missing sectors."
    pair_records = []
    for sector, group in universe_df.groupby("sector"):
        symbols = sorted(group["symbol"].unique())
        for stock_a, stock_b in combinations(symbols, 2):
            pair_records.append({
                "stock_a": stock_a,
                "stock_b": stock_b,
                "sector": sector
            })

    return pd.DataFrame(pair_records)

#####################################################

def load_symbol_backtest_returns(symbol, data_dir):
    df = pd.read_parquet(
        data_dir / f"{symbol}_backtest_1min_warmup.parquet"
    )

    df = df.sort_values("date").copy()

    # Correlation universe uses RTH only, not SMA warm-up period
    df = df.loc[
        (df["date"].dt.time >= pd.to_datetime("09:30").time()) &
        (df["date"].dt.time <= pd.to_datetime("16:00").time())
    ].copy()

    df["log_return_1m"] = np.log(df["close"] / df["close"].shift(1))
    df["trading_day"] = df["date"].dt.date

    return df[["date", "trading_day", "symbol", "close", "log_return_1m"]]
###############################################################################
def compute_pair_daily_corr_previous_60d(stock_a,stock_b,data_dir,window_size):
    df_a = load_symbol_backtest_returns(stock_a, data_dir)
    df_b = load_symbol_backtest_returns(stock_b, data_dir)
    df_pair = (df_a[["date", "log_return_1m"]].rename(columns={"log_return_1m": "ret_a"}).merge(df_b[["date", "log_return_1m"]].rename(columns={"log_return_1m": "ret_b"}),
            on="date",
            how="inner").sort_values("date").dropna().reset_index(drop=True))
    df_pair["trading_day_raw"] = df_pair["date"].dt.date
    df_pair["rolling_corr_60d"] = (df_pair["ret_a"].rolling(window_size).corr(df_pair["ret_b"]))
    daily_corr = (df_pair.groupby("trading_day_raw", as_index=False).tail(1)[["trading_day_raw", "rolling_corr_60d"]].copy())
    daily_corr["stock_a"] = stock_a
    daily_corr["stock_b"] = stock_b
    return daily_corr

#############################################################################
def build_daily_tradable_universe(candidate_pairs,universe_df,data_dir,top_n=50,window_days=60,bars_per_day=391):
    window_size = window_days * bars_per_day
    all_corrs = []
    for i, row in candidate_pairs.iterrows():
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]
        sector = row["sector"]
        print(f"{i+1}/{len(candidate_pairs)} - {stock_a}-{stock_b}")
        pair_corr = compute_pair_daily_corr_previous_60d(stock_a=stock_a,stock_b=stock_b,data_dir=data_dir,window_size=window_size)
        pair_corr["sector"] = sector
        all_corrs.append(pair_corr)

    daily_corrs = pd.concat(all_corrs, ignore_index=True)
    trading_days = sorted(daily_corrs["trading_day_raw"].unique())
    next_day_map = {trading_days[i]: trading_days[i + 1] for i in range(len(trading_days) - 1)}
    daily_corrs["trading_day"] = daily_corrs["trading_day_raw"].map(next_day_map)

    daily_corrs = daily_corrs.dropna(subset=["trading_day", "rolling_corr_60d"]).copy()
    market_cap_map = universe_df.set_index("symbol")["market_cap"].to_dict()

    def assign_roles(row):
        a = row["stock_a"]
        b = row["stock_b"]
        if market_cap_map[a] >= market_cap_map[b]:
            return pd.Series({"leader": a, "follower": b})
        else:
            return pd.Series({"leader": b, "follower": a})

    roles = daily_corrs.apply(assign_roles, axis=1)
    daily_corrs = pd.concat([daily_corrs, roles], axis=1)

    daily_top50 = (daily_corrs.sort_values(["trading_day", "rolling_corr_60d"],ascending=[True, False]).groupby("trading_day", as_index=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return daily_top50
from pathlib import Path
import pandas as pd


def build_buy_hold_benchmark_from_raw(raw_path,start_date,end_date,initial_capital=100_000,):
    df = pd.read_parquet(raw_path)
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df["trading_day"] = df["date"].dt.date

    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    df = df.loc[(df["trading_day"] >= start_date) & (df["trading_day"] <= end_date) & (df["date"].dt.time >= pd.to_datetime("09:30").time()) & (df["date"].dt.time <= pd.to_datetime("16:00").time())].copy()

    daily_close = (df.groupby("trading_day", as_index=False).tail(1)[["trading_day", "close"]].rename(columns={"close": "benchmark_close"}).reset_index(drop=True))

    entry_price = daily_close["benchmark_close"].iloc[0]
    shares = initial_capital / entry_price

    daily_close["benchmark_equity"] = shares * daily_close["benchmark_close"]
    daily_close["benchmark_return"] = daily_close["benchmark_equity"].pct_change().fillna(0)
    daily_close["symbol"] = "QQQ"

    return daily_close
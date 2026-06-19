"""
Backtest Data Ingestion Pipeline

Construct symbol-level 1-minute parquet files for strategy backtesting.

Pipeline

1. Load raw Nasdaq100 dataset.
2. Filter trading window (09:15–16:00).
3. Remove invalid constituents using avoid-ticker rules.
4. Resample each symbol-day to a complete 1-minute grid.
5. Forward-fill prices and zero-fill flows.
6. Save one parquet per symbol.

Output
------
data/backtest_1min_with_warmup_by_symbol/

Notes
-----
09:15–09:29 is used exclusively for SMA warm-up.
Trading decisions start at 09:30.
"""


from pathlib import Path
import gc
import pandas as pd
import polars as pl

def apply_avoid_tickers_polars(lf, avoid_dict):
    for period, tickers in avoid_dict.items():
        start_str, end_str = period.split("/")
        start = pl.datetime(int(start_str[:4]),int(start_str[5:7]),int(start_str[8:10]))
        end = pl.datetime(int(end_str[:4]),int(end_str[5:7]),int(end_str[8:10]))
        lf = lf.filter(~((pl.col("date") >= start) & (pl.col("date") <= end) & (pl.col("symbol").is_in(tickers))))

    return lf
#####################################################################

def resample_symbol_day_to_1min_window(df_symbol,start_time=(9, 15),end_time=(16, 0)):
    df_symbol = df_symbol.sort_values("date").copy()
    df_symbol["trading_day"] = df_symbol["date"].dt.date

    out = []
    for _, df_day in df_symbol.groupby("trading_day", observed=True):
        symbol = df_day["symbol"].iloc[0]
        market_cap = df_day["market_cap"].iloc[0]
        sic_description = df_day["sic_description"].iloc[0]
        company_name = df_day["company_name"].iloc[0]
        day = pd.Timestamp(df_day["date"].iloc[0].date())
        full_index = pd.date_range(start=day + pd.Timedelta(hours=start_time[0], minutes=start_time[1]),end=day + pd.Timedelta(hours=end_time[0], minutes=end_time[1]),
            freq="1min")
        df_day = df_day.set_index("date")
        resampled = df_day.resample("1min").agg({
            "open": "last",
            "high": "last",
            "low": "last",
            "close": "last",
            "vwap": "last",
            "volume": "sum",
            "transactions": "sum",
        })
        resampled = resampled.reindex(full_index)
        price_cols = ["open", "high", "low", "close", "vwap"]
        flow_cols = ["volume", "transactions"]
        resampled[price_cols] = resampled[price_cols].ffill().bfill()
        resampled[flow_cols] = resampled[flow_cols].fillna(0)
        resampled["symbol"] = symbol
        resampled["market_cap"] = market_cap
        resampled["sic_description"] = sic_description
        resampled["company_name"] = company_name
        resampled = resampled.reset_index().rename(columns={"index": "date"})
        out.append(resampled)

    return pd.concat(out, ignore_index=True)

######################################################

def build_backtest_warmup_parquets(
    input_file="nasdaq100_with_meta.parquet",
    data_dir="../data",
    output_dir="../data/backtest_1min_with_warmup_by_symbol",
    avoid_dict=None,
    start_time=(9, 15),
    end_time=(16, 0),
    output_suffix="_backtest_1min_warmup.parquet",
):
    """
    Build 1-minute symbol-level parquet files for the backtest.
    """

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / input_file
    print(f"Loading raw parquet lazily: {raw_path}")
    lf = pl.scan_parquet(raw_path)
    lf = lf.filter((pl.col("date").dt.time() >= pl.time(*start_time)) &(pl.col("date").dt.time() <= pl.time(*end_time)))
    if avoid_dict is not None:
        lf = apply_avoid_tickers_polars(lf, avoid_dict)
    symbols = (lf.select("symbol").unique().collect().get_column("symbol").to_list())
    print(f"Symbols to process: {len(symbols)}")

    for i, sym in enumerate(symbols, start=1):
        print(f"{i}/{len(symbols)} - {sym}")
        df_sym = (lf.filter(pl.col("symbol") == sym).collect().to_pandas())
        df_sym_1min = resample_symbol_day_to_1min_window(df_symbol=df_sym,start_time=start_time,end_time=end_time)
        df_sym_1min.to_parquet(output_dir / f"{sym}{output_suffix}",index=False)
        del df_sym, df_sym_1min
        gc.collect()

    print(f"Done. Files saved in: {output_dir}")
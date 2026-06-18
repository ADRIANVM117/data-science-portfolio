import pandas as pd 
import numpy as np 
def resample_symbol_by_day_to_1min(df_symbol):
    df_symbol = df_symbol.sort_values("date").copy()
    df_symbol["trading_day"] = df_symbol["date"].dt.date

    out = []

    for _, df_day in df_symbol.groupby("trading_day", observed=True):
        symbol = df_day["symbol"].iloc[0]
        market_cap = df_day["market_cap"].iloc[0]
        sic_description = df_day["sic_description"].iloc[0]
        company_name = df_day["company_name"].iloc[0]

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

        price_cols = ["open", "high", "low", "close", "vwap"]
        flow_cols = ["volume", "transactions"]

        resampled[price_cols] = resampled[price_cols].ffill()
        resampled[flow_cols] = resampled[flow_cols].fillna(0)

        resampled["symbol"] = symbol
        resampled["market_cap"] = market_cap
        resampled["sic_description"] = sic_description
        resampled["company_name"] = company_name

        out.append(resampled.reset_index())

    return pd.concat(out, ignore_index=True)
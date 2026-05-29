import requests
import pandas as pd


def get_intraday_prices(
    symbol: str = "SPY",
    api_key: str = "TU_API_KEY",
    interval: str = "5min",
    outputsize: str = "full",
    adjusted: str = "true",
    extended_hours: str = "false",
    month: str | None = None
) -> pd.DataFrame:
    """
    Download intraday OHLCV data from Alpha Vantage.

    If month is provided, it downloads intraday data for that specific month.
    Example: month="2026-05"
    """

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "adjusted": adjusted,
        "extended_hours": extended_hours,
        "apikey": api_key
    }

    if month is not None:
        params["month"] = month

    response = requests.get(url, params=params)
    data = response.json()

    key = f"Time Series ({interval})"

    if key not in data:
        raise ValueError(
            f"API response does not contain '{key}'. Response: {data}"
        )

    df = pd.DataFrame.from_dict(
        data[key],
        orient="index"
    )

    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume"
    })

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = df.astype({
        "Open": float,
        "High": float,
        "Low": float,
        "Close": float,
        "Volume": int
    })

    df["symbol"] = symbol

    if month is not None:
        df["source_month"] = month

    return df

######
def get_intraday_prices_multiple_months(
    symbol: str,
    api_key: str,
    months: list[str],
    interval: str = "5min",
    adjusted: str = "true",
    extended_hours: str = "false"
) -> pd.DataFrame:
    """
    Download and concatenate Alpha Vantage intraday data
    across multiple months.
    """

    dfs = []

    for month in months:
        print(f"Downloading {symbol} {interval} data for {month}...")

        df_month = get_intraday_prices(
            symbol=symbol,
            api_key=api_key,
            interval=interval,
            outputsize="full",
            adjusted=adjusted,
            extended_hours=extended_hours,
            month=month
        )

        dfs.append(df_month)

    df_all = pd.concat(dfs).sort_index()

    df_all = df_all[~df_all.index.duplicated(keep="last")]

    return df_all
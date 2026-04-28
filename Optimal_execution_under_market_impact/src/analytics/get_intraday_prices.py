import requests
import pandas as pd
def get_intraday_prices(
    symbol: str = "SPY",
    api_key: str = "TU_API_KEY",
    interval: str = "5min",
    outputsize: str = "compact",
    adjusted: str = "true",
    extended_hours: str = "false"
) -> pd.DataFrame:
    """
    Download intraday OHLCV data from Alpha Vantage.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. "SPY", "AAPL", "MSFT".
    api_key : str
        Alpha Vantage API key.
    interval : str
        Intraday interval: "1min", "5min", "15min", "30min", "60min".
    outputsize : str
        "compact" returns latest 100 rows.
        "full" returns more historical intraday data when available.
    adjusted : str
        "true" or "false".
    extended_hours : str
        "true" includes pre/post-market.
        "false" keeps regular hours only.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.
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

    response = requests.get(url, params=params)
    data = response.json()

    key = f"Time Series ({interval})"

    if key not in data:
        raise ValueError(f"API response does not contain '{key}'. Response: {data}")

    df = pd.DataFrame.from_dict(data[key], orient="index")

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

    return df
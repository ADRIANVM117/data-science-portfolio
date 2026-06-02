import requests
import pandas as pd
import ast
import time

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


def safe_parse_list(x):
    """
    Polymarket sometimes returns list-like fields as strings.
    This converts them safely into Python lists.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def fetch_polymarket_markets(limit=500, offset=0, closed=True):
    """
    Fetch markets from Polymarket Gamma API.
    Gamma API is public and does not require authentication.
    """
    url = f"{GAMMA_BASE}/markets"
    params = {
        "limit": limit,
        "offset": offset,
        "closed": str(closed).lower()
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def fetch_many_markets(max_pages=50, limit=500, closed=True):
    all_markets = []

    for page in range(max_pages):
        offset = page * limit

        try:
            markets = fetch_polymarket_markets(
                limit=limit,
                offset=offset,
                closed=closed
            )
        except requests.exceptions.HTTPError as e:
            print(f"Stopping at page={page}, offset={offset}. Error: {e}")
            break

        if not markets:
            break

        all_markets.extend(markets)
        print(f"Downloaded page {page + 1}, total markets: {len(all_markets)}")

        time.sleep(0.25)

    return pd.DataFrame(all_markets)




def filter_fed_rates_markets(df):
    """
    Keep markets related to Fed, rates, rate cuts, FOMC, inflation, CPI.
    """
    keywords = [
        "fed",
        "federal reserve",
        "fomc",
        "rate",
        "rates",
        "interest rate",
        "rate cut",
        "rate hike",
        "inflation",
        "cpi",
        "powell"
    ]

    text_cols = ["question", "title", "description", "slug"]

    available_cols = [c for c in text_cols if c in df.columns]

    text = (
        df[available_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    mask = text.apply(lambda x: any(k in x for k in keywords))

    return df.loc[mask].copy()


def extract_yes_token_id(row):
    """
    Extract CLOB token ID for YES outcome.
    """
    outcomes = safe_parse_list(row.get("outcomes", []))
    token_ids = safe_parse_list(row.get("clobTokenIds", []))

    if not outcomes or not token_ids:
        return None

    for outcome, token_id in zip(outcomes, token_ids):
        if str(outcome).lower() in ["yes", "y"]:
            return token_id

    return token_ids[0]


def get_price_history(clob_token_id, start_ts=None, end_ts=None, interval=None, fidelity_minutes=60):
    url = f"{CLOB_BASE}/prices-history"

    params = {
        "market": str(clob_token_id)
    }

    if start_ts is not None:
        params["startTs"] = int(start_ts)

    if end_ts is not None:
        params["endTs"] = int(end_ts)

    if interval is not None:
        params["interval"] = interval
    else:
        params["fidelity"] = fidelity_minutes

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        print("Bad request:", response.url)
        print("Response:", response.text[:300])
        return pd.DataFrame(columns=["timestamp", "implied_probability"])

    data = response.json().get("history", [])

    if not data:
        return pd.DataFrame(columns=["timestamp", "implied_probability"])

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["implied_probability"] = df["p"].astype(float)

    return df[["timestamp", "implied_probability"]].sort_values("timestamp")




def get_price_history_chunked(
    clob_token_id,
    start_ts,
    end_ts,
    fidelity_minutes=60,
    chunk_days=7,        # Reducido a 7 días para evitar el bloqueo del servidor
    sleep_seconds=0.15   # Incrementado ligeramente para evitar Rate Limiting
):
    """
    Fetch price history by splitting the query into micro time chunks.
    Bypasses the strict Polymarket CLOB max-interval limit.
    """
    url = f"{CLOB_BASE}/prices-history"

    start_dt = pd.to_datetime(start_ts, unit="s", utc=True)
    end_dt = pd.to_datetime(end_ts, unit="s", utc=True)

    all_chunks = []
    current_start = start_dt
    chunk_delta = pd.Timedelta(days=chunk_days)

    while current_start < end_dt:
        current_end = min(current_start + chunk_delta, end_dt)

        # Estructura limpia de parámetros: enviamos estrictamente lo necesario
        params = {
            "market": str(clob_token_id),
            "startTs": int(current_start.timestamp()),
            "endTs": int(current_end.timestamp()),
            "fidelity": int(fidelity_minutes)
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json().get("history", [])
                if data:
                    all_chunks.extend(data)
            else:
                print(
                    f"Skipping chunk {current_start.date()} to {current_end.date()} "
                    f"| HTTP {response.status_code} | {response.text[:150]}"
                )
        except Exception as e:
            print(f"Network error on chunk {current_start.date()}: {e}")

        current_start = current_end
        time.sleep(sleep_seconds)

    if not all_chunks:
        return pd.DataFrame(columns=["timestamp", "implied_probability"])

    # Procesamiento y limpieza del dataset unificado
    df = pd.DataFrame(all_chunks)
    df = df.drop_duplicates(subset=["t"])

    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df["implied_probability"] = df["p"].astype(float)

    # Retorno limpio de DataFrame (Corregido: sin coma al final)
    return df[["timestamp", "implied_probability"]].sort_values("timestamp").reset_index(drop=True)





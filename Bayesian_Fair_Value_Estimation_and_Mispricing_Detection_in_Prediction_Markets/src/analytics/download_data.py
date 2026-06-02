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


def fetch_many_markets(max_pages=10, limit=500, closed=True):
    """
    Fetch multiple pages of Polymarket markets.
    """
    all_markets = []

    for page in range(max_pages):
        offset = page * limit
        markets = fetch_polymarket_markets(
            limit=limit,
            offset=offset,
            closed=closed
        )

        if not markets:
            break

        all_markets.extend(markets)
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


def get_price_history(clob_token_id, fidelity_minutes=60):
    """
    Fetch historical implied probability for a YES token.
    CLOB prices-history is a public endpoint.
    """
    url = f"{CLOB_BASE}/prices-history"

    params = {
        "market": clob_token_id,
        "fidelity": fidelity_minutes
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json().get("history", [])

    if not data:
        return pd.DataFrame(columns=["timestamp", "implied_probability"])

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s")
    df["implied_probability"] = df["p"].astype(float)

    return df[["timestamp", "implied_probability"]].sort_values("timestamp")

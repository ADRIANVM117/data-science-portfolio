import json
import requests

GAMMA_URL = "https://gamma-api.polymarket.com/markets"


def get_top_market_tokens(limit: int = 20) -> list[str]:
    """
    Obtiene los mercados más líquidos y devuelve
    los token IDs YES y NO.
    """

    response = requests.get(
        GAMMA_URL,
        params={
            "active": True,
            "closed": False,
            "order": "liquidity",
            "ascending": False,
            "limit": limit,
        },
        timeout=30,
    )

    response.raise_for_status()

    markets = response.json()

    asset_ids = []

    for market in markets:

        token_ids = market.get("clobTokenIds")

        if not token_ids:
            continue

        try:

            # Gamma devuelve:
            # '["12345","67890"]'
            # NO:
            # ["12345","67890"]

            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)

            yes_token = str(token_ids[0])
            no_token = str(token_ids[1])

            asset_ids.append(yes_token)
            asset_ids.append(no_token)

        except Exception as e:
            print(
                f"Error procesando mercado "
                f"{market.get('question', 'UNKNOWN')}: {e}"
            )
            continue

    return asset_ids
"""Explicit serial Alpha Vantage boundary with bounded, sanitized telemetry."""
from __future__ import annotations

from dataclasses import dataclass
import time
import requests

from .catalog import is_downloadable
from .ingest import ProviderResponseError, api_key_from_environment, decode_raw

API_URL = "https://www.alphavantage.co/query"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

@dataclass(frozen=True)
class ProviderTelemetry:
    attempts: int; duration_seconds: float; http_status: int | None; stage: str; classification: str

class ProviderRequestError(ProviderResponseError):
    def __init__(self, telemetry: ProviderTelemetry):
        super().__init__(f"provider request failed: {telemetry.classification}"); self.telemetry = telemetry

def _failure(error: requests.RequestException) -> tuple[str, str]:
    if isinstance(error, requests.exceptions.ProxyError): return "PROXY", "PROXY_ERROR"
    if isinstance(error, requests.exceptions.SSLError): return "TLS", "TLS_ERROR"
    if isinstance(error, requests.exceptions.ConnectTimeout): return "CONNECT", "CONNECT_TIMEOUT"
    if isinstance(error, requests.exceptions.ReadTimeout): return "READ", "READ_TIMEOUT"
    if isinstance(error, requests.exceptions.ConnectionError): return "CONNECT", "CONNECTION_ERROR"
    return "TRANSPORT", "TRANSPORT_ERROR"

def _provider_response(*, params: dict[str, str], attempts: int = 3, connect_timeout: float = 8.0, read_timeout: float = 15.0, max_operation_seconds: float = 45.0) -> tuple[requests.Response, ProviderTelemetry]:
    """Bound attempts and each connect/read wait; read timeout is not treated as a total deadline."""
    if attempts < 1 or min(connect_timeout, read_timeout, max_operation_seconds) <= 0: raise ValueError("invalid provider timeout configuration")
    request_params = {**params, "apikey": api_key_from_environment()}; started = time.monotonic(); last = None
    for attempt in range(1, attempts + 1):
        if time.monotonic() - started >= max_operation_seconds: break
        try:
            response = requests.get(API_URL, params=request_params, timeout=(connect_timeout, read_timeout)); elapsed = time.monotonic() - started
            if response.status_code in TRANSIENT_STATUS_CODES:
                last = ProviderTelemetry(attempt, elapsed, response.status_code, "HTTP", "HTTP_TRANSIENT")
                if attempt < attempts and elapsed + 2 ** (attempt - 1) < max_operation_seconds: time.sleep(2 ** (attempt - 1)); continue
                raise ProviderRequestError(last)
            if response.status_code >= 400: raise ProviderRequestError(ProviderTelemetry(attempt, elapsed, response.status_code, "HTTP", "HTTP_CLIENT_ERROR"))
            return response, ProviderTelemetry(attempt, elapsed, response.status_code, "RESPONSE", "HTTP_SUCCESS")
        except ProviderRequestError: raise
        except requests.RequestException as error:
            stage, classification = _failure(error); elapsed = time.monotonic() - started; last = ProviderTelemetry(attempt, elapsed, None, stage, classification)
            if attempt < attempts and elapsed + 2 ** (attempt - 1) < max_operation_seconds: time.sleep(2 ** (attempt - 1)); continue
    raise ProviderRequestError(last or ProviderTelemetry(0, time.monotonic()-started, None, "DEADLINE", "OPERATION_DEADLINE"))

def capture_plan(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    symbols: set[str] = set(); plan = []
    for row in rows:
        symbol = row["provider_symbol_candidate"].strip().upper()
        if is_downloadable(row) and symbol not in symbols: symbols.add(symbol); plan.append(row)
    return plan

def fetch_daily_adjusted(symbol: str) -> bytes:
    response, _ = _provider_response(params={"function":"TIME_SERIES_DAILY_ADJUSTED", "symbol":symbol, "outputsize":"full"})
    raw = response.content
    if any(key in decode_raw(raw) for key in ("Error Message", "Information", "Note")): raise ProviderResponseError("provider returned an unusable response")
    return raw

def search_symbol_with_telemetry(query: str) -> tuple[list[dict[str, str]], ProviderTelemetry]:
    response, telemetry = _provider_response(params={"function":"SYMBOL_SEARCH", "keywords":query})
    try: result = response.json()
    except ValueError: raise ProviderResponseError("provider returned non-JSON search response") from None
    if "Note" in result or "Information" in result: raise ProviderRequestError(ProviderTelemetry(telemetry.attempts, telemetry.duration_seconds, telemetry.http_status, "PROVIDER", "PROVIDER_THROTTLE_OR_INFO"))
    if "Error Message" in result: raise ProviderRequestError(ProviderTelemetry(telemetry.attempts, telemetry.duration_seconds, telemetry.http_status, "PROVIDER", "PROVIDER_ERROR"))
    matches = result.get("bestMatches")
    if not isinstance(matches, list): raise ProviderRequestError(ProviderTelemetry(telemetry.attempts, telemetry.duration_seconds, telemetry.http_status, "PROVIDER", "PROVIDER_SCHEMA_ERROR"))
    fields = {"1. symbol":"provider_symbol", "2. name":"provider_name", "3. type":"provider_type", "4. region":"provider_region", "8. currency":"provider_currency", "9. matchScore":"match_score"}
    return ([{public:str(match.get(source,"")) for source,public in fields.items()} for match in matches if isinstance(match,dict)], telemetry)

def search_symbol(query: str) -> list[dict[str, str]]: return search_symbol_with_telemetry(query)[0]

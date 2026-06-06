from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Awaitable

import websockets

# CORREGIDO: Importación directa desde tu archivo local order_book.py
from order_book import OrderBook

logger = logging.getLogger(__name__)

POLYMARKET_MARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketMarketDataClient:
    """
    Async Polymarket market-data WebSocket client corregido y optimizado.
    """

    def __init__(
        self,
        asset_ids: list[str],
        on_book_update: Callable[[OrderBook], Awaitable[None]] | None = None,
    ) -> None:
        if not asset_ids:
            raise ValueError("asset_ids cannot be empty.")

        self.asset_ids = asset_ids
        self.books: dict[str, OrderBook] = {
            asset_id: OrderBook(asset_id=asset_id) for asset_id in asset_ids
        }
        self.on_book_update = on_book_update
        self._running = False

    async def connect_forever(self) -> None:
        """Connect with automatic reconnect and exponential backoff."""
        self._running = True
        backoff = 1.0

        while self._running:
            try:
                async with websockets.connect(
                    POLYMARKET_MARKET_WS,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=10_000,
                ) as ws:
                    logger.info("Connected to Polymarket market WebSocket.")
                    await self._subscribe(ws)
                    backoff = 1.0

                    async for raw_message in ws:
                        if not self._running:
                            break
                        await self._handle_raw_message(raw_message)

            except asyncio.CancelledError:
                logger.info("WebSocket client task cancelled.")
                raise
            except Exception as exc:
                logger.exception("WebSocket error: %s. Reconnecting...", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def stop(self) -> None:
        self._running = False

    async def _subscribe(self, ws) -> None:
        payload = {
            "asset_ids": self.asset_ids,
            "type": "market",
        }
        await ws.send(json.dumps(payload))
        logger.info("Subscribed to %d asset_ids.", len(self.asset_ids))

    async def _handle_raw_message(self, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON message: %s", raw_message)
            return

        if isinstance(message, list):
            for item in message:
                await self._handle_message(item)
        else:
            await self._handle_message(message)

    async def _handle_message(self, message: dict) -> None:
        event_type = message.get("event_type")
        asset_id = message.get("asset_id")

        if not asset_id and "market" not in message:
            return

        if event_type == "book":
            book = self.books.setdefault(asset_id, OrderBook(asset_id=asset_id))
            book.update_from_snapshot(message)
            await self._emit(book)

        elif event_type == "price_change":
            if not asset_id:
                return
            
            book = self.books.setdefault(asset_id, OrderBook(asset_id=asset_id))
            for change in message.get("price_changes", []):
                book.apply_price_change(change)
            await self._emit(book)

        elif event_type in {"last_trade_price", "best_bid_ask"}:
            logger.debug("Market event: %s | %s", event_type, message)
        else:
            logger.debug("Unhandled event_type=%s message=%s", event_type, message)

    async def _emit(self, book: OrderBook) -> None:
        if self.on_book_update is not None:
            await self.on_book_update(book)

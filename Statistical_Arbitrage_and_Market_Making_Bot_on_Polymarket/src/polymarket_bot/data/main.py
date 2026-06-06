import asyncio
import logging
from websocket_client import PolymarketMarketDataClient
from order_book import OrderBook

# Configuración de logs visibles
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def mi_estrategia_callback(book: OrderBook) -> None:
    """
    Esta función es el puente: el WebSocket recibe los datos, actualiza el
    OrderBook interno y te entrega el estado limpio aquí listo para operar.
    """
    print(f" Cambio detectado para Asset: {book.asset_id[:8]}...")
    print(f"   Best Bid: {book.best_bid.price if book.best_bid else 'N/A'}")
    print(f"   Best Ask: {book.best_ask.price if book.best_ask else 'N/A'}")
    print(f"   Spread: {book.spread} | Mid Price: {book.mid_price}")

async def main():
    # ID del mercado (ejemplo activo de Polymarket)
    asset_id = "21680511520182414704332517616149176378453443171308311245037613580554217147746"
    
    # Unimos el cliente con el callback que procesa el OrderBook
    bot_client = PolymarketMarketDataClient(
        asset_ids=[asset_id],
        on_book_update=mi_estrategia_callback
    )
    
    try:
        await bot_client.connect_forever()
    except KeyboardInterrupt:
        print("\nDeteniendo el bot de forma segura...")
        await bot_client.stop()

if __name__ == "__main__":
    asyncio.run(main())

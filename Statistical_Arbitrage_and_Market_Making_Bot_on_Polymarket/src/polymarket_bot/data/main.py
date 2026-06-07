import asyncio
import logging
from websocket_client import PolymarketMarketDataClient
from order_book import OrderBook
from market_discovery import get_top_market_tokens
from features import compute_book_features
#Configuración de logs limpia y optimizada para producción
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class QuantTradingEngine:
    def __init__(self, asset_ids: list[str]):

        self.asset_ids = asset_ids
        # Cola asíncrona intermedia para transferir snapshots del libro sin bloquear el WS
        self.queue: asyncio.Queue[OrderBook] = asyncio.Queue(maxsize=5000)
        self.client = PolymarketMarketDataClient(asset_ids=self.asset_ids, on_book_update=self._enqueue_book)
        self._strategy_task: asyncio.Task | None = None

    async def _enqueue_book(self, book: OrderBook) -> None:
        """Callback del productor: Recibe el libro actualizado y lo empuja a la cola."""
        try:
            # put_nowait es ultra rápido y no bloquea el hilo de lectura del WebSocket
            self.queue.put_nowait(book)
        except asyncio.QueueFull:
            logger.warning("Cola llena (maxsize superado). Descartando actualización vieja.")

    async def strategy_worker(self) -> None:
        """Consumidor: Hilo lógico independiente dedicado a cálculos matemáticos y trading."""
        logger.info("Hilo de ejecución de estrategia de trading inicializado.")
        while True:
            try:
                book = await self.queue.get()
                features = compute_book_features(book)
                if features is None:
                    self.queue.task_done()
                continue
            # --- FASE 2: MARKET MICROSTRUCTURE FEATURES ---
                print(f"\n[Features] Asset: {features.asset_id[:10]}...")
                print(f"   Bid/Ask: {features.best_bid} / {features.best_ask}")
                print(f"   Spread: {features.spread} | Mid: {features.mid_price}")
                print(f"   Imbalance: {features.book_imbalance}")
                print(f"   Microprice: {features.microprice}")
                print(f"   Top Depth: {features.top_depth}")

                self.queue.task_done()

            except asyncio.CancelledError:
                logger.info("Hilo de estrategia cancelado de forma segura.")
                break

            except Exception as e:
                logger.error("Error crítico ejecutando el modelo en el worker: %s", e)

    async def start(self) -> None:
        """Inicia todas las tareas concurrentes del motor de forma segura."""
        # Lanzamos el consumidor (estrategia) en paralelo
        self._strategy_task = asyncio.create_task(self.strategy_worker())
        
        # Lanzamos el productor (WebSocket)
        await self.client.connect_forever()

    async def stop(self) -> None:
        """Detiene de forma controlada el WebSocket y las tareas de procesamiento."""
        logger.info("Iniciando secuencia de apagado seguro...")
        await self.client.stop()
        if self._strategy_task:
            self._strategy_task.cancel()
            await asyncio.gather(self._strategy_task, return_exceptions=True)
        logger.info("Motor de trading totalmente detenido.")

async def main():
    # Token YES de un mercado con alto volumen y trading constante
    
    mercados_activos = get_top_market_tokens(limit=20)
    


    engine = QuantTradingEngine(asset_ids=mercados_activos)

    
    try:
        await engine.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import logging
from websocket_client import PolymarketMarketDataClient
from order_book import OrderBook

# Configuración de logs limpia y optimizada para producción
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
                
                # --- AQUÍ INICIA TU FASE 2: MODELADO ESTADÍSTICO / MARKET MAKING ---
                # Este bloque simula tus algoritmos de arbitraje o cálculo de valor teórico
                print(f"\n[Estrategia] Procesando Asset: {book.asset_id[:10]}...")
                print(f"   Best Bid: {book.best_bid.price if book.best_bid else 'N/A'}")
                print(f"   Best Ask: {book.best_ask.price if book.best_ask else 'N/A'}")
                print(f"   Spread: {book.spread if book.spread else 'N/A'} | Mid Price: {book.mid_price if book.mid_price else 'N/A'}")
                
                # Notificar a la cola que la tarea asociada al objeto ha sido completada
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
    mercados_activos = [
        "22046754593467471415174021272023581561726002341257416393181467440628329621347"
    ]
    
    engine = QuantTradingEngine(asset_ids=mercados_activos)

    
    try:
        await engine.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())

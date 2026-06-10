import asyncio
import logging
from websocket_client import PolymarketMarketDataClient
from order_book import OrderBook
from market_discovery import get_top_market_tokens
from features import compute_book_features
from state_manager import MarketState
from signal_engine import generate_signal, SignalType
#Configuración de logs limpia y optimizada para producción
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class QuantTradingEngine:
    def __init__(self, asset_ids: list[str]):

        self.asset_ids = asset_ids
        self.state = MarketState()
        # Cola asíncrona intermedia para transferir snapshots del libro sin bloquear el WS
        self.queue: asyncio.Queue[OrderBook] = asyncio.Queue(maxsize=5000)
        self.client = PolymarketMarketDataClient(asset_ids=self.asset_ids, on_book_update=self._enqueue_book)
        self._strategy_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None

    

    async def _enqueue_book(self, book: OrderBook) -> None:
        """Callback del productor: Recibe el libro actualizado y lo empuja a la cola."""
        try:
            # put_nowait es ultra rápido y no bloquea el hilo de lectura del WebSocket
            self.queue.put_nowait(book)
        except asyncio.QueueFull:
            logger.warning("Cola llena (maxsize superado). Descartando actualización vieja.")

    async def strategy_worker(self) -> None:
        """Consume live books, compute features, and update market state."""
        logger.info("Hilo de ejecución de estrategia de trading inicializado.")

        while True:
            try:
                book = await self.queue.get()
                features = compute_book_features(book)

                if features is not None:
                    self.state.update(features)

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
        self._monitor_task = asyncio.create_task(self.monitor_worker(interval_seconds=5))
        
        # Lanzamos el productor (WebSocket)
        await self.client.connect_forever()

    async def monitor_worker(self, interval_seconds: int = 5) -> None:
        """Print periodic market microstructure summaries."""
        logger.info("Monitor de microestructura inicializado.")
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                
                tradable_features = self.state.tradable_features()

                if not tradable_features:
                    logger.info("Monitor: aún no hay activos tradables.")
                    continue

                print("\n" + "_" * 80)
                print("TRADABLE MARKET MICROSTRUCTURE SNAPSHOT")
                print("_" * 80)
                print("\nTop Tradable Spreads:")
                for f in self.state.top_tradable_spreads(n=5):
                    print(
                        f"Asset {f.asset_id[:10]} | "
                        f"Spread={f.spread} | "
                        f"Mid={f.mid_price} | "
                        f"Depth={f.top_depth}")
                
                print("\nTop Tradable Buy Imbalance:")
                for f in self.state.top_tradable_buy_imbalance(n=5):
                    print(
                        f"Asset {f.asset_id[:10]} | "
                        f"Imbalance={f.book_imbalance:.4f} | "
                        f"Microprice={f.microprice:.4f} | "
                        f"Mid={f.mid_price}")

                print("\nTop Tradable Sell Imbalance:")
                for f in self.state.top_tradable_sell_imbalance(n=5):
                    print(
                        f"Asset {f.asset_id[:10]} | "
                        f"Imbalance={f.book_imbalance:.4f} | "
                        f"Microprice={f.microprice:.4f} | "
                        f"Mid={f.mid_price}")

                print("\nTop Tradable Depth:")
                for f in self.state.top_tradable_depth(n=5):
                    print(
                        f"Asset {f.asset_id[:10]} | "
                        f"Depth={f.top_depth} | "
                        f"Spread={f.spread} | "
                        f"Mid={f.mid_price}")

                # SIGNAL ENGINE
                signals = [ generate_signal(f) for f in tradable_features ]
                print("\nSignal Diagnostics:")
                for s in signals[:10]:
                    print(f"Asset {s.asset_id[:10]} | "
                      f"Signal={s.signal.value} | "
                      f"Edge={s.microprice_edge:.5f} | "
                      f"Imbalance={s.imbalance:.4f}")

                active_signals = [ s for s in signals if s.signal != SignalType.NEUTRAL ]

                print("\nActive Signals:")

                if not active_signals:
                    print("No active signals.")
                else:
                    for s in active_signals[:10]:
                        print(
                        f"Asset {s.asset_id[:10]} | "
                        f"Signal={s.signal.value} | "
                        f"Edge={s.microprice_edge:.5f} | "
                        f"Imbalance={s.imbalance:.4f} | "
                        f"Spread={s.spread} | "
                        f"Depth={s.depth}"
                    )

            except asyncio.CancelledError: 
                logger.info("Monitor de microestructura cancelado de forma segura.")
                break

    
    async def stop(self) -> None:
        """Detiene de forma controlada el WebSocket y las tareas de procesamiento."""
        logger.info("Iniciando secuencia de apagado seguro...")

        await self.client.stop()

        tasks = []
        
        if self._strategy_task:
            self._strategy_task.cancel()
            tasks.append(self._strategy_task)

        if self._monitor_task:
            self._monitor_task.cancel()
            tasks.append(self._monitor_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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

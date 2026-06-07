import asyncio
import websockets

async def test():

    print("Attempting connection...")

    async with websockets.connect(
        "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        open_timeout=30,
    ) as ws:

        print("CONNECTED!")

        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(test())
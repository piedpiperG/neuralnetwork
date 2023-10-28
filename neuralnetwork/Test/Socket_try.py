import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print(f"Received message: {message}")
        await websocket.send(message)

start_server = websockets.serve(echo, "127.0.0.1", 3404)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

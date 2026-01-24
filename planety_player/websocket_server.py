import asyncio
import json
import threading
import time
from typing import Optional, Set

import websockets


class WebSocketServer:
    """WebSocket server to broadcast tracking data."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server = None
        self._thread: Optional[threading.Thread] = None

    async def handler(self, websocket):
        """Handle new WebSocket connection."""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True,
            )

    def send(self, data: dict):
        """Thread-safe method to send data."""
        if self.loop and self.clients:
            message = json.dumps(data)
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    async def _run_server(self):
        """Run the WebSocket server."""
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server running on ws://{self.host}:{self.port}")
            await asyncio.Future()

    def _thread_target(self):
        """Thread target for running asyncio loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    def start(self):
        """Start the WebSocket server in a background thread."""
        self._thread = threading.Thread(target=self._thread_target, daemon=True)
        self._thread.start()
        time.sleep(0.5)

    def stop(self):
        """Stop the WebSocket server."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

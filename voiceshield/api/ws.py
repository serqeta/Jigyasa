from __future__ import annotations

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState


class WebSocketManager:
    """Tracks live WebSocket connections and broadcasts to all."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]

    async def broadcast(self, data: dict) -> None:
        payload = json.dumps(data)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            # Skip sockets that aren't fully connected — sending to a closed/
            # closing socket raises inside uvicorn ("socket.send() raised
            # exception") and, over a churny link, that per-chunk noise adds
            # latency to the live loop. Drop them straight away instead.
            if ws.application_state != WebSocketState.CONNECTED:
                dead.append(ws)
                continue
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = WebSocketManager()


async def ws_risk_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; data is pushed via manager.broadcast
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

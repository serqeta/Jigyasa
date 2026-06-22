from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from voiceshield.api.rest import router as rest_router
from voiceshield.api.ws import manager, ws_risk_endpoint


def create_app(runner: Any | None = None) -> FastAPI:
    """
    FastAPI app factory.

    Args:
        runner: A PipelineRunner instance. If None, the app starts in idle
                mode (useful for tests that inject state directly).
    """
    app = FastAPI(title="VoiceShield", version="1.0.0")
    app.state.runner = runner

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(rest_router)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.websocket("/v1/ws/risk")
    async def ws_risk(websocket: WebSocket) -> None:
        await ws_risk_endpoint(websocket)

    # Serve the advisory dashboard as static files
    ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
    ui_dir = os.path.abspath(ui_dir)
    if os.path.isdir(ui_dir):
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    @app.on_event("startup")
    async def _start_pipeline() -> None:
        # Warm up numba JIT (librosa.yin compiles on first call, adding ~4s penalty)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _warmup_jit)
        if runner is None:
            return
        asyncio.create_task(_pipeline_loop(runner))

    return app


def _warmup_jit() -> None:
    """Call each librosa function once to trigger numba JIT before the first request."""
    import numpy as np

    from voiceshield.pipeline.runner import _extract_features

    _extract_features(np.zeros(8000, dtype=np.float32))


async def _pipeline_loop(runner: Any) -> None:
    """Background task: run one chunk per 500 ms and broadcast to WebSocket clients."""
    import asyncio

    while True:
        loop = asyncio.get_event_loop()
        try:
            entry = await loop.run_in_executor(None, runner.run_once)
            await manager.broadcast(entry.to_dict())
        except EOFError:
            break
        except Exception:
            pass
        await asyncio.sleep(0)  # yield; the run_in_executor already waited ~500 ms

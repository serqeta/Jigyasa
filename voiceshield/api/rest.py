from __future__ import annotations

import asyncio
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/v1")


@router.get("/risk/current")
async def get_current(request: Request) -> JSONResponse:
    runner = request.app.state.runner
    entry = runner.timeline.latest()
    if entry is None:
        raise HTTPException(status_code=503, detail="No data yet")
    return JSONResponse(entry.to_dict())


@router.get("/risk/timeline")
async def get_timeline(request: Request) -> JSONResponse:
    runner = request.app.state.runner
    return JSONResponse(runner.timeline.to_json())


@router.post("/analyze")
async def analyze_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a WAV file and run the full VoiceShield pipeline synchronously."""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_analysis, tmp_path)
        return JSONResponse(result)
    finally:
        os.unlink(tmp_path)


_jit_warmed = False


def _run_analysis(tmp_path: str) -> dict:
    global _jit_warmed
    from voiceshield.audio.source import FileSource
    from voiceshield.classifier.fallback import FallbackScorer
    from voiceshield.pipeline.runner import PipelineRunner, _extract_features

    if not _jit_warmed:
        import numpy as np

        _extract_features(np.zeros(8000, dtype=np.float32))
        _jit_warmed = True

    source = FileSource(tmp_path)
    scorer = FallbackScorer()
    runner = PipelineRunner(source, scorer)

    entries: list = []
    runner.run_forever(entries.append)

    mean_score = sum(e.score for e in entries) / len(entries) if entries else 0.0
    return {
        "entries": [e.to_dict() for e in entries],
        "summary": {
            "total_chunks": len(entries),
            "duration_s": round(entries[-1].time, 2) if entries else 0.0,
            "first_amber_t": runner.state_engine.first_amber_t,
            "first_red_t": runner.state_engine.first_red_t,
            "final_state": (
                runner.state_engine._current.value
                if any(e.state != "grey" for e in entries)
                else "grey"
            ),
            "mean_score": round(mean_score, 4),
        },
    }


@router.post("/reset")
async def reset_runner(request: Request) -> JSONResponse:
    """Reset the live pipeline's state engine and timeline."""
    runner = request.app.state.runner
    if runner is None:
        raise HTTPException(status_code=503, detail="No active runner")
    runner.reset()
    return JSONResponse({"status": "reset"})

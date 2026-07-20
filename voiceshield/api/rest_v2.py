"""
/v2/ API (Stage 2): ensemble scoring, replay detection, evidence export.

/v1/ remains frozen for Stage 1 clients; everything ensemble-aware lives
here. The ensemble is loaded lazily on first use and cached for the
process lifetime (the HF models take seconds to load).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from voiceshield import config

router = APIRouter(prefix="/v2")

_ensemble = None
_ensemble_lock = threading.Lock()


def _get_ensemble():
    global _ensemble
    if _ensemble is None:
        with _ensemble_lock:
            if _ensemble is None:
                from voiceshield.classifier import get_scorers

                _ensemble = get_scorers()
    return _ensemble


@router.get("/models")
async def get_models() -> JSONResponse:
    """Loaded ensemble components and their fusion weights."""
    ensemble = _get_ensemble()
    components = {
        name: {
            "type": type(scorer).__name__,
            "model_id": getattr(scorer, "model_id", None),
            "fusion_weight": config.FUSION_WEIGHTS.get(name),
        }
        for name, scorer in ensemble.items()
    }
    components["replay"] = {
        "type": "ReplayDetector",
        "model_id": None,
        "fusion_weight": config.FUSION_WEIGHTS.get("replay"),
        "enabled": config.ENABLE_REPLAY_DETECTION,
    }
    return JSONResponse({"components": components, "device": config.get_device()})


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
    """Upload an audio file and run the full Stage 2 ensemble on it."""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _run_ensemble_analysis, tmp_path)
        return JSONResponse(result)
    finally:
        os.unlink(tmp_path)


def _run_ensemble_analysis(tmp_path: str) -> dict:
    from voiceshield.audio.source import FileSource
    from voiceshield.pipeline.runner import PipelineRunner

    runner = PipelineRunner(FileSource(tmp_path), ensemble=_get_ensemble())
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
            "component_scores": entries[-1].component_scores if entries else {},
        },
    }


@router.post("/evidence/export")
async def evidence_export(request: Request) -> JSONResponse:
    """
    Explicit save-for-audit (G10): package the current rolling buffer as
    evidence. Never triggered automatically.
    """
    runner = request.app.state.runner
    if runner is None:
        raise HTTPException(status_code=503, detail="No active runner")

    audio = runner.buffer.latest_seconds(config.BUFFER_SECONDS)
    if len(audio) == 0:
        raise HTTPException(status_code=409, detail="Rolling buffer is empty")

    latest = runner.timeline.latest()
    from voiceshield.evidence import export_evidence

    loop = asyncio.get_event_loop()
    manifest = await loop.run_in_executor(
        None,
        export_evidence,
        audio,
        runner.timeline.to_json(),
        latest.component_scores if latest else None,
    )
    return JSONResponse(manifest)

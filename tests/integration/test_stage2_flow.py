"""
Stage 2 integration: ensemble pipeline flow and /v2/ API surface.

Uses fast rule-based ensemble members so these tests stay CPU-cheap; the
real pretrained models are exercised by tests/integration/test_hf_models.py
(skipped when weights are not cached).
"""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import voiceshield.config as cfg
from voiceshield.api.app import create_app
from voiceshield.audio.source import FileSource
from voiceshield.classifier.fallback import FallbackScorer
from voiceshield.classifier.phase_pitch import PhasePitchScorer
from voiceshield.pipeline.runner import PipelineRunner

SR = cfg.SAMPLE_RATE


@pytest.fixture(autouse=True)
def _energy_vad(monkeypatch):
    # Synthetic sine proxies aren't classified as speech by the neural VAD;
    # pin these flow tests to the deterministic energy VAD.
    monkeypatch.setattr(cfg, "USE_SILERO_VAD", False)


def _speech_wav(tmp_path, seconds: float = 3.0) -> str:
    """Bursty speech stand-in: the SNR gate needs speech/silence contrast."""
    rng = np.random.default_rng(3)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    voiced = 0.4 * np.sin(2 * np.pi * 170 * t * (1 + 0.05 * np.sin(2 * np.pi * 4 * t)))
    gate = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(np.float32)  # ~70% duty cycle
    audio = (voiced * gate + 0.005 * rng.standard_normal(n)).astype(np.float32)
    p = str(tmp_path / "speech.wav")
    sf.write(p, audio, SR)
    return p


def _fast_ensemble() -> dict:
    return {"stage1": FallbackScorer(), "phase_pitch": PhasePitchScorer()}


def _run_ensemble(tmp_path) -> PipelineRunner:
    runner = PipelineRunner(FileSource(_speech_wav(tmp_path)), ensemble=_fast_ensemble())
    runner.run_forever(lambda e: None)
    return runner


def test_ensemble_entries_carry_component_scores(tmp_path):
    runner = _run_ensemble(tmp_path)
    entry = runner.timeline.latest()
    assert entry is not None
    # Fake ensemble here is {stage1, phase_pitch} — replay is a real learned
    # model added by get_scorers() only when models/replay_lora exists, not
    # auto-injected into arbitrary ensembles.
    assert entry.component_scores is not None
    assert set(entry.component_scores) >= {"stage1", "phase_pitch"}
    assert all(0.0 <= v <= 1.0 for v in entry.component_scores.values())


def test_single_scorer_mode_still_works(tmp_path):
    """Stage 1 compatibility: positional (source, scorer) construction."""
    runner = PipelineRunner(FileSource(_speech_wav(tmp_path)), FallbackScorer())
    entries = []
    runner.run_forever(entries.append)
    assert entries
    assert entries[-1].component_scores.get("stage1") is not None


def test_v2_risk_endpoints(tmp_path):
    runner = _run_ensemble(tmp_path)
    client = TestClient(create_app(runner=None))
    client.app.state.runner = runner

    current = client.get("/v2/risk/current")
    assert current.status_code == 200
    body = current.json()
    assert "component_scores" in body and "replay" in body

    timeline = client.get("/v2/risk/timeline")
    assert timeline.status_code == 200
    assert isinstance(timeline.json(), list)
    assert len(timeline.json()) >= 1

    # /v1/ contract untouched by Stage 2 fields being present
    v1 = client.get("/v1/risk/current")
    assert v1.status_code == 200
    assert {"time", "score", "state", "snr_db", "top_artifact"} <= set(v1.json())


def test_v2_evidence_export(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "EVIDENCE_DIR", str(tmp_path / "evidence"))
    runner = _run_ensemble(tmp_path)
    client = TestClient(create_app(runner=None))
    client.app.state.runner = runner

    resp = client.post("/v2/evidence/export")
    assert resp.status_code == 200
    manifest = resp.json()
    assert manifest["audio_sha256"]
    assert manifest["timeline"]
    assert (tmp_path / "evidence").exists()


def test_v2_evidence_requires_runner():
    client = TestClient(create_app(runner=None))
    assert client.post("/v2/evidence/export").status_code == 503

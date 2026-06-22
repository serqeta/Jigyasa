"""
TEST-INT.1, INT.2, INT.3: Integration tests wiring FileSource through
the full pipeline without requiring real speech fixtures.
"""

import numpy as np
import soundfile as sf

import voiceshield.config as cfg
from voiceshield.audio.source import FileSource
from voiceshield.classifier.fallback import FallbackScorer
from voiceshield.pipeline.runner import PipelineRunner
from voiceshield.pipeline.timeline import TimelineEntry

SR = cfg.SAMPLE_RATE


def _write_wav(tmp_path, name: str, audio: np.ndarray) -> str:
    p = str(tmp_path / name)
    sf.write(p, audio, SR)
    return p


def test_pipeline_produces_entries(tmp_path):
    """TEST-INT.1: FileSource → PipelineRunner → timeline has ≥1 entry with all fields."""
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav = _write_wav(tmp_path, "tone.wav", audio)

    source = FileSource(wav)
    runner = PipelineRunner(source, FallbackScorer())

    entries: list[TimelineEntry] = []
    runner.run_forever(entries.append)

    assert len(entries) >= 1
    e = entries[0]
    assert isinstance(e.time, float)
    assert isinstance(e.score, float)
    assert e.state in {"green", "amber", "red", "grey"}
    assert isinstance(e.snr_db, float)


def test_silent_pipeline_all_grey(tmp_path):
    """TEST-INT.2: Silent input → all chunks should be GREY (SNR < 8 dB)."""
    audio = np.zeros(5 * SR, dtype=np.float32)
    wav = _write_wav(tmp_path, "silent.wav", audio)

    source = FileSource(wav)
    runner = PipelineRunner(source, FallbackScorer())

    entries: list[TimelineEntry] = []
    runner.run_forever(entries.append)

    assert len(entries) > 0
    assert all(e.state == "grey" for e in entries), [e.state for e in entries]


def test_fallback_classifier_toggle(tmp_path, monkeypatch):
    """TEST-INT.3: USE_FALLBACK_CLASSIFIER=True path runs without exceptions."""
    monkeypatch.setattr(cfg, "USE_FALLBACK_CLASSIFIER", True)

    t = np.linspace(0, 1.0, SR, endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 300 * t)
    wav = _write_wav(tmp_path, "short.wav", audio)

    source = FileSource(wav)
    runner = PipelineRunner(source, FallbackScorer())

    entries: list[TimelineEntry] = []
    runner.run_forever(entries.append)

    assert len(entries) >= 1
    assert all(0.0 <= e.score <= 1.0 for e in entries)


def test_timeline_monotonic_times(tmp_path):
    """Timeline entries have strictly increasing times."""
    t = np.linspace(0, 3.0, 3 * SR, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 200 * t)
    wav = _write_wav(tmp_path, "tone200.wav", audio)

    source = FileSource(wav)
    runner = PipelineRunner(source, FallbackScorer())

    entries: list[TimelineEntry] = []
    runner.run_forever(entries.append)

    times = [e.time for e in entries]
    assert times == sorted(times)
    assert len(set(times)) == len(times)

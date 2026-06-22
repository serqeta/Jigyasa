"""
End-to-end acceptance tests — TEST-E2E10.1, 10.2, 10.3, 10.5.

Requires real audio fixtures in tests/fixtures/:
  genuine_male_16k.wav, tts_synthetic_16k.wav, cloned_voice_16k.wav, silent_16k.wav

Tests are skipped (not failed) when fixtures are absent so CI stays green without them.
"""

import time

import pytest

from voiceshield.audio.source import FileSource
from voiceshield.classifier.fallback import FallbackScorer
from voiceshield.pipeline.runner import PipelineRunner  # noqa: F401 used in latency test
from voiceshield.pipeline.timeline import TimelineEntry


def _run(wav_path: str) -> list[TimelineEntry]:
    source = FileSource(wav_path)
    scorer = FallbackScorer()
    runner = PipelineRunner(source, scorer)
    entries: list[TimelineEntry] = []
    runner.run_forever(entries.append)
    return entries


def _skip_if_missing(fixture):
    if fixture is None:
        pytest.skip("Fixture not present in tests/fixtures/")


def test_genuine_stays_green(genuine_male_wav):
    """TEST-E2E10.1: Genuine speech → mean score < 0.30, no RED chunks."""
    _skip_if_missing(genuine_male_wav)
    entries = _run(genuine_male_wav)
    assert entries, "No entries produced"
    mean_score = sum(e.score for e in entries) / len(entries)
    assert mean_score < 0.30, f"Mean score too high: {mean_score:.3f}"
    assert all(e.state != "red" for e in entries), "Genuine speech produced RED state"


def test_tts_goes_red(tts_synthetic_wav):
    """TEST-E2E10.2: TTS synthetic voice → first_red_t ≤ 10.0 s."""
    _skip_if_missing(tts_synthetic_wav)
    entries = _run(tts_synthetic_wav)
    first_red = next(
        (e.first_red_t for e in entries if e.first_red_t is not None), None
    )
    assert first_red is not None, "Pipeline never reached RED on TTS audio"
    assert first_red <= 10.0, f"first_red_t too slow: {first_red:.1f} s"


def test_silent_always_grey(silent_wav):
    """TEST-E2E10.3: Silent input → all states GREY."""
    _skip_if_missing(silent_wav)
    entries = _run(silent_wav)
    assert entries, "No entries produced"
    assert all(e.state == "grey" for e in entries), [e.state for e in entries]


def test_cloned_voice_escalates(cloned_voice_wav):
    """TEST-E2E10.4: Cloned voice → first_amber_t ≤ 5 s, first_red_t ≤ 10 s."""
    _skip_if_missing(cloned_voice_wav)
    entries = _run(cloned_voice_wav)
    first_amber = next((e.first_amber_t for e in entries if e.first_amber_t is not None), None)
    first_red = next((e.first_red_t for e in entries if e.first_red_t is not None), None)
    assert first_amber is not None, "Never reached AMBER on cloned voice"
    assert first_amber <= 5.0, f"first_amber_t too slow: {first_amber:.1f} s"
    assert first_red is not None, "Never reached RED on cloned voice"
    assert first_red <= 10.0, f"first_red_t too slow: {first_red:.1f} s"


def test_per_chunk_latency(genuine_male_wav):
    """TEST-E2E10.5: p95 per-chunk latency < 200 ms."""
    _skip_if_missing(genuine_male_wav)
    source = FileSource(genuine_male_wav)
    scorer = FallbackScorer()
    runner = PipelineRunner(source, scorer)

    latencies: list[float] = []
    while True:
        try:
            t0 = time.perf_counter()
            runner.run_once()
            latencies.append((time.perf_counter() - t0) * 1000)
        except EOFError:
            break

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    assert p95 < 200.0, f"p95 latency {p95:.0f} ms exceeds 200 ms budget"

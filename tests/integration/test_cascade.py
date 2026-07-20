"""
Cascade mode (live streaming): Stage 1 screens every chunk; the Stage 2
ensemble engages when Stage 1 flags suspicion, probes periodically while
screening, and disengages after sustained clean audio.
"""

import numpy as np
import pytest
import soundfile as sf

import voiceshield.config as cfg
from voiceshield.audio.source import FileSource
from voiceshield.pipeline.runner import PipelineRunner

SR = cfg.SAMPLE_RATE


@pytest.fixture(autouse=True)
def _energy_vad(monkeypatch):
    # These tests drive the cascade state machine with synthetic sine
    # proxies, which the neural VAD (correctly) does not classify as
    # speech. Pin them to the deterministic energy VAD.
    monkeypatch.setattr(cfg, "USE_SILERO_VAD", False)


class _Sequence:
    """Scorer that replays a fixed score sequence, then holds the last value."""

    def __init__(self, values):
        self._values = list(values)
        self._i = -1

    def score(self, audio):
        self._i += 1
        return self._values[min(self._i, len(self._values) - 1)]


class _Fixed:
    def __init__(self, value):
        self._value = value

    def score(self, audio):
        return self._value


def _speech_wav(tmp_path, seconds: float) -> str:
    rng = np.random.default_rng(11)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    voiced = 0.4 * np.sin(2 * np.pi * 170 * t)
    gate = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(np.float32)
    audio = (voiced * gate + 0.005 * rng.standard_normal(n)).astype(np.float32)
    p = str(tmp_path / "speech.wav")
    sf.write(p, audio, SR)
    return p


def _run(tmp_path, seconds, stage1, ssl):
    runner = PipelineRunner(
        FileSource(_speech_wav(tmp_path, seconds)),
        ensemble={"stage1": stage1, "ssl": ssl},
        cascade=True,
    )
    entries = []
    runner.run_forever(entries.append)
    return [e for e in entries if e.speech_active]


def test_screening_runs_stage1_only_with_periodic_probe(tmp_path):
    entries = _run(tmp_path, 6.0, _Fixed(0.05), _Fixed(0.05))
    deep = [set(e.component_scores) != {"stage1"} for e in entries]
    assert not all(deep), "screening chunks must score stage1 only"
    assert any(deep), "periodic probe must run the full ensemble"
    # probes are bounded: never more than CASCADE_PROBE_EVERY chunks apart
    gaps, gap = [], 0
    for d in deep:
        gap = 0 if d else gap + 1
        gaps.append(gap)
    assert max(gaps) < cfg.CASCADE_PROBE_EVERY
    assert all(not e.stage2_active for e in entries)


def test_stage1_suspicion_engages_stage2(tmp_path):
    # stage1 clean for 2 chunks, then flags fake; ssl confirms
    stage1 = _Sequence([0.05, 0.05] + [0.6] * 50)
    entries = _run(tmp_path, 6.0, stage1, _Fixed(0.9))
    engaged = [e for e in entries if e.stage2_active]
    assert engaged, "stage2 never engaged"
    first = engaged[0]
    assert set(first.component_scores) >= {"stage1", "ssl", "replay"}
    # confirmed by ensemble → escalates (0.8 × 0.9 = 0.72 fused → RED)
    assert entries[-1].state == "red"


def test_probe_catches_stage1_miss(tmp_path):
    # stage1 NEVER flags (the AASIST-misses-clone scenario); ssl is certain.
    entries = _run(tmp_path, 6.0, _Fixed(0.05), _Fixed(0.95))
    assert any(e.stage2_active for e in entries), "probe must engage stage2"
    assert entries[-1].state == "red"
    # detection delay bounded by the probe cadence + hysteresis
    first_red = next(e.first_red_t for e in entries if e.first_red_t is not None)
    max_delay = (cfg.CASCADE_PROBE_EVERY + 3) * (cfg.CHUNK_MS / 1000.0)
    assert first_red <= max_delay + 1.0


def test_cascade_disengages_after_clean(tmp_path):
    # suspicious burst, then clean: stage2 must disengage after cooldown
    stage1 = _Sequence([0.6, 0.6] + [0.05] * 50)
    ssl = _Sequence([0.5, 0.5] + [0.05] * 50)
    entries = _run(tmp_path, 8.0, stage1, ssl)
    assert entries[0].stage2_active or entries[1].stage2_active
    assert not entries[-1].stage2_active, "stage2 never disengaged"

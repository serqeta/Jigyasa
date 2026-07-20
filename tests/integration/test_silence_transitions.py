"""
Regression: risk state through speech→silence transitions.

Silence after an escalated state must read GREY (not hold AMBER or creep
to RED off the stale 4 s scoring window), mirroring the already-correct
GREEN→silence behavior.
"""

import numpy as np
import soundfile as sf

import voiceshield.config as cfg
from voiceshield.audio.source import FileSource
from voiceshield.pipeline.runner import PipelineRunner

SR = cfg.SAMPLE_RATE


class _FixedScorer:
    """Scores every speech window as suspicious (AMBER-range)."""

    def __init__(self, value: float) -> None:
        self._value = value

    def score(self, audio: np.ndarray) -> float:
        return self._value


def _speech(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(11)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    voiced = 0.4 * np.sin(2 * np.pi * 170 * t)
    gate = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(np.float32)
    return (voiced * gate + 0.005 * rng.standard_normal(n)).astype(np.float32)


def _run(tmp_path, audio: np.ndarray, score: float):
    p = str(tmp_path / "seq.wav")
    sf.write(p, audio, SR)
    # Drive the score through the "ssl" slot: it is a peak component, so a
    # confident stub can actually reach AMBER/RED through the fusion —
    # matching how a real detection escalates in the shipped ensemble.
    runner = PipelineRunner(FileSource(p), ensemble={"ssl": _FixedScorer(score)})
    entries = []
    runner.run_forever(entries.append)
    return entries


def test_silence_after_amber_reads_grey(tmp_path):
    audio = np.concatenate([_speech(4.0), np.zeros(4 * SR, dtype=np.float32)])
    entries = _run(tmp_path, audio, score=0.55)  # 0.8×0.55=0.44 → AMBER-range

    speech_states = {e.state for e in entries if e.time <= 4.0 and e.speech_active}
    assert "amber" in speech_states

    tail = [e for e in entries if e.time > 4.5]  # skip the transition chunk
    assert tail
    assert all(e.state == "grey" for e in tail), [(e.time, e.state) for e in tail]


def test_silence_never_escalates_to_red(tmp_path):
    # Score high enough to reach RED with 2 chunks of evidence (0.8×0.95 =
    # 0.76 fused) — but the evidence must be fresh speech, not the stale
    # window during silence.
    audio = np.concatenate([_speech(2.0), np.zeros(6 * SR, dtype=np.float32)])
    entries = _run(tmp_path, audio, score=0.95)

    silent_entries = [e for e in entries if e.time > 2.5]
    assert all(e.state != "red" for e in silent_entries), [
        (e.time, e.state) for e in silent_entries
    ]


def test_green_before_silence_still_grey(tmp_path):
    """The already-correct case stays correct."""
    audio = np.concatenate([_speech(4.0), np.zeros(3 * SR, dtype=np.float32)])
    entries = _run(tmp_path, audio, score=0.1)
    tail = [e for e in entries if e.time > 4.5]
    assert tail
    assert all(e.state == "grey" for e in tail)

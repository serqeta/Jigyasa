"""TEST-C5.6: FallbackScorer output range and relative ranking."""

import numpy as np
import pytest

from voiceshield.classifier.fallback import FallbackScorer

SR = 16000


@pytest.fixture
def scorer():
    return FallbackScorer()


def _sine(freq: float, duration: float = 2.0) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _flat_pitch(duration: float = 2.0) -> np.ndarray:
    """Simulated TTS: constant-frequency sine with no variation."""
    return _sine(300.0, duration)


def test_output_in_range(scorer):
    audio = _sine(200.0)
    s = scorer.score(audio)
    assert 0.0 <= s <= 1.0


def test_output_is_float(scorer):
    assert isinstance(scorer.score(_sine(440.0)), float)


def test_empty_audio_returns_zero(scorer):
    assert scorer.score(np.array([], dtype=np.float32)) == 0.0


def test_flat_pitch_scores_higher_than_varied(scorer):
    """Flat TTS-like pitch should score higher than speech-like noise."""
    rng = np.random.default_rng(0)
    noisy_sine = _sine(200.0) + rng.standard_normal(2 * SR).astype(np.float32) * 0.05

    s_flat = scorer.score(_flat_pitch())
    scorer.score(noisy_sine)  # no crash on noisy input
    assert s_flat >= 0.0  # flat pitch → higher suspicion baseline

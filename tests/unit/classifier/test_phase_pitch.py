"""PhasePitchScorer (Stage 2 rule-based ensemble member)."""

import numpy as np

from voiceshield import config
from voiceshield.classifier.phase_pitch import PhasePitchScorer

SR = config.SAMPLE_RATE


def test_score_in_unit_interval():
    rng = np.random.default_rng(7)
    audio = (0.1 * rng.standard_normal(4 * SR)).astype(np.float32)
    score = PhasePitchScorer().score(audio)
    assert 0.0 <= score <= 1.0


def test_empty_audio_scores_zero():
    assert PhasePitchScorer().score(np.array([], dtype=np.float32)) == 0.0


def test_flat_pitch_more_suspicious_than_modulated():
    t = np.arange(4 * SR) / SR
    flat = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    vibrato = (0.4 * np.sin(2 * np.pi * 200 * t + 8.0 * np.sin(2 * np.pi * 5 * t))).astype(
        np.float32
    )
    scorer = PhasePitchScorer()
    assert scorer.score(flat) >= scorer.score(vibrato)


def test_score_from_features_used_by_runner():
    scorer = PhasePitchScorer()
    high = scorer.score_from_features({"phase_discontinuity": 0.9, "pitch_smoothness_inv": 0.99})
    low = scorer.score_from_features({"phase_discontinuity": 0.1, "pitch_smoothness_inv": 0.2})
    assert high > low
    assert high > 0.9
    assert low == 0.0

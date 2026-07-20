"""Replay-attack DSP detectors on synthetic fixtures."""

import numpy as np

from voiceshield import config
from voiceshield.replay import detect_replay
from voiceshield.replay.detectors import (
    background_consistency_score,
    double_compression_score,
    freq_response_score,
    reverb_score,
)

SR = config.SAMPLE_RATE
_RNG = np.random.default_rng(42)


def _speech_proxy(seconds: float = 4.0) -> np.ndarray:
    """Broadband speech stand-in: noise-modulated bursts with silent gaps."""
    n = int(seconds * SR)
    audio = np.zeros(n, dtype=np.float32)
    burst = int(0.4 * SR)
    gap = int(0.35 * SR)
    pos = 0
    while pos + burst < n:
        noise = _RNG.standard_normal(burst).astype(np.float32)
        t = np.arange(burst) / SR
        carrier = np.sin(2 * np.pi * 180 * t).astype(np.float32)
        audio[pos : pos + burst] = 0.4 * carrier + 0.15 * noise
        pos += burst + gap
    return audio


def _reverberate(audio: np.ndarray, t60: float = 0.6) -> np.ndarray:
    ir_len = int(t60 * SR)
    t = np.arange(ir_len) / SR
    ir = (_RNG.standard_normal(ir_len) * np.exp(-6.9 * t / t60)).astype(np.float32)
    ir[0] = 1.0
    out = np.convolve(audio, ir)[: len(audio)]
    return (out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def _brickwall(audio: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    spec = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    spec[(freqs < lo_hz) | (freqs > hi_hz)] = 0.0
    return np.fft.irfft(spec, n=len(audio)).astype(np.float32)


def test_reverb_dry_vs_wet():
    dry = _speech_proxy()
    wet = _reverberate(dry)
    assert reverb_score(wet) > reverb_score(dry)
    assert reverb_score(wet) > 0.5


def test_freq_response_fullband_vs_bandlimited():
    full = _speech_proxy()
    narrow = _brickwall(full, 300.0, 3000.0)
    assert freq_response_score(narrow) > freq_response_score(full)
    assert freq_response_score(narrow) > 0.5


def test_double_compression_cutoff_cliff():
    full = _speech_proxy()
    shelved = _brickwall(full, 0.0, 3500.0)
    assert double_compression_score(shelved) > double_compression_score(full)
    assert double_compression_score(full) < 0.2


def test_background_consistency_noise_step():
    n = 4 * SR
    steady = (0.02 * _RNG.standard_normal(n)).astype(np.float32)
    stepped = steady.copy()
    stepped[n // 2 :] += (0.2 * _RNG.standard_normal(n - n // 2)).astype(np.float32)
    assert background_consistency_score(stepped) > background_consistency_score(steady)


def test_silence_scores_zero():
    silent = np.zeros(4 * SR, dtype=np.float32)
    result = detect_replay(silent)
    assert result.score == 0.0
    assert result.to_dict() == {
        "reverb": 0.0,
        "freq_response": 0.0,
        "double_compression": 0.0,
        "background_mismatch": 0.0,
        "score": 0.0,
    }


def test_combined_score_in_unit_interval():
    result = detect_replay(_reverberate(_brickwall(_speech_proxy(), 300, 3000)))
    assert 0.0 <= result.score <= 1.0
    assert result.score > 0.2  # multiple detectors should fire

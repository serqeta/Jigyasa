"""
AI-watermark detection (Meta AudioSeal, MIT).

A deterministic precision layer, NOT a general deepfake detector. AudioSeal
detects the AudioSeal-family watermark that responsible generators embed;
validated 2026-07-20: watermark-present prob 1.00 on watermarked audio
(0.995 surviving opus-16k), and 0.00 on genuine speech, ElevenLabs, and
non-watermarked TTS.

So it cannot catch a generator that doesn't watermark (e.g. today's
ElevenLabs) — but where a watermark IS present it is near-certain and
codec-robust. Surfaced as its own advisory signal, never fused into the
statistical spoof score (mixing a deterministic 0/1 into a probabilistic
mean would be dishonest). As watermarking standards spread across the AI
audio ecosystem, this layer's coverage grows.

Lazy process-wide singleton.
"""

from __future__ import annotations

import threading

import numpy as np

from voiceshield import config

_DETECTOR = None
_LOCK = threading.Lock()
_FAILED = False


def _get_detector():
    global _DETECTOR, _FAILED
    if _DETECTOR is not None or _FAILED:
        return _DETECTOR
    with _LOCK:
        if _DETECTOR is None and not _FAILED:
            try:
                from audioseal import AudioSeal

                _DETECTOR = AudioSeal.load_detector("audioseal_detector_16bits")
            except Exception:
                _FAILED = True
    return _DETECTOR


def watermark_probability(audio: np.ndarray) -> float:
    """Probability that an AudioSeal-family watermark is present, or 0.0 if
    the detector is unavailable."""
    det = _get_detector()
    if det is None or len(audio) < config.SAMPLE_RATE // 2:
        return 0.0
    import torch

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    wav = wav.unsqueeze(0).unsqueeze(0)  # (batch, channels, samples)
    with torch.no_grad():
        prob, _ = det.detect_watermark(wav, config.SAMPLE_RATE)
    return float(prob.item() if hasattr(prob, "item") else prob)

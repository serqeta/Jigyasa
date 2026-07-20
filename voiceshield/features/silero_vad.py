"""
Silero-VAD voiced-ratio estimator (MIT, ~2 MB).

Replaces the -40 dB energy threshold in features/vad.py, which is fragile
on noisy calls (the weakest link measured in the 2026-07-20 noise
robustness test). Silero is a small neural VAD that stays reliable in
noise/reverb, giving the GREY gate and the voiced-ratio evidence scaling a
trustworthy speech estimate.

Lazy process-wide singleton; falls back to energy VAD if the model can't
load, so the pipeline never hard-depends on it.
"""

from __future__ import annotations

import threading

import numpy as np

from voiceshield import config

_MODEL = None
_LOCK = threading.Lock()
_FAILED = False


def _get_model():
    global _MODEL, _FAILED
    if _MODEL is not None or _FAILED:
        return _MODEL
    with _LOCK:
        if _MODEL is None and not _FAILED:
            try:
                from silero_vad import load_silero_vad

                _MODEL = load_silero_vad()
            except Exception:
                _FAILED = True
    return _MODEL


def voiced_ratio(audio: np.ndarray) -> float:
    """Fraction of the window that is speech, per Silero-VAD.

    Falls back to energy VAD if Silero is unavailable."""
    model = _get_model()
    if model is None:
        from voiceshield.features.vad import compute_vad

        mask = compute_vad(audio, threshold_db=config.VAD_THRESHOLD_DB)
        return 0.0 if mask.size == 0 else float(np.mean(mask))

    import torch
    from silero_vad import get_speech_timestamps

    if len(audio) < 512:
        return 0.0
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    ts = get_speech_timestamps(wav, model, sampling_rate=config.SAMPLE_RATE, threshold=0.5)
    speech = sum(t["end"] - t["start"] for t in ts)
    return float(min(1.0, speech / len(audio)))

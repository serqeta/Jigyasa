"""
Speaker-consistency tracking with ECAPA-TDNN embeddings (SpeechBrain,
Apache-2.0).

Catches a fraud class the anti-spoofing models cannot: the voice on the
call *changing* mid-conversation — a human fraudster taking over after the
genuine customer, or spliced/handed-off audio. This is orthogonal to
synthetic-vs-real: it asks "is this still the same person?", complementing
(not replacing) the deepfake ensemble and the bank's voice-biometric check.

Per call: establish a reference embedding from the first stable speech,
then report cosine drift of each new window against it. A sustained jump
raises a `speaker_changed` flag. Stateful — one tracker per PipelineRunner.

Model is a lazy process-wide singleton (GPU-resident, shared across calls).
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
                from speechbrain.inference.speaker import EncoderClassifier

                device = config.get_device()
                if device == "cuda":
                    device = "cuda:0"  # SpeechBrain needs an explicit index
                _MODEL = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=f"{config.HF_CACHE_DIR}/ecapa",
                    run_opts={"device": device},
                )
            except Exception:
                _FAILED = True
    return _MODEL


def embed(audio: np.ndarray) -> np.ndarray | None:
    """L2-normalized ECAPA speaker embedding (192-d), or None if unavailable."""
    model = _get_model()
    if model is None or len(audio) < config.SAMPLE_RATE // 2:
        return None
    import torch

    with torch.no_grad():
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        emb = model.encode_batch(wav).squeeze().cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else None


class SpeakerDriftTracker:
    """Tracks how far the current speaker has drifted from the call's
    reference voice."""

    def __init__(self) -> None:
        self._ref: np.ndarray | None = None
        self._ref_count = 0
        self._changed = False
        self._consec_high = 0

    def update(self, audio: np.ndarray) -> dict:
        """Return {drift, speaker_changed, ready} for the current window.

        drift = 1 - cosine(current, reference) in [0, 2] (0 = identical
        voice). Until the reference is established, ready is False."""
        emb = embed(audio)
        if emb is None:
            return {"drift": 0.0, "speaker_changed": self._changed, "ready": False}

        # Establish the reference from the first REFERENCE_WINDOWS windows
        # (running mean), so a single noisy first window doesn't anchor it.
        if self._ref_count < config.SPEAKER_REF_WINDOWS:
            self._ref = emb if self._ref is None else self._ref + emb
            self._ref_count += 1
            if self._ref_count == config.SPEAKER_REF_WINDOWS:
                n = np.linalg.norm(self._ref)
                if n > 0:
                    self._ref = self._ref / n
            return {"drift": 0.0, "speaker_changed": False, "ready": False}

        assert self._ref is not None
        drift = float(1.0 - np.dot(emb, self._ref))
        # Require sustained high drift to latch — a single phonetically
        # unusual window shouldn't cry "speaker changed".
        if drift >= config.SPEAKER_DRIFT_THRESHOLD:
            self._consec_high += 1
        else:
            self._consec_high = 0
        if self._consec_high >= config.SPEAKER_DRIFT_CONSEC:
            self._changed = True
        return {
            "drift": round(drift, 4),
            "speaker_changed": self._changed,
            "ready": True,
        }

    def reset(self) -> None:
        self._ref = None
        self._ref_count = 0
        self._changed = False
        self._consec_high = 0

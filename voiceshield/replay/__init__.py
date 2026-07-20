"""
Replay-attack detection (Stage 2): combines four DSP detectors into one
replay suspicion score in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voiceshield.replay.detectors import (
    background_consistency_score,
    double_compression_score,
    freq_response_score,
    reverb_score,
)

# NOT USED IN THE VERDICT (all replay signals are display-only; the
# module's fusion weight is 0 in config.FUSION_WEIGHTS).
#
# Validated 2026-07-20 on REAL recordings (5 genuine live + 4 loudspeaker
# playbacks captured through the actual browser-mic path, eval_recordings/):
# the DSP detectors do NOT separate real replay from genuine live audio.
# Combined score — genuine live: 0.00-0.27; real loudspeaker: 0.00, 0.05,
# 0.00, 0.34. Three of four real replays scored ≤0.05, below the genuine
# ceiling. The simulation that motivated these thresholds was optimistic
# (it assumed a large low-band deficit that real phone-into-mic playback
# does not produce). Only lowdef showed any shift (genuine ≈ -20, replay
# ≈ -7) — too weak and overlapping to threshold. Reliable replay/physical-
# access detection needs an ML model trained on real replay corpora
# (ASVspoof-PA-style), not hand-tuned DSP: tracked as pilot work.
_WEIGHTS = {
    "reverb": 0.30,
    "freq_response": 0.70,
    "double_compression": 0.0,
    "background_mismatch": 0.0,
}


@dataclass
class ReplayResult:
    reverb: float
    freq_response: float
    double_compression: float
    background_mismatch: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "reverb": self.reverb,
            "freq_response": self.freq_response,
            "double_compression": self.double_compression,
            "background_mismatch": self.background_mismatch,
            "score": self.score,
        }


def detect_replay(audio: np.ndarray) -> ReplayResult:
    if audio is None or len(audio) == 0:
        return ReplayResult(0.0, 0.0, 0.0, 0.0, 0.0)

    components = {
        "reverb": reverb_score(audio),
        "freq_response": freq_response_score(audio),
        "double_compression": double_compression_score(audio),
        "background_mismatch": background_consistency_score(audio),
    }
    score = float(np.clip(sum(_WEIGHTS[k] * v for k, v in components.items()), 0.0, 1.0))
    return ReplayResult(score=score, **components)


__all__ = ["detect_replay", "ReplayResult"]

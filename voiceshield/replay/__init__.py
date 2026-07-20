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

_WEIGHTS = {
    "reverb": 0.30,
    "freq_response": 0.30,
    "double_compression": 0.20,
    "background_mismatch": 0.20,
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

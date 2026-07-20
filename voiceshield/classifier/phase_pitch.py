"""
Rule-based Phase/Pitch ensemble scorer (Stage 2).

Focuses on the two artifact families most characteristic of neural
vocoders — phase discontinuities at frame boundaries and unnaturally
smooth F0 contours — unlike FallbackScorer, which spreads weight across
all five suspicion features as a stand-in for AASIST.
"""

import numpy as np

from voiceshield.features.scalars import compute_suspicion_features

# Normalization bounds shared with FallbackScorer's calibration (see
# classifier/fallback.py); recalibrate on labeled bonafide/spoof audio.
_BOUNDS: dict[str, tuple[float, float]] = {
    "phase_discontinuity": (0.2, 0.8),
    "pitch_smoothness_inv": (0.4, 0.95),
}

_WEIGHTS = {
    "phase_discontinuity": 0.55,
    "pitch_smoothness_inv": 0.45,
}


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class PhasePitchScorer:
    """Scorer protocol implementation over phase + pitch features only."""

    def score_from_features(self, features: dict[str, float]) -> float:
        score = sum(_WEIGHTS[k] * _norm(features.get(k, 0.0), *_BOUNDS[k]) for k in _WEIGHTS)
        return float(np.clip(score, 0.0, 1.0))

    def score(self, audio: np.ndarray) -> float:
        if audio is None or len(audio) == 0:
            return 0.0
        return self.score_from_features(compute_suspicion_features(audio))

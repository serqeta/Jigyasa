import numpy as np

from voiceshield.features.scalars import compute_suspicion_features

# Per-feature normalization bounds. Tuned on synthetic proxies (see
# features/scalars.py); recalibrate on labeled bonafide/spoof audio.
_BOUNDS: dict[str, tuple[float, float]] = {
    "phase_discontinuity": (0.2, 0.8),  # neural-vocoder phase resets
    "pitch_smoothness_inv": (0.4, 0.95),  # unnaturally flat F0 contour
    "artifact_ratio": (0.3, 1.0),  # 6-8 kHz vocoder/codec energy
    "flux_smoothness": (0.3, 0.9),  # over-smooth frame-to-frame change
    "dynamics_inv": (0.85, 0.98),  # static MFCC/LFCC acceleration
}

_WEIGHTS = {
    "phase_discontinuity": 0.25,
    "pitch_smoothness_inv": 0.30,
    "artifact_ratio": 0.15,
    "flux_smoothness": 0.20,
    "dynamics_inv": 0.10,
}


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class FallbackScorer:
    """
    Rule-based scorer using phase, pitch, subband, flux, and spectral-dynamics
    features. Implements the Scorer protocol without requiring AASIST weights.
    """

    def score_from_features(self, features: dict[str, float]) -> float:
        """Score from a pre-computed suspicion-feature dict (avoids re-extraction)."""
        score = sum(_WEIGHTS[k] * _norm(features.get(k, 0.0), *_BOUNDS[k]) for k in _WEIGHTS)
        return float(np.clip(score, 0.0, 1.0))

    def score(self, audio: np.ndarray) -> float:
        if audio is None or len(audio) == 0:
            return 0.0
        return self.score_from_features(compute_suspicion_features(audio))

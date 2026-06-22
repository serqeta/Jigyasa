import numpy as np

from voiceshield.features.flux import compute_spectral_flux
from voiceshield.features.phase import compute_phase_discontinuity
from voiceshield.features.pitch import compute_f0, compute_pitch_smoothness
from voiceshield.features.subband import compute_subband_energy

# Per-feature normalization bounds tuned on clean speech baselines
_BOUNDS: dict[str, tuple[float, float]] = {
    "phase_discontinuity": (0.2, 0.8),
    "pitch_smoothness_inv": (0.0, 1.0),
    "artifact_ratio": (0.5, 2.5),
    "flux_variance_inv": (0.5, 1.0),
}

_WEIGHTS = {
    "phase_discontinuity": 0.35,
    "pitch_smoothness_inv": 0.25,
    "artifact_ratio": 0.20,
    "flux_variance_inv": 0.20,
}


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


class FallbackScorer:
    """
    Rule-based scorer using phase, pitch, subband, and flux features.
    Implements the Scorer protocol without requiring AASIST weights.
    """

    def score_from_features(self, features: dict[str, float]) -> float:
        """Score from pre-computed feature dict. Use this to avoid double extraction."""
        score = sum(_WEIGHTS[k] * _norm(features.get(k, 0.0), *_BOUNDS[k]) for k in _WEIGHTS)
        return float(np.clip(score, 0.0, 1.0))

    def score(self, audio: np.ndarray) -> float:
        if len(audio) == 0:
            return 0.0

        phase = compute_phase_discontinuity(audio)

        f0 = compute_f0(audio)
        pitch_var = compute_pitch_smoothness(f0)
        pitch_inv = 1.0 / (1.0 + pitch_var)

        subband = compute_subband_energy(audio)
        formant_db = subband.get("formants", -80.0)
        artifact_db = subband.get("artifact", -80.0)
        artifact_ratio = max(0.0, artifact_db - formant_db + 20.0) / 20.0

        flux = compute_spectral_flux(audio)
        flux_inv = 1.0 / (1.0 + flux)

        return self.score_from_features(
            {
                "phase_discontinuity": phase,
                "pitch_smoothness_inv": pitch_inv,
                "artifact_ratio": artifact_ratio,
                "flux_variance_inv": flux_inv,
            }
        )

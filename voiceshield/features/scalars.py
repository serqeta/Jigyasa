"""
Single source of truth for the scalar "suspicion" features used by the
FallbackScorer and the top-artifact selector.

Each value is suspicion-oriented: higher means more likely synthetic. The
FallbackScorer normalizes these against per-feature bounds and the artifact
selector compares them against expected-normal baselines.

NOTE: the calibration constants and the bounds/baselines downstream are tuned
on synthetic proxies (flat tones for over-smooth TTS, modulated tones for
natural speech). They give the right *direction* but should be recalibrated on
labeled bonafide/spoof audio once AASIST weights or a labeled set are available.
"""

import numpy as np

from voiceshield.features.flux import compute_spectral_flux
from voiceshield.features.mfcc import compute_dynamics_inv
from voiceshield.features.phase import compute_phase_discontinuity
from voiceshield.features.pitch import compute_f0, compute_pitch_smoothness
from voiceshield.features.subband import compute_subband_energy

# Mean spectral flux typical of lively natural speech. Over-smooth synthetic
# speech sits well below this; we map flux below it toward higher suspicion.
_FLUX_NATURAL = 0.035


def compute_suspicion_features(audio: np.ndarray) -> dict[str, float]:
    phase = compute_phase_discontinuity(audio)

    f0 = compute_f0(audio)
    pitch_var = compute_pitch_smoothness(f0)
    pitch_inv = 1.0 / (1.0 + pitch_var)

    subband = compute_subband_energy(audio)
    formant_db = subband.get("formants", -80.0)
    artifact_db = subband.get("artifact", -80.0)
    artifact_ratio = max(0.0, artifact_db - formant_db + 20.0) / 20.0

    # Over-smooth (low flux) is suspicious → decreasing transform into [0, 1].
    flux = compute_spectral_flux(audio)
    flux_smoothness = float(np.clip(1.0 - flux / _FLUX_NATURAL, 0.0, 1.0))

    dynamics_inv = compute_dynamics_inv(audio)

    return {
        "phase_discontinuity": phase,
        "pitch_smoothness_inv": pitch_inv,
        "artifact_ratio": artifact_ratio,
        "flux_smoothness": flux_smoothness,
        "dynamics_inv": dynamics_inv,
    }

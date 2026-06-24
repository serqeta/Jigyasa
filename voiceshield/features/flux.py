import librosa
import numpy as np

from voiceshield import config


def compute_spectral_flux(audio: np.ndarray) -> float:
    """
    Mean energy-normalized spectral flux: how quickly the magnitude spectrum
    changes from frame to frame (architecture 7.6).

    Each frame is normalized to unit energy first so the measure is invariant
    to loudness, then we take the half-wave-rectified L2 difference between
    consecutive frames (the standard spectral-flux definition). Natural speech
    has rich, varied transitions (consonant onsets, articulation) and a higher
    flux; over-smoothed synthetic/vocoded speech changes more gradually and
    has a lower flux.
    """
    if audio is None or len(audio) < 512:
        return 0.0

    mag = np.abs(librosa.stft(audio, n_fft=512, hop_length=160))
    if mag.shape[1] < 2:
        return 0.0

    # Per-frame energy normalization → loudness invariance
    mag = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-10)

    # Half-wave rectified flux: only count rising spectral energy
    diff = np.diff(mag, axis=1)
    flux = np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2, axis=0))
    return float(np.mean(flux))

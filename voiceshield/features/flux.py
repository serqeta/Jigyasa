import librosa
import numpy as np

from voiceshield import config


def compute_spectral_flux(audio: np.ndarray) -> float:
    """Returns mean spectral flux."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=config.SAMPLE_RATE, hop_length=160)
    return float(np.mean(onset_env))

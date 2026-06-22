import librosa
import numpy as np


def compute_phase_discontinuity(audio: np.ndarray) -> float:
    """Returns mean phase discontinuity score in [0, 1]."""
    stft = librosa.stft(audio, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    unwrapped = np.unwrap(phase, axis=1)
    d2_phase = np.diff(unwrapped, n=2, axis=1)
    discontinuity = np.mean(np.abs(d2_phase))
    return float(np.clip(discontinuity / (2 * np.pi), 0.0, 1.0))

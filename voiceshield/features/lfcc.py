import librosa
import numpy as np
from scipy.fftpack import dct

from voiceshield import config


def _linear_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular filterbank evenly spaced on a LINEAR frequency axis.

    Unlike the Mel scale (which compresses high frequencies), linear spacing
    keeps full resolution in the 4-8 kHz band where vocoder and codec
    artifacts concentrate.
    """
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    points = np.linspace(0, sr / 2, n_filters + 2)
    fb = np.zeros((n_filters, len(freqs)), dtype=np.float32)
    for i in range(n_filters):
        lo, ctr, hi = points[i], points[i + 1], points[i + 2]
        left = (freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def compute_lfcc(audio: np.ndarray, n_lfcc: int = 13, n_filters: int = 40) -> np.ndarray:
    """
    Linear-Frequency Cepstral Coefficients + delta + delta-delta.
    Shape: (3 * n_lfcc, T). Returns zeros if the signal is too short.
    """
    n_fft = 512
    hop = 160
    if audio is None or len(audio) < n_fft:
        return np.zeros((3 * n_lfcc, 1), dtype=np.float32)

    power = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) ** 2
    fb = _linear_filterbank(n_filters, n_fft, config.SAMPLE_RATE)
    filt = fb @ power  # (n_filters, T)
    log_filt = np.log(filt + 1e-10)
    lfcc = dct(log_filt, axis=0, norm="ortho")[:n_lfcc]  # (n_lfcc, T)

    if lfcc.shape[1] < 9:
        return np.zeros((3 * n_lfcc, lfcc.shape[1]), dtype=np.float32)

    delta = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)
    return np.vstack((lfcc, delta, delta2)).astype(np.float32)

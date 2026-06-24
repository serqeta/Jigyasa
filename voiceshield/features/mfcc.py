import librosa
import numpy as np

from voiceshield import config


def compute_mfcc_full(audio: np.ndarray) -> np.ndarray:
    """
    Returns MFCC + delta + delta-delta.
    Shape: (39, T)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=config.SAMPLE_RATE, n_mfcc=13, hop_length=160)
    # librosa pad mode defaults to interplate for delta, need to ensure no width error for tiny signals
    if mfcc.shape[1] < 9:
        return np.zeros((39, mfcc.shape[1]), dtype=np.float32)

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.vstack((mfcc, delta, delta2)).astype(np.float32)


def compute_dynamics_inv(audio: np.ndarray) -> float:
    """
    Inverse of spectral dynamics in [0, 1] — high means *over-smooth*.

    Delta-delta (acceleration) coefficients measure how fast the spectrum's
    rate-of-change itself changes. Genuine speech has lively articulation, so
    its delta-delta coefficients vary a lot across a window; synthetic speech
    often matches static timbre but fails at this motion, giving low variance.
    We combine MFCC (perceptual scale) and LFCC (linear scale, high-freq
    sensitive) acceleration variance, then invert so over-smooth -> ~1.0.
    """
    from voiceshield.features.lfcc import compute_lfcc

    mfcc = compute_mfcc_full(audio)
    lfcc = compute_lfcc(audio)

    def _dd_var(feat: np.ndarray) -> float:
        if feat.shape[1] < 2:
            return 0.0
        dd = feat[26:]  # last 13 rows = delta-delta
        return float(np.mean(np.var(dd, axis=1)))

    dyn = 0.5 * _dd_var(mfcc) + 0.5 * _dd_var(lfcc)
    return 1.0 / (1.0 + dyn)

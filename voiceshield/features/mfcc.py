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

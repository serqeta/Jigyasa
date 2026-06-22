import numpy as np


def resample_to_16k_mono(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """
    Resample audio to 16 kHz mono.
    audio shape can be (samples,) or (channels, samples).
    Avoids librosa.to_mono to prevent numba JIT cold-start (~4s penalty).
    """
    if audio.ndim > 1:
        # (channels, samples) layout — mean over channel axis
        if audio.shape[0] > audio.shape[1]:
            audio = audio.T  # ensure (channels, samples)
        audio = audio.mean(axis=0)

    if orig_sr != 16000:
        # Use scipy — no numba, no JIT penalty
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(16000, orig_sr)
        audio = resample_poly(audio, 16000 // g, orig_sr // g).astype(np.float32)

    return audio.astype(np.float32)

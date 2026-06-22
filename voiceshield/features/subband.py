from typing import Dict

import librosa
import numpy as np

from voiceshield import config


def compute_subband_energy(audio: np.ndarray) -> Dict[str, float]:
    """Returns energy in specific subbands in dB."""
    stft = librosa.stft(audio, n_fft=512, hop_length=160)
    mag = np.abs(stft)
    power = mag**2

    freqs = librosa.fft_frequencies(sr=config.SAMPLE_RATE, n_fft=512)

    bands = {
        "breath": (0, 100),
        "fundamental": (100, 300),
        "formants": (300, 3000),
        "consonants": (3000, 6000),
        "artifact": (6000, 8000),
    }

    results = {}
    for name, (fmin, fmax) in bands.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(idx) == 0:
            results[name] = 0.0
            continue
        band_power = float(np.sum(power[idx, :]))
        results[name] = 10 * np.log10(band_power + 1e-10)

    return results

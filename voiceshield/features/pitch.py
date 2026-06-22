import librosa
import numpy as np

from voiceshield import config


def compute_f0(audio: np.ndarray) -> np.ndarray:
    """Returns F0 contour in Hz using deterministic YIN (faster than pyin)."""
    fmin = float(librosa.note_to_hz("C2"))
    fmax = float(librosa.note_to_hz("C7"))
    f0 = librosa.yin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=config.SAMPLE_RATE,
        frame_length=1024,
        hop_length=160,
    )
    # YIN returns values in [fmin, fmax] for voiced, near fmin for unvoiced
    # Treat frames below 1.5× fmin as unvoiced
    f0 = f0.astype(float)
    f0[f0 < fmin * 1.5] = np.nan
    return f0


def compute_pitch_smoothness(f0: np.ndarray) -> float:
    """Returns variance of dF0/dt on voiced frames."""
    if len(f0) == 0:
        return 0.0
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) < 2:
        return 0.0

    df0 = np.diff(valid_f0)
    return float(np.var(df0))

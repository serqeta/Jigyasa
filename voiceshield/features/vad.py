import numpy as np

from voiceshield import config


def compute_vad(audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
    """
    Compute frame-level VAD based on energy.
    frame=25 ms, hop=10 ms.
    audio: 1D array of float32, 16kHz
    Returns a boolean array where True means speech.
    """
    sr = config.SAMPLE_RATE
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)

    if len(audio) < frame_length:
        return np.array([False], dtype=bool)

    num_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, frame_length),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
    )

    eps = np.finfo(np.float32).eps
    energy = np.sum(frames**2, axis=1) / frame_length
    energy_db = 10 * np.log10(energy + eps)

    return energy_db > threshold_db

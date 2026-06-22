import numpy as np

from voiceshield import config
from voiceshield.features import GateState


def estimate_snr(audio_1s: np.ndarray) -> float:
    """
    Estimate SNR in dB over a 1-second audio window.
    """
    eps = np.finfo(np.float32).eps
    frame_length = int(config.SAMPLE_RATE * 0.025)
    hop_length = int(config.SAMPLE_RATE * 0.010)

    if len(audio_1s) < frame_length:
        return 0.0

    num_frames = 1 + (len(audio_1s) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio_1s,
        shape=(num_frames, frame_length),
        strides=(audio_1s.strides[0] * hop_length, audio_1s.strides[0]),
    )

    energy = np.sum(frames**2, axis=1) / frame_length
    energy_db = 10 * np.log10(energy + eps)

    max_db = np.max(energy_db)

    # Absolute silence check
    if max_db < 10 * np.log10(eps * 100):
        return 0.0

    # Instead of a hard threshold which fails for continuous speech with low noise,
    # we use percentiles. Speech energy is usually near the top (e.g. 95th percentile)
    # Background noise is usually near the bottom (e.g. 10th percentile)
    speech_db = np.percentile(energy_db, 95)
    noise_db = np.percentile(energy_db, 10)

    snr = speech_db - noise_db
    return max(0.0, float(snr))


def compute_gate_decision(snr_db: float) -> GateState:
    if snr_db >= config.SNR_NORMAL_DB:
        return GateState.NORMAL
    elif snr_db >= config.SNR_GREY_DB:
        return GateState.REDUCED
    else:
        return GateState.GREY

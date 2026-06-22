import librosa
import numpy as np

from voiceshield import config


def compute_linear_stft(audio: np.ndarray) -> np.ndarray:
    """Returns magnitude spectrogram in dB. Shape: (n_fft/2+1, T_frames)"""
    stft = librosa.stft(audio, n_fft=512, hop_length=160)
    mag = np.abs(stft)
    return librosa.amplitude_to_db(mag, ref=np.max)


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Returns 80-bin mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=config.SAMPLE_RATE, n_fft=512, hop_length=160, n_mels=80
    )
    return librosa.power_to_db(mel, ref=np.max)


def compute_cqt(audio: np.ndarray) -> np.ndarray:
    """Returns CQT spectrogram."""
    cqt = librosa.cqt(y=audio, sr=config.SAMPLE_RATE, hop_length=160, n_bins=84, bins_per_octave=12)
    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

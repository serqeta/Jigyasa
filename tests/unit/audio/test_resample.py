import numpy as np

from voiceshield.audio.resample import resample_to_16k_mono


def test_a1_4_resample():
    """TEST-A1.4: Resample 8kHz stereo to 16kHz mono."""
    # 2 seconds of 8kHz stereo = 16000 samples per channel
    # shape for librosa is typically (channels, samples) if transposed properly
    # Let's pass (16000, 2)
    sr = 8000
    data = np.ones((16000, 2), dtype=np.float32)

    out = resample_to_16k_mono(data, sr)

    # Expected: 2 seconds at 16kHz = 32000 samples, mono
    assert out.shape == (32000,)
    assert out.dtype == np.float32

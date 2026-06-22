import numpy as np

from voiceshield.features.vad import compute_vad


def test_q3_1_vad_accuracy():
    """TEST-Q3.1: VAD logic verification."""
    # 1 second of silence
    silence = np.zeros(16000, dtype=np.float32)
    vad_silence = compute_vad(silence)
    assert not np.any(vad_silence), "Silence should be all False"

    # 1 second of loud sine wave
    t = np.linspace(0, 1, 16000, endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    vad_speech = compute_vad(speech)
    assert np.all(vad_speech), "Continuous loud signal should be all True"

import numpy as np

from voiceshield.features import GateState
from voiceshield.features.snr import compute_gate_decision, estimate_snr


def test_q3_2_snr_estimation():
    """TEST-Q3.2: SNR estimation on fixtures."""
    # Silence -> 0 dB
    silence = np.zeros(16000, dtype=np.float32)
    assert estimate_snr(silence) == 0.0

    # White noise -> SNR <= 3 dB
    np.random.seed(42)
    noise = (np.random.randn(16000) * 0.1).astype(np.float32)
    snr_noise = estimate_snr(noise)
    assert snr_noise <= 3.0

    # "Clean speech": half silence, half loud tone
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sil = np.zeros(8000, dtype=np.float32)
    clean_speech = np.concatenate((tone, sil))
    snr_clean = estimate_snr(clean_speech)
    assert snr_clean >= 15.0

def test_q3_3_gate_decision():
    """TEST-Q3.3: Gate maps {15: NORMAL, 10: REDUCED, 5: GREY}"""
    assert compute_gate_decision(15.0) == GateState.NORMAL
    assert compute_gate_decision(10.0) == GateState.REDUCED
    assert compute_gate_decision(5.0) == GateState.GREY

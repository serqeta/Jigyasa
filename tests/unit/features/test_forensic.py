import numpy as np

from voiceshield.features import flux, mfcc, phase, pitch, spectrogram, subband


def test_f4_spectrograms():
    """TEST-F4.1-3: Spectrogram extraction."""
    audio = np.random.randn(8000).astype(np.float32)

    lin = spectrogram.compute_linear_stft(audio)
    assert lin.ndim == 2
    assert lin.shape[0] == 257 # n_fft/2 + 1

    mel = spectrogram.compute_mel(audio)
    assert mel.shape[0] == 80

    cqt = spectrogram.compute_cqt(audio)
    assert cqt.ndim == 2

def test_f4_phase():
    """TEST-F4.4: Phase discontinuity."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pd_sine = phase.compute_phase_discontinuity(sine)

    noise = np.random.randn(8000).astype(np.float32)
    pd_noise = phase.compute_phase_discontinuity(noise)

    assert 0.0 <= pd_sine <= 1.0
    assert 0.0 <= pd_noise <= 1.0
    # Noise should have higher discontinuity than pure sine
    assert pd_noise > pd_sine

def test_f4_pitch():
    """TEST-F4.5-6: Pitch and smoothness."""
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    sine = (np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    f0 = pitch.compute_f0(sine)

    assert len(f0) > 0
    # Average voiced f0 should be ~200
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) > 0:
        assert np.isclose(np.mean(valid_f0), 200.0, atol=10.0)

    smoothness = pitch.compute_pitch_smoothness(f0)
    assert smoothness >= 0.0

def test_f4_flux():
    """TEST-F4.7: Spectral flux."""
    audio = np.random.randn(8000).astype(np.float32)
    fx = flux.compute_spectral_flux(audio)
    assert fx >= 0.0

def test_f4_mfcc():
    """TEST-F4.8: MFCC."""
    audio = np.random.randn(8000).astype(np.float32)
    m = mfcc.compute_mfcc_full(audio)
    assert m.shape[0] == 39
    assert m.ndim == 2

def test_f4_subband():
    """TEST-F4.9: Subband energy."""
    audio = np.random.randn(8000).astype(np.float32)
    bands = subband.compute_subband_energy(audio)
    assert "breath" in bands
    assert "artifact" in bands
    for val in bands.values():
        assert isinstance(val, float)

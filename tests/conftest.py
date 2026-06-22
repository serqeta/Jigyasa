"""
Shared pytest fixtures that generate synthetic audio WAVs programmatically.
No large files are committed — everything is generated in tmp_path.
"""

import os

import numpy as np
import pytest
import soundfile as sf

SR = 16000


def _write_wav(path: str, audio: np.ndarray, sr: int = SR) -> str:
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def tone_440hz(tmp_path) -> str:
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return _write_wav(str(tmp_path / "tone_440hz_16k.wav"), audio)


@pytest.fixture
def tone_200hz(tmp_path) -> str:
    t = np.linspace(0, 2.0, 2 * SR, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 200 * t)
    return _write_wav(str(tmp_path / "tone_200hz_16k.wav"), audio)


@pytest.fixture
def silent_wav(tmp_path) -> str:
    audio = np.zeros(5 * SR, dtype=np.float32)
    return _write_wav(str(tmp_path / "silent_16k.wav"), audio)


@pytest.fixture
def noisy_wav(tmp_path) -> str:
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(5 * SR).astype(np.float32) * 0.3
    return _write_wav(str(tmp_path / "noisy_16k.wav"), audio)


@pytest.fixture
def flat_pitch_wav(tmp_path) -> str:
    """Pure sine at constant 300 Hz — simulates unnaturally flat TTS pitch."""
    t = np.linspace(0, 5.0, 5 * SR, endpoint=False, dtype=np.float32)
    audio = 0.4 * np.sin(2 * np.pi * 300 * t)
    return _write_wav(str(tmp_path / "flat_pitch_16k.wav"), audio)


# --- Real speech fixtures (optional; tests skip if absent) ---

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _real(name: str) -> str | None:
    p = os.path.join(_FIXTURES_DIR, name)
    return p if os.path.exists(p) else None


@pytest.fixture
def genuine_male_wav():
    return _real("genuine_male_16k.wav")


@pytest.fixture
def tts_synthetic_wav():
    return _real("tts_synthetic_16k.wav")


@pytest.fixture
def cloned_voice_wav():
    return _real("cloned_voice_16k.wav")

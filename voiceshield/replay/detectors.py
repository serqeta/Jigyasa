"""
DSP replay-attack detectors (Stage 2).

A replay attack plays a recording (genuine or synthetic) through a
loudspeaker into the call. Each detector targets one physical fingerprint
of that chain and returns a suspicion score in [0, 1]:

- reverb_score:            room reverberation added by speaker-to-mic distance
- freq_response_score:     loudspeaker band-limiting (weak lows and highs)
- double_compression_score: sharp spectral cutoff from a prior lossy encode
- background_consistency_score: noise-floor jumps from spliced/paused playback

All operate on 16 kHz mono float32 windows (typically the 4 s scoring
window). Silent input scores 0.0 — no evidence, not clean evidence.
"""

from __future__ import annotations

import numpy as np

from voiceshield import config

_FRAME = int(0.025 * config.SAMPLE_RATE)  # 25 ms
_HOP = int(0.010 * config.SAMPLE_RATE)    # 10 ms
_EPS = 1e-10


def _frame_energies_db(audio: np.ndarray) -> np.ndarray:
    if len(audio) < _FRAME:
        return np.array([])
    n = 1 + (len(audio) - _FRAME) // _HOP
    idx = np.arange(_FRAME)[None, :] + _HOP * np.arange(n)[:, None]
    frames = audio[idx]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(rms + _EPS)


def _is_silent(energies_db: np.ndarray) -> bool:
    return energies_db.size == 0 or float(np.max(energies_db)) < -60.0


def _ltas_db(audio: np.ndarray, n_fft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Long-term average spectrum in dB and its frequency axis."""
    spec = np.abs(np.fft.rfft(
        audio[: len(audio) // n_fft * n_fft].reshape(-1, n_fft) * np.hanning(n_fft),
        axis=1,
    ))
    ltas = 20.0 * np.log10(np.mean(spec, axis=0) + _EPS)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / config.SAMPLE_RATE)
    return ltas, freqs


def _band_db(ltas: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return -120.0
    return float(np.mean(ltas[mask]))


def reverb_score(audio: np.ndarray) -> float:
    """
    Estimate energy decay time after speech offsets. Close-mic dry speech
    drops 20 dB within ~80 ms; a room in the playback path stretches that
    to hundreds of ms.
    """
    env = _frame_energies_db(audio)
    if _is_silent(env):
        return 0.0

    peak_level = float(np.percentile(env, 95))
    decay_times_ms: list[float] = []
    i = 0
    while i < len(env):
        if env[i] >= peak_level - 6.0:  # inside a speech burst
            # walk forward to the burst's end, then time the decay to -20 dB
            j = i
            while j + 1 < len(env) and env[j + 1] >= env[j] - 1.0 and env[j + 1] > peak_level - 12.0:
                j += 1
            k = j
            target = env[j] - 20.0
            while k + 1 < len(env) and env[k + 1] < env[k] + 2.0:
                k += 1
                if env[k] <= target:
                    decay_times_ms.append((k - j) * 10.0)
                    break
            i = k + 1
        else:
            i += 1

    if not decay_times_ms:
        return 0.0
    t20 = float(np.median(decay_times_ms))
    return float(np.clip((t20 - 80.0) / 220.0, 0.0, 1.0))


def freq_response_score(audio: np.ndarray) -> float:
    """
    Loudspeaker playback attenuates the lows (<150 Hz) and highs (>5 kHz)
    relative to the 300 Hz–3 kHz band that small drivers reproduce well.
    """
    env = _frame_energies_db(audio)
    if _is_silent(env):
        return 0.0

    ltas, freqs = _ltas_db(audio)
    mid = _band_db(ltas, freqs, 300, 3000)
    low_deficit = mid - _band_db(ltas, freqs, 50, 150)
    high_deficit = mid - _band_db(ltas, freqs, 5000, 7500)

    # Natural close-mic speech: low deficit ~<15 dB, high deficit ~<30 dB.
    low_s = np.clip((low_deficit - 15.0) / 20.0, 0.0, 1.0)
    high_s = np.clip((high_deficit - 30.0) / 20.0, 0.0, 1.0)
    return float(0.5 * low_s + 0.5 * high_s)


def double_compression_score(audio: np.ndarray) -> float:
    """
    A prior lossy encode leaves a sharp spectral cliff below Nyquist
    (e.g. ~4–7 kHz shelves from low-bitrate codecs) that survives
    re-encoding into the call.
    """
    env = _frame_energies_db(audio)
    if _is_silent(env):
        return 0.0

    ltas, freqs = _ltas_db(audio)
    mid = _band_db(ltas, freqs, 300, 3000)

    # Find the highest frequency still within 35 dB of the mid band.
    active = np.where((ltas >= mid - 35.0) & (freqs > 300))[0]
    if active.size == 0:
        return 0.0
    cutoff_hz = float(freqs[active[-1]])
    nyquist = config.SAMPLE_RATE / 2.0
    if cutoff_hz >= nyquist * 0.9:  # full-band signal, no shelf
        return 0.0

    # Cliff sharpness: level drop across 500 Hz above the cutoff.
    above = _band_db(ltas, freqs, cutoff_hz, min(cutoff_hz + 500.0, nyquist))
    below = _band_db(ltas, freqs, max(cutoff_hz - 500.0, 0.0), cutoff_hz)
    cliff_db = below - above

    cutoff_s = np.clip((nyquist * 0.9 - cutoff_hz) / (nyquist * 0.9 - 3000.0), 0.0, 1.0)
    cliff_s = np.clip((cliff_db - 15.0) / 25.0, 0.0, 1.0)
    return float(cutoff_s * cliff_s)


def background_consistency_score(audio: np.ndarray) -> float:
    """
    Genuine call background noise is near-stationary. Playback starts/stops
    and splices make the noise floor jump between segments.
    """
    env = _frame_energies_db(audio)
    if _is_silent(env):
        return 0.0

    seg_frames = 50  # 0.5 s of 10 ms hops
    floors = [
        float(np.percentile(env[s : s + seg_frames], 10))
        for s in range(0, len(env) - seg_frames + 1, seg_frames)
    ]
    if len(floors) < 2:
        return 0.0
    spread_db = float(np.max(floors) - np.min(floors))
    return float(np.clip((spread_db - 6.0) / 12.0, 0.0, 1.0))

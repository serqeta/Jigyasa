from __future__ import annotations

import time
from typing import Callable

import numpy as np

from voiceshield import config
from voiceshield.audio.buffer import RollingBuffer
from voiceshield.audio.source import AudioSource
from voiceshield.classifier.protocol import Scorer
from voiceshield.features import GateState
from voiceshield.features.artifact import top_artifact_name
from voiceshield.features.scalars import compute_suspicion_features
from voiceshield.features.snr import compute_gate_decision, estimate_snr
from voiceshield.features.vad import compute_vad
from voiceshield.logger import StructuredLogger
from voiceshield.pipeline.state_engine import StateEngine
from voiceshield.pipeline.timeline import Timeline, TimelineEntry


def _extract_features(audio: np.ndarray) -> dict[str, float]:
    """Compute scalar suspicion features used by FallbackScorer and top_artifact."""
    return compute_suspicion_features(audio)


def _voiced_ratio(audio: np.ndarray) -> float:
    """Fraction of frames in the window that contain speech energy."""
    mask = compute_vad(audio, threshold_db=config.VAD_THRESHOLD_DB)
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask))


def _extract_rich_visuals(audio: np.ndarray) -> dict[str, list]:
    """Extract downsampled linear/mel/cqt spectrograms, pitch and phase contours for visualization."""
    import librosa

    from voiceshield.features.pitch import compute_f0
    from voiceshield.features.spectrogram import compute_cqt, compute_linear_stft, compute_mel

    # 1. 2D Spectrograms (dB scaled)
    spec_linear = compute_linear_stft(audio)
    spec_mel = compute_mel(audio)
    spec_cqt = compute_cqt(audio)

    # Downsample to keep payload size tiny (approx 65x50, 40x50, 42x50)
    spec_linear_down = spec_linear[::4, ::2]
    spec_mel_down = spec_mel[::2, ::2]
    spec_cqt_down = spec_cqt[::2, ::2]

    # 2. Pitch contour (1D)
    f0 = compute_f0(audio)
    pitch_contour = [float(x) if not np.isnan(x) else None for x in f0[::2]]

    # 3. Phase contour (1D)
    stft = librosa.stft(audio, n_fft=512, hop_length=160)
    phase = np.angle(stft)
    unwrapped = np.unwrap(phase, axis=1)
    d2_phase = np.diff(unwrapped, n=2, axis=1)
    phase_frames = np.mean(np.abs(d2_phase), axis=0)
    phase_contour = (np.clip(phase_frames / (2 * np.pi), 0.0, 1.0)).astype(float)
    phase_contour_down = phase_contour[::2].tolist()

    return {
        "spec_linear": spec_linear_down.tolist(),
        "spec_mel": spec_mel_down.tolist(),
        "spec_cqt": spec_cqt_down.tolist(),
        "pitch_contour": pitch_contour,
        "phase_contour": phase_contour_down,
    }


class PipelineRunner:
    """
    Ties AudioSource → RollingBuffer → SNR gate → features →
    Scorer → StateEngine → Timeline.
    """

    def __init__(self, source: AudioSource, scorer: Scorer) -> None:
        self._source = source
        self._scorer = scorer
        self._buffer = RollingBuffer()
        self._state_engine = StateEngine()
        self._timeline = Timeline()
        self._log = StructuredLogger("voiceshield.runner")

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    @property
    def state_engine(self) -> StateEngine:
        return self._state_engine

    def run_once(self) -> TimelineEntry:
        """Process one 500 ms chunk and return the resulting TimelineEntry."""
        t0 = time.perf_counter()

        chunk = self._source.read_chunk()
        chunk_idx, t_start, t_end = self._buffer.push(chunk)

        # SNR gate over the latest 1 second
        audio_1s = self._buffer.latest_seconds(1.0)
        snr_db = estimate_snr(audio_1s)
        gate: GateState = compute_gate_decision(snr_db)

        # Feature window — pad with zeros if buffer not yet full
        audio_win = self._buffer.latest_seconds(config.STAGE1_WINDOW_SECONDS)
        if len(audio_win) == 0:
            audio_win = chunk

        # speech_active is derived purely from the SNR gate.
        # The SNR estimator (95th–10th percentile spread) already discriminates
        # speech from silence reliably; a frame-level VAD can't reliably add
        # signal because a window full of speech has no clean noise-floor estimate.
        # voiced_ratio is kept as a logged diagnostic only.
        voiced_ratio = _voiced_ratio(audio_win)
        speech_active = gate != GateState.GREY

        from voiceshield.classifier.fallback import FallbackScorer

        if not speech_active:
            score = 0.0
            artifact = None
        else:
            # Extract features once; FallbackScorer reuses them, AASIST uses raw audio
            features = _extract_features(audio_win)
            if isinstance(self._scorer, FallbackScorer):
                score = self._scorer.score_from_features(features)
            else:
                score = self._scorer.score(audio_win)
            artifact = top_artifact_name(features)

        # State engine
        risk_state = self._state_engine.update(score, gate, t_end)

        # Skip expensive visual extraction on GREY frames (no usable speech).
        # REDUCED-gate frames still carry real speech and get visuals.
        if gate != GateState.GREY:
            visuals = _extract_rich_visuals(audio_win)
        else:
            visuals = {
                "spec_linear": [],
                "spec_mel": [],
                "spec_cqt": [],
                "pitch_contour": [],
                "phase_contour": [],
            }

        entry = TimelineEntry(
            time=t_end,
            score=score,
            state=risk_state.value,
            snr_db=snr_db,
            top_artifact=artifact,
            speech_active=speech_active,
            voiced_ratio=round(voiced_ratio, 3),
            first_amber_t=self._state_engine.first_amber_t,
            first_red_t=self._state_engine.first_red_t,
            spec_linear=visuals["spec_linear"],
            spec_mel=visuals["spec_mel"],
            spec_cqt=visuals["spec_cqt"],
            pitch_contour=visuals["pitch_contour"],
            phase_contour=visuals["phase_contour"],
        )
        self._timeline.append(entry)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log.info(
            "chunk_processed",
            {
                "chunk_idx": chunk_idx,
                "t_end": t_end,
                "score": round(score, 4),
                "state": risk_state.value,
                "snr_db": round(snr_db, 1),
                "voiced_ratio": round(voiced_ratio, 2),
                "latency_ms": round(latency_ms, 1),
            },
        )
        return entry

    def run_forever(self, callback: Callable[[TimelineEntry], None]) -> None:
        """Process chunks until EOFError (FileSource exhausted) or KeyboardInterrupt."""
        while True:
            try:
                entry = self.run_once()
                callback(entry)
            except EOFError:
                break

    def reset(self) -> None:
        """Reset the state engine and timeline for a fresh analysis."""
        self._state_engine = StateEngine()
        self._timeline = Timeline()

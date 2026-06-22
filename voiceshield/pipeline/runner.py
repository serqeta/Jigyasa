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
from voiceshield.features.flux import compute_spectral_flux
from voiceshield.features.phase import compute_phase_discontinuity
from voiceshield.features.pitch import compute_f0, compute_pitch_smoothness
from voiceshield.features.snr import compute_gate_decision, estimate_snr
from voiceshield.features.subband import compute_subband_energy
from voiceshield.logger import StructuredLogger
from voiceshield.pipeline.state_engine import StateEngine
from voiceshield.pipeline.timeline import Timeline, TimelineEntry


def _extract_features(audio: np.ndarray) -> dict[str, float]:
    """Compute scalar forensic features used by FallbackScorer and top_artifact."""
    phase = compute_phase_discontinuity(audio)
    f0 = compute_f0(audio)
    pitch_var = compute_pitch_smoothness(f0)
    pitch_inv = 1.0 / (1.0 + pitch_var)
    flux = compute_spectral_flux(audio)
    flux_inv = 1.0 / (1.0 + flux)
    subband = compute_subband_energy(audio)
    formant_db = subband.get("formants", -80.0)
    artifact_db = subband.get("artifact", -80.0)
    artifact_ratio = max(0.0, artifact_db - formant_db + 20.0) / 20.0

    return {
        "phase_discontinuity": phase,
        "pitch_smoothness_inv": pitch_inv,
        "artifact_ratio": artifact_ratio,
        "flux_variance_inv": flux_inv,
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

        # Extract features once; FallbackScorer reuses them, AASISTScorer uses raw audio
        from voiceshield.classifier.fallback import FallbackScorer

        features = _extract_features(audio_win)
        if isinstance(self._scorer, FallbackScorer):
            score = self._scorer.score_from_features(features)
        else:
            score = self._scorer.score(audio_win)

        # State engine
        risk_state = self._state_engine.update(score, gate, t_end)

        # Top artifact
        artifact = top_artifact_name(features)

        entry = TimelineEntry(
            time=t_end,
            score=score,
            state=risk_state.value,
            snr_db=snr_db,
            top_artifact=artifact,
            first_amber_t=self._state_engine.first_amber_t,
            first_red_t=self._state_engine.first_red_t,
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

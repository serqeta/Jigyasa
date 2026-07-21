from __future__ import annotations

import time
from collections import deque
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
from voiceshield.pipeline.explain import explain_verdict
from voiceshield.pipeline.state_engine import StateEngine, fuse_scores
from voiceshield.pipeline.timeline import Timeline, TimelineEntry


def _extract_features(audio: np.ndarray) -> dict[str, float]:
    """Compute scalar suspicion features used by FallbackScorer and top_artifact."""
    return compute_suspicion_features(audio)


def _voiced_ratio(audio: np.ndarray) -> float:
    """Fraction of the window that contains speech.

    Uses Silero-VAD when enabled (robust in noise); otherwise the
    energy-threshold VAD."""
    if config.USE_SILERO_VAD:
        from voiceshield.features.silero_vad import voiced_ratio

        return voiced_ratio(audio)
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

    def __init__(
        self,
        source: AudioSource,
        scorer: Scorer | None = None,
        ensemble: dict[str, Scorer] | None = None,
        cascade: bool = False,
    ) -> None:
        """
        Single-scorer mode (Stage 1, `scorer=`) and ensemble mode
        (Stage 2, `ensemble=` from classifier.get_scorers()) share one
        code path: a lone scorer becomes the "stage1" component and the
        fusion of one component is the score itself.

        cascade=True (live streaming): only the "stage1" component scores
        every chunk; the rest of the ensemble engages when stage1 crosses
        config.CASCADE_TRIGGER (plus a periodic deep probe) and disengages
        after config.CASCADE_COOLDOWN_CHUNKS clean chunks.
        """
        if ensemble is None:
            if scorer is None:
                raise ValueError("PipelineRunner needs a scorer or an ensemble")
            ensemble = {"stage1": scorer}
        self._ensemble = ensemble
        self._cascade = cascade and len(ensemble) > 1
        self._stage2_active = False
        self._cascade_clean = 0
        self._chunks_since_probe = 0
        self._score_history: deque[float] = deque(maxlen=config.SCORE_SMOOTHING_CHUNKS)
        self._drift_tracker = None
        if config.ENABLE_SPEAKER_DRIFT:
            from voiceshield.analysis import SpeakerDriftTracker

            self._drift_tracker = SpeakerDriftTracker()
        self._buffer = RollingBuffer()
        self._source = source
        self._state_engine = StateEngine()
        self._timeline = Timeline()
        self._log = StructuredLogger("voiceshield.runner")

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    @property
    def buffer(self) -> RollingBuffer:
        return self._buffer

    @property
    def source(self) -> AudioSource:
        return self._source

    @property
    def state_engine(self) -> StateEngine:
        return self._state_engine

    def _should_run_deep(self, screen_score: float) -> tuple[bool, bool]:
        """Cascade gate → (deep, is_probe): deep-scan when engaged, when the
        screener flags suspicion, or on the periodic probe that guards
        against a screener miss."""
        if self._stage2_active or screen_score >= config.CASCADE_TRIGGER:
            self._chunks_since_probe = 0
            return True, False
        self._chunks_since_probe += 1
        if self._chunks_since_probe >= config.CASCADE_PROBE_EVERY:
            self._chunks_since_probe = 0
            return True, True
        return False, False

    def _update_cascade(self, fused: float) -> None:
        """Engage on suspicious fused score; disengage after sustained clean."""
        if fused >= config.SCORE_AMBER:
            self._stage2_active = True
            self._cascade_clean = 0
        elif self._stage2_active:
            self._cascade_clean += 1
            if self._cascade_clean >= config.CASCADE_COOLDOWN_CHUNKS:
                self._stage2_active = False
                self._cascade_clean = 0

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

        # speech_active requires BOTH signals:
        # - SNR gate (1 s window) — discriminates speech from silence/noise, but
        #   lags up to 1 s after speech stops because its window still holds
        #   speech energy.
        # - chunk-level VAD on the newest 500 ms — silence there means no NEW
        #   evidence arrived, so scoring the (stale) 4 s window would just
        #   re-judge old audio and could escalate the state during silence.
        voiced_ratio = _voiced_ratio(audio_win)
        chunk_voiced = _voiced_ratio(chunk)
        speech_active = gate != GateState.GREY and chunk_voiced >= config.MIN_VOICED_RATIO

        component_scores: dict[str, float] = {}
        replay_result = None
        drift = {"drift": 0.0, "speaker_changed": False}

        if not speech_active:
            score = 0.0
            artifact = None
            self._score_history.clear()  # silence breaks the smoothing window
        else:
            # Extract features once; rule-based scorers reuse them, neural
            # scorers consume raw audio.
            features = _extract_features(audio_win)

            def _component(scorer: Scorer) -> float:
                if hasattr(scorer, "score_from_features"):
                    return scorer.score_from_features(features)
                return scorer.score(audio_win)

            # Screen with the configured screener (falling back to the
            # legacy "stage1" slot); an ensemble without a screener always
            # deep-scans (screen score 1.0 ≥ any trigger).
            screen_name = (
                config.CASCADE_SCREENER if config.CASCADE_SCREENER in self._ensemble else "stage1"
            )
            screener = self._ensemble.get(screen_name)
            if screener is not None:
                component_scores[screen_name] = _component(screener)
            screen_score = component_scores.get(screen_name, 1.0)

            if self._cascade:
                run_deep, is_probe = self._should_run_deep(screen_score)
            else:
                run_deep, is_probe = True, False

            # Confidence gate (low side): a confidently-clean screener on a
            # non-probe chunk makes the diversity models redundant. Probes
            # and suspicious chunks always run the full ensemble, so
            # instant-RED keeps requiring multi-model consensus.
            gated = (
                self._cascade
                and not is_probe
                and not self._stage2_active
                and screen_score < config.CONFIDENCE_GATE_LOW
            )
            if run_deep and not gated:
                for name, scorer in self._ensemble.items():
                    if name not in component_scores:
                        component_scores[name] = _component(scorer)
                # The learned replay scorer (if loaded) is a normal ensemble
                # member, already scored above as component_scores["replay"].
                # The legacy DSP replay module is superseded and not wired
                # (zero-weight, non-discriminative — see docs/REPLAY_FINDINGS.md).
                if "replay" in component_scores:
                    replay_result = {
                        "score": round(component_scores["replay"], 4),
                        "model": "echofake-lora",
                    }

            raw_score = fuse_scores(component_scores)
            # Weak-evidence scaling: mostly-silent windows (speech onsets)
            # make the SSL models spike; evidence grows with voiced content.
            raw_score *= min(1.0, voiced_ratio / config.EVIDENCE_FULL_VOICED_RATIO)

            # Temporal smoothing: median over the recent speech chunks kills
            # single-chunk transients (both false blips and evidence noise).
            self._score_history.append(raw_score)
            score = float(np.median(self._score_history))

            # Cascade engagement reacts to RAW evidence (a single hot probe
            # must engage stage 2 so the next chunks can confirm it); only
            # the user-facing state is smoothed.
            if self._cascade and run_deep:
                self._update_cascade(raw_score)
            artifact = top_artifact_name(features)

            # Speaker-consistency: is this still the same voice as the call
            # started with? Advisory only — never feeds the spoof score.
            if self._drift_tracker is not None:
                d = self._drift_tracker.update(audio_win)
                if d["ready"]:
                    drift = d

        # State engine sees GREY whenever there is no fresh speech, whatever
        # the SNR gate said — silence must read GREY, never held/escalated risk.
        effective_gate = gate if speech_active else GateState.GREY
        risk_state = self._state_engine.update(score, effective_gate, t_end)

        # Explain *why* this verdict — a faithful decomposition of the fusion
        # (which detector drove it, consensus vs. peak-evidence, top cue).
        explanation = explain_verdict(
            component_scores,
            score,
            risk_state.value,
            top_artifact=artifact,
            speaker_changed=bool(drift["speaker_changed"]),
        )

        # Skip expensive visual extraction when no fresh speech arrived.
        # REDUCED-gate frames still carry real speech and get visuals.
        if speech_active:
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
            component_scores={k: round(v, 4) for k, v in component_scores.items()},
            replay=replay_result,
            stage2_active=(not self._cascade) or self._stage2_active,
            speaker_drift=float(drift["drift"]),
            speaker_changed=bool(drift["speaker_changed"]),
            first_amber_t=self._state_engine.first_amber_t,
            first_red_t=self._state_engine.first_red_t,
            spec_linear=visuals["spec_linear"],
            spec_mel=visuals["spec_mel"],
            spec_cqt=visuals["spec_cqt"],
            pitch_contour=visuals["pitch_contour"],
            phase_contour=visuals["phase_contour"],
            explanation=explanation,
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
        """Reset the state engine, timeline, and cascade for a fresh analysis."""
        self._state_engine = StateEngine()
        self._timeline = Timeline()
        self._stage2_active = False
        self._cascade_clean = 0
        self._chunks_since_probe = 0
        self._score_history.clear()
        if self._drift_tracker is not None:
            self._drift_tracker.reset()

# Architecture

## Purpose

VoiceShield detects synthetic/cloned voices on live calls in real time:
500 ms chunks, a 10 s in-memory rolling buffer (no persistent recording),
an ensemble of anti-spoofing scorers fused into one risk score, and an
advisory dashboard. Stage 1 = single AASIST scorer + /v1/ API; Stage 2 =
pretrained ensemble + replay detection + risk fusion + evidence export
+ /v2/ API.

## Boundaries

| Boundary | Input | Output | Owner |
|---|---|---|---|
| Audio capture | mic / file | float32 mono 16 kHz chunks (8000,) | `voiceshield/audio` |
| Quality gate | latest 1 s window | SNR dB + GateState (NORMAL/REDUCED/GREY) | `voiceshield/features` (vad, snr) |
| Forensic features | 4 s window | suspicion scalars + visuals | `voiceshield/features` |
| Classifiers | 4 s window | per-component spoof score [0,1] | `voiceshield/classifier` |
| Replay detection | 4 s window | 4 detector scores + combined [0,1] | `voiceshield/replay` |
| Risk fusion + state | component score dict + gate | fused score, RiskState w/ hysteresis | `voiceshield/pipeline/state_engine` |
| Timeline | TimelineEntry per chunk | 20-entry ring (10 s) | `voiceshield/pipeline/timeline` |
| API | pipeline state | /v1 (frozen), /v2 (ensemble), WS stream | `voiceshield/api` |
| Evidence | rolling buffer + timeline | PNGs + SHA-256 + JSON package (pull-only) | `voiceshield/evidence` |
| Dashboard | /v2 REST + WS | advisory UI | `Interface/` (React) |

## Data Shape Contracts

- All audio inside the pipeline is float32 mono 16 kHz in [-1, 1].
- Every scorer implements `Scorer.score(audio) -> float` in [0, 1],
  spoof-positive. HF label polarity is normalized inside `HFScorer` via
  `config.HF_SCORERS[*]["spoof_label"]`.
- The state engine consumes an arbitrary `dict[str, float]` of component
  scores (score-source-agnostic); fusion weights renormalize over the
  components present.
- GREY gate wins over every score and every previously escalated state;
  escalation requires fresh speech in the newest chunk (chunk-level VAD).

## Module Ownership Rules

- One primary responsibility per module.
- No cross-layer shortcuts without explicit architecture update.
- `/v1/` responses are frozen; ensemble-aware changes go to `/v2/`.
- Model checkpoints live under `models/` (git-ignored); licenses in NOTICE.md.

## Execution Flow

1. Entry: `scripts/run_file.py` / `scripts/run_live.py` (ensemble by default,
   `--stage1` for single-scorer) → `create_app(runner)`.
2. Boundary parse/validate: AudioSource normalizes to 16 kHz mono chunks.
3. Core execution: RollingBuffer → SNR gate + chunk VAD → features →
   ensemble scorers + replay → fuse_scores → StateEngine → TimelineEntry.
4. Persistence/output: none by default (privacy G10); evidence package only
   on explicit POST /v2/evidence/export.
5. Event/log emission: one structured JSON log line per chunk
   (`chunk_processed` with score, state, snr_db, latency_ms).

## Refactor Checklist

- [ ] Boundary contracts unchanged or versioned.
- [ ] Ownership map still accurate.
- [ ] Integration tests cover boundary paths.
- [ ] Documentation updated in same change.

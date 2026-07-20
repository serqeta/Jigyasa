# PLANS.md

Use this file for multi-step work where durable context matters.

> Stage 1 (all phases 0–10) is complete: pipeline, AASIST-L native inference,
> state engine, /v1/ API, React Interface, 42 unit + 5 integration tests green.
> Remaining Stage 1 debt: e2e acceptance fixtures absent (tests skip), formal
> p95 benchmark unrun. Tracked, not blocking Stage 2.

## Objective

- Outcome: Build Stage 2 (“Enhanced Prototype”): a multi-model ensemble
  (AASIST + pretrained SSL anti-spoof scorer + pretrained spectrogram
  anti-spoof scorer + rule-based PhasePitch scorer), DSP-based replay
  detection, weighted risk fusion across all components, an evidence export
  package (spectrogram PNG, phase heatmap PNG, SHA-256 audio hash, scores
  JSON envelope), a versioned /v2/ API, and Interface panels for component
  scores, replay verdict, and evidence download.
- Why it matters: Stage 1’s single-model score is brittle against top-tier
  TTS (risk R3). The ensemble + replay detection harden detection; evidence
  export makes alerts auditable.
- Non-goals: SIP/VoIP integration, multi-tenant orchestration, K8s,
  Indian phoneme analysis, federated learning, liveness veto.

## Constraints

- Runtime: Python 3.12 venv (.venv, uv-managed), CUDA torch on RTX 4050
  Laptop GPU (6 GB VRAM) — all Stage 2 models must fit alongside AASIST;
  fall back to CPU transparently (config.DEVICE auto-detect).
- Models: real pretrained anti-spoofing checkpoints only (user directive:
  “download the models and their weights, don’t skip”). No decorative
  random-weight scorers. Licenses recorded in NOTICE.md.
- Privacy: evidence export ONLY on explicit request (G10 — manual
  save-for-audit); rolling buffer remains the only routine audio storage.
- Compatibility: /v1/ contract frozen; Stage 2 additions live under /v2/.
  Scorer protocol and compute_final(dict) seams are the integration points.
- Performance: per-chunk budget stays < 200 ms p95 with GPU inference;
  ensemble models score on the 4 s rolling window like AASIST.

## Context Snapshot

- Seams: `classifier/protocol.py` (Scorer), `classifier/__init__.py`
  (factory), `pipeline/state_engine.py` (`StateEngine.update`,
  `compute_final(component_scores, gate)`), `pipeline/runner.py`
  (single-scorer today → ensemble), `pipeline/timeline.py` (TimelineEntry),
  `api/rest.py` (+`api/ws.py`), `Interface/src/` React app.
- Model acquisition research: background agent surveying HF Hub for
  verified SSL-based and spectrogram-based anti-spoof checkpoints.
- Known risks: 6 GB VRAM ceiling; HF checkpoint label-orientation traps
  (which class = spoof); ensemble latency; MP3-compression false positives
  interacting with double-compression replay detector.

## Execution Plan

1. **Phase S2.0: GPU + deps foundation** — CUDA torch/torchaudio,
   `transformers`, config.DEVICE auto-detect, AASIST moved to DEVICE,
   requirements.txt updated.
   - Verification: `torch.cuda.is_available()` true; existing 42 unit tests
     still green; AASIST scores identically (±1e-4) on CPU vs GPU fixture.

2. **Phase S2.1: Model acquisition** — download verified pretrained
   checkpoints (SSL slot + spectrogram slot) via extended
   `scripts/download_model.py`; NOTICE.md licenses; smoke-load both.
   - Verification: both load on GPU, produce [0,1] scores on fixtures,
     genuine vs TTS fixture scores separate in the right direction.

3. **Phase S2.2: Ensemble scorers** — `classifier/ssl_scorer.py`,
   `classifier/spec_scorer.py`, `classifier/phase_pitch.py`; factory
   `get_scorers() -> dict[str, Scorer]` (name → scorer, load-if-available);
   runner scores all components per chunk.
   - Verification: unit tests per scorer ([0,1] range, orientation checks,
     protocol conformance); TEST-INT ensemble timeline run.

4. **Phase S2.3: Replay detection** — `voiceshield/replay/` with reverb
   tail, frequency-response anomaly, double-compression, background
   consistency detectors → combined `replay_score`.
   - Verification: synthetic fixtures (dry vs artificially reverbed,
     full-band vs band-limited, once vs twice-compressed) separate correctly.

5. **Phase S2.4: Risk fusion** — config.FUSION_WEIGHTS; state engine fuses
   weighted component dict (weights renormalized over available components),
   GREY override, hysteresis, 0.85 short-circuit, latching all preserved.
   - Verification: fusion unit tests incl. missing-component renormalization;
     all existing state-engine tests still green.

6. **Phase S2.5: Evidence export** — `voiceshield/evidence/export.py`:
   spectrogram PNG + phase heatmap PNG + SHA-256(buffer) + JSON envelope
   (component scores, timeline, config snapshot) to `evidence/` on request.
   - Verification: unit test asserts package contents + hash correctness;
     no evidence written during normal pipeline runs.

7. **Phase S2.6: /v2/ API** — `/v2/risk/current`, `/v2/risk/timeline`
   (component_scores + replay_score enriched), `/v2/analyze`,
   `POST /v2/evidence/export`; WS payload gains component scores.
   - Verification: API tests for all /v2/ routes; /v1/ responses
     byte-compatible with Stage 1 tests.

8. **Phase S2.7: Interface** — component score breakdown bars, replay
   indicator, evidence export button; switch data layer to /v2/.
   - Verification: manual smoke — TTS fixture shows per-model bars moving,
     evidence button downloads package.

9. **Phase S2.8: Validation & docs** — `make check` + `make test` green,
   GPU latency benchmark, docs/ARCHITECTURE.md rewritten with real content,
   tasks/progress.md + README updated.

## Checkpoints

- [x] Stage 1 baseline verified (tests green, AASIST native, lint fixed)
- [x] S2.0 GPU foundation (torch 2.12.1+cu130, RTX 4050; numpy pinned <2.5 for numba)
- [x] S2.1 models downloaded + smoke-loaded (XLS-R 1.2G, AST 329M, WavLM 361M; NOTICE.md)
- [x] S2.2 ensemble scorers (HFScorer generic + PhasePitchScorer; get_scorers factory)
- [x] S2.3 replay detection (4 DSP detectors, unit-tested on synthetic fixtures)
- [x] S2.4 fusion (weighted renormalizing mean; compute_final spec-consistent)
- [x] S2.5 evidence export (PNGs + SHA-256 + JSON envelope, pull-only per G10)
- [x] S2.6 /v2/ API (models/current/timeline/analyze/evidence + /v2/ws/risk)
- [x] S2.7 Interface (EnsemblePanel, evidence button, /v2 data layer; vite build green)
- [x] S2.8 validation: 83 tests green, make check green, ensemble p95 = 166 ms (<200 gate)
- [x] Post-review fix: GREY-wins state engine (silence after AMBER read AMBER and
      could escalate to RED off the stale 4 s window; now chunk-level VAD gates
      fresh evidence and GREY always displays during silence — regression-tested)

## Decision Log

- 2026-07-08 — Use real pretrained anti-spoof checkpoints for SSL and
  spectrogram ensemble slots (user has GPU; directive to not skip models).
  Selection deferred to verified research results, not memory.
- 2026-07-08 — Fusion = weighted mean over available component scores with
  renormalization, preserving Stage 1 GREY/hysteresis/short-circuit
  semantics in StateEngine; component dict is score-source-agnostic per the
  Stage-2 hand-off contract.
- 2026-07-08 — Evidence export is pull-only (explicit API call), never
  automatic, to preserve the G10 privacy guarantee.

## Final Verification

- Commands run: `make check`, `make test`, `make smoke`, GPU latency bench.
- Key outputs: TBD
- Follow-up tasks: Stage 1 e2e fixtures still needed for the formal
  acceptance gates (TEST-E2E10.x) — now doubly useful to validate ensemble.

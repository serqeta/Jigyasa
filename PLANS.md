# PLANS.md

Use this file for multi-step work where durable context matters.

## Objective

- Outcome: Build the end-to-end VoiceShield Stage 1 MVP pipeline. It captures live/file audio, processes it in 500 ms chunks held in a 10s rolling buffer, applies a quality gate, runs a classifier (AASIST/Fallback), determines risk states (GREEN, AMBER, RED, GREY), and streams the result to a vanilla HTML/JS advisory dashboard.
- Why it matters: This delivers the headline promise of the project: flagging synthetic/cloned voices within 10 seconds of a live call while preserving privacy.
- Non-goals: Stage 2 ensemble (WavLM, CNNs), replay detection, comprehensive evidence export, SIP/VoIP integration.

## Constraints

- Runtime/tooling constraints: Python 3.11+, FastAPI, Vanilla HTML/JS/CSS. CPU-only inference (no GPUs needed).
- Security/compliance constraints: **No persistent call recording** (privacy preservation via rolling buffer).
- Performance/reliability constraints: < 200 ms p95 processing latency per 500ms chunk.
- Harness constraints: Strict boundaries via `make check`, `make test`, structured logging with `trace_id` & `run_id`.
- Design Constraints: UI folder is `voiceshield/ui/`. Light glassmorphism only on the outer card; the badge stays solid and high-contrast. Demo defaults to `run_file.py`.

## Context Snapshot

- Relevant files/modules: `voiceshield/config.py`, `audio/`, `features/`, `classifier/`, `pipeline/`, `api/`, `voiceshield/ui/`.
- Existing commands/workflows: `scripts/harness/smoke.sh`, `scripts/harness/test.sh`, `scripts/audit_harness.sh`, `make smoke`, `make check`, `make test`.
- Known risks: Latency budget could be tight; hysteresis logic might delay true positives. Short-circuit at 0.85 mitigates this.

## Execution Plan

1. **Phase 0: Foundation & Harness Initialization**
   - Expected output: Configured `voiceshield/config.py` and `voiceshield/logger.py` with structured JSON logging matching Harness Observability rules. `USE_FALLBACK_CLASSIFIER=False`.
   - Verification: `TEST-S0.3`, `TEST-S0.4` pass.

2. **Phase 1 & 2: Audio Capture & Rolling Buffer**
   - Expected output: `AudioSource`, `MicSource`, `FileSource`, and `RollingBuffer` (160k samples max).
   - Verification: `make test` runs `TEST-A1.1` to `TEST-B2.3` successfully.

3. **Phase 3: SNR Quality Gate**
   - Expected output: Energy-based VAD and SNR estimation. Gate outputs `NORMAL`, `REDUCED`, or `GREY`.
   - Verification: `make test` on `TEST-Q3.1` to `TEST-Q3.3`.

4. **Phase 4: Forensic Feature Extraction**
   - Expected output: Spectrograms, Phase Heatmap, Pitch, Flux, MFCCs (39, T), Subband Energy.
   - Verification: `make test` on `TEST-F4.1` to `TEST-F4.9` with synthetic fixtures.

5. **Phase 5: Classifier (AASIST & Fallback)**
   - Expected output: Factory pattern in `voiceshield/classifier/__init__.py`. Load AASIST-L weights natively. Fallback only if absent. Score mapping to `GREEN`, `AMBER`, `RED`.
   - Verification: Score is within [0.0, 1.0]. `make test` on `TEST-C5.1` through `TEST-C5.6`.

6. **Phase 6 & 7: Timeline & Risk State Engine**
   - Expected output: 10s ring buffer (`TimelineEntry`). Risk hysteresis logic: `GREY` override (highest priority), `SCORE_RED_OVERRIDE = 0.85` single-chunk short-circuit, `first_amber_t`/`first_red_t` latching.
   - Verification: `make test` on `TEST-T6.1` to `TEST-R7.3`.

7. **Phase 8: Streaming API**
   - Expected output: FastAPI application with v1 routes: `/healthz`, `/v1/risk/current`, `/v1/risk/timeline`, `/v1/ws/risk`.
   - Verification: `make test` on `TEST-API8.1` to `TEST-API8.4`.

8. **Phase 9: Advisory Dashboard**
   - Expected output: Sleek Vanilla HTML/JS UI in `voiceshield/ui/`. Features light outer glassmorphism, solid contrast badge, sparklines, and WebSocket updates.
   - Verification: Dashboard dynamically reflects states within 1 chunk.

9. **Phase 10: E2E Validation & Acceptance**
   - Expected output: Typer/Argparse CLIs (`run_file.py` default, `run_live.py` separate). Pipeline passes all 5 curated `.wav` fixtures.
   - Verification: Mandatory acceptance gates: `cloned-voice` test (`first_amber_t` ≤ 5 s and `first_red_t` ≤ 10 s) and the p95 latency benchmark.

## Checkpoints

- [x] Baseline captured (Repository layout, AGENTS.md setup)
- [ ] Implementation complete
- [ ] Static checks passed (`make check`)
- [ ] Tests passed (`make test`)
- [ ] Docs updated (`docs/ARCHITECTURE.md`, `docs/OBSERVABILITY.md`)

## Decision Log

- Date: TBD
  - Decision: Use Factory Pattern in `voiceshield/classifier/__init__.py`.
  - Reason: AASIST is primary model; fallback is safety net. Ensures correct evaluation path.

## Final Verification

- Commands run: `make check`, `make test`, `make smoke`, `scripts/audit_harness.sh .`
- Key outputs: TBD
- Follow-up tasks: TBD

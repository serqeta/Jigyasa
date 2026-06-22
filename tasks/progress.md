# Progress

---

## Phase 0 — Setup *(session 1)*

### Actions taken
- Created repository directory structure based on Section 5 of `planning.txt`.
- Created `Interface/` directory alongside `voiceshield/` for frontend separation.
- Checked `SETUP.md` and adapted the setup specifically for the Antigravity system (`.agents/AGENTS.md` and `tasks/` memory).

### Files modified
- Created `voiceshield/` and `Interface/` structure.
- Created `.agents/AGENTS.md` for project context and rules.
- Created `tasks/todo.md`, `tasks/lessons.md`, `tasks/findings.md`, `tasks/progress.md` for memory persistence.

### Issues encountered
- Adapting Claude-specific instructions to Antigravity style. Resolution: Mapped `.claude` to `.agents` and `CLAUDE.md` to `AGENTS.md`.

---

## Phases 1–9 — Core Pipeline + React Interface *(sessions 2–N)*

### Actions taken
- Implemented all audio, feature, classifier, pipeline, and API layers
- Built complete React Interface with VoiceShieldContext, NavBar, SourcePanel, RiskDisplay, Sparkline, TimelineTable, StatusBar
- API exposes `POST /v1/analyze`, `GET /healthz`, `GET /v1/risk/current`, `GET /v1/risk/timeline`, `WS /v1/ws/risk`, `POST /v1/reset`

### Files created/modified
- `voiceshield/config.py`, `voiceshield/logger.py`
- `audio/source.py`, `audio/buffer.py`, `audio/resample.py`
- `features/` — all 9 forensic features + artifact.py
- `classifier/` — protocol, state_mapper, fallback, aasist_model, aasist_loader, aasist_inference, __init__ factory
- `pipeline/timeline.py`, `pipeline/state_engine.py`, `pipeline/runner.py`
- `api/app.py`, `api/rest.py`, `api/ws.py`
- `Interface/src/` — all components + VoiceShieldContext

### Issues encountered
- None recorded

---

## Phase 10 — Performance Optimisations *(2026-06-23)*

### Problem
5-second audio file was taking 9+ seconds to analyze. Three compounding root causes:

1. **Double feature extraction** — `_run_analysis` extracted features twice per chunk (once for score, once for artifact). Fixed by `score_from_features()` on FallbackScorer and extracting once in `runner.py`.
2. **`librosa.to_mono` numba cold-start** — 4.1s penalty on first call per process due to numba JIT compilation. Fixed by replacing with `audio.mean(axis=0)` in `audio/resample.py`.
3. **`librosa.yin` numba cold-start** — 3.6s penalty on first chunk. Fixed by:
   - `_warmup_jit()` called at FastAPI startup (`api/app.py`)
   - Lazy warmup in `_run_analysis` guarded by `_jit_warmed` flag (`api/rest.py`)

### Actions taken
- `audio/resample.py`: replaced `librosa.to_mono` with numpy `audio.mean(axis=0)`, replaced `librosa.resample` with `scipy.signal.resample_poly`
- `classifier/fallback.py`: added `score_from_features(features: dict) -> float` method
- `pipeline/runner.py`: extract features once, pass to `score_from_features()` if FallbackScorer
- `api/app.py`: added `_warmup_jit()` called in `startup` event
- `api/rest.py`: added `_jit_warmed` global flag + lazy warmup in `_run_analysis`

### Result
5s audio: 9+ seconds → **~917ms** post-warmup

---

## Phase 11 Planning — Forensic Dashboard *(2026-06-23)*

### Context
User uploaded their own voice (sample.mp3, 6s), got GREY then AMBER result with score 0.32–0.41. Correctly identified as false positive — MP3 compression artifacts trigger `phase_discontinuity` feature (weight 0.35, too high for non-WAV input).

User then specified 6 forensic enhancement features for the Interface dashboard:
1. Phase discontinuity heatmap (canvas waterfall) — *planned, not implemented*
2. Component score breakdown stacked bar (phase/pitch/subband/flux) — *planned, not implemented*
3. Pitch F0 contour line chart — *planned, not implemented*
4. Subband energy mini-bars (5 bands) — *planned, not implemented*
5. "Why this state" caption under risk badge — *planned, not implemented*
6. Audio provenance chip in NavBar (codec, SR, duration, SHA-256) — *planned, not implemented*

### Status
All 6 features planned in detail in `tasks/todo.md` Phase 11. Backend must be extended first (expose F0 contour, subband energies, component scores, state reason, provenance in API responses), then frontend components can be built. No code written yet.

### Next session entry point
Start with `11A — Backend` tasks in `tasks/todo.md`. Begin with `pipeline/runner.py` → `pipeline/timeline.py` → `api/rest.py` → then frontend.

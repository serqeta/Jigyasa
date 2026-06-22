# VoiceShield — Task Tracker

## Status key
`done` · `in_progress` · `pending` · `blocked`

---

## Phase 0 — Foundation & Setup
**Status: done**

- [x] `voiceshield/config.py` — all constants (SR, thresholds, window sizes)
- [x] `voiceshield/logger.py` — structured JSON logger with trace_id / run_id
- [x] `.agents/AGENTS.md` — project rules and harness commands
- [x] `tasks/` memory system (todo, progress, lessons, findings)

---

## Phase 1–2 — Audio Capture & Rolling Buffer
**Status: done**

- [x] `AudioSource` ABC + `FileSource` (WAV/MP3 via soundfile)
- [x] `MicSource` stub (sounddevice)
- [x] `audio/resample.py` — 16 kHz mono normalisation (numpy + scipy, no librosa.to_mono)
- [x] `RollingBuffer` — 10s ring, 160k samples, `latest_seconds(n)` API

---

## Phase 3–4 — SNR Gate & Forensic Features
**Status: done**

- [x] `features/vad.py` — energy-based VAD (25ms frames, 10ms hop)
- [x] `features/snr.py` — dB estimate + `compute_gate_decision()` → GateState
- [x] `features/phase.py` — `compute_phase_discontinuity()` → float scalar
- [x] `features/pitch.py` — `compute_f0()` → (T,) ndarray + `compute_pitch_smoothness()` → float
- [x] `features/flux.py` — `compute_spectral_flux()` → float
- [x] `features/subband.py` — `compute_subband_energy()` → dict[str, float] (5 bands)
- [x] `features/mfcc.py` — `compute_mfcc_full()` → (39, T) ndarray (defined, not used in pipeline)
- [x] `features/spectrogram.py` — linear STFT, mel, CQT (defined, not used in pipeline)
- [x] `features/artifact.py` — `top_artifact_name(features)` → str | None

---

## Phase 5 — Classifier
**Status: done**

- [x] `classifier/protocol.py` — `RiskState` enum + `Scorer` Protocol
- [x] `classifier/state_mapper.py` — `score_to_state(score)` → RiskState
- [x] `classifier/fallback.py` — `FallbackScorer` with `score()` + `score_from_features()`
- [x] `classifier/aasist_model.py` — AASIST-L architecture stub
- [x] `classifier/aasist_loader.py` — `load_aasist()` with FileNotFoundError fallback
- [x] `classifier/aasist_inference.py` — `AASISTScorer` implementing Scorer
- [x] `classifier/__init__.py` — factory: `get_scorer()` (AASIST if weights present, else FallbackScorer)

---

## Phase 6–7 — Timeline, State Engine & Pipeline Runner
**Status: done**

- [x] `pipeline/timeline.py` — `TimelineEntry` dataclass + `Timeline` ring (20 entries)
- [x] `pipeline/state_engine.py` — hysteresis, GREY override, 0.85 short-circuit, latching
- [x] `pipeline/runner.py` — `PipelineRunner` + `_extract_features()` (single extraction)
- [x] Single feature extraction per chunk (avoids double numba JIT cost)

---

## Phase 8 — Streaming API
**Status: done**

- [x] `api/app.py` — FastAPI factory, CORS, JIT warmup at startup, background pipeline loop
- [x] `api/rest.py` — `POST /v1/analyze`, `GET /v1/risk/current`, `GET /v1/risk/timeline`
- [x] `api/ws.py` — `WebSocketManager`, `broadcast()`, `/v1/ws/risk` endpoint
- [x] `POST /v1/reset` — resets live pipeline state

---

## Phase 9 — React Interface (Vite + React 18)
**Status: done**

- [x] `VoiceShieldContext.jsx` — shared state via useReducer (file upload, WS stream, health check)
- [x] `NavBar.jsx` — server URL editor, health badge, mode indicator
- [x] `SourcePanel.jsx` — file picker + WS stream toggle
- [x] `RiskDisplay.jsx` — risk badge, advisory text, chunk metrics, sparkline, summary grid
- [x] `Sparkline.jsx` — canvas score history with threshold zones and state-coloured segments
- [x] `TimelineTable.jsx` — scrollable chunk log with filter
- [x] `StatusBar.jsx` — status footer

---

## Phase 10 — Performance Optimisations
**Status: done** *(2026-06-23)*

- [x] Replaced `librosa.to_mono` with `audio.mean(axis=0)` → eliminated 4.1s numba cold-start
- [x] Replaced `librosa.resample` with `scipy.signal.resample_poly` → numba-free
- [x] JIT warmup at FastAPI startup (`_warmup_jit()` in `api/app.py`)
- [x] Lazy warmup in `_run_analysis` for `/v1/analyze` (guards with `_jit_warmed` flag)
- [x] Eliminated double feature extraction per chunk via `score_from_features()`
- [x] **Result:** 5s audio file analyzes in ~917ms post-warmup (was 9+ seconds before)

---

## Phase 11 — Forensic Dashboard Enhancements
**Status: pending**

Six features specified. None yet implemented. Backend must ship before frontend.

### 11A — Backend: expose richer data per chunk
- [ ] `pipeline/runner.py` `_extract_features()`: return `f0_contour` array + full `subband` dict alongside scalar dict
- [ ] `pipeline/timeline.py`: add optional fields to `TimelineEntry` — `f0_contour: list[float] | None`, `subband_energies: dict | None`, `component_scores: dict | None`, `state_reason: str | None`
- [ ] `api/rest.py` `_run_analysis()`: compute per-feature normalized scores (component breakdown), collect per chunk
- [ ] `api/rest.py` `_run_analysis()`: compute provenance via `soundfile.info()` + `hashlib.sha256` before tempfile write
- [ ] Return provenance as top-level key in `/v1/analyze` JSON response
- [ ] `api/ws.py`: include new fields in WS broadcast

### 11B — Backend: "why this state" caption
- [ ] `pipeline/state_engine.py` `update()`: generate `state_reason` string from dominant feature + gate situation
- [ ] Expose via `TimelineEntry.state_reason`
- [ ] Examples: "Phase discontinuity (0.74) exceeded synthetic threshold" / "SNR below 8 dB — audio quality gate active"

### 11C — Frontend: context and state shape
- [ ] `VoiceShieldContext.jsx` initialState: add `provenance: null`
- [ ] `BATCH_RESULTS` reducer case: store `data.provenance` into `state.provenance`

### 11D — Frontend: "Why this state" caption
- [ ] `RiskDisplay.jsx`: add one sentence (grey italic) under the advisory action div
- [ ] Reads from `state.current?.state_reason`; hidden when null

### 11E — Frontend: provenance chip in NavBar
- [ ] `NavBar.jsx`: add expandable chip after the mode indicator
- [ ] Shows: codec · sample rate · duration · channels · peak amplitude · SHA-256 (first 8 chars)
- [ ] Only visible after successful file analysis (`state.provenance != null`)

### 11F — Frontend: collapsible "Forensic evidence" section
Create `Interface/src/components/forensic/`:
- [ ] `ComponentBreakdown.jsx` — stacked/grouped bar of 4 normalized sub-scores (phase / pitch / subband / flux)
- [ ] `PitchContour.jsx` — canvas line chart of F0 Hz over time; NaN frames shown as gaps
- [ ] `SubbandBars.jsx` — 5 mini horizontal energy bars (breath / fundamental / formants / consonants / artifact) with dB labels
- [ ] `ForensicEvidence.jsx` — collapsible wrapper, open by default, contains all three above
- [ ] Add `<ForensicEvidence />` below `<SummaryGrid />` in `RiskDisplay.jsx`
- [ ] Reads from `state.current` — `component_scores`, `f0_contour`, `subband_energies`

---

## Known Issues / Calibration Debt

- [ ] **FallbackScorer AMBER false positives on MP3:** `phase_discontinuity` weight (0.35) is too aggressive for MP3 inputs — codec artifacts trigger it. Genuine voices uploaded as MP3 score AMBER (observed: 0.32–0.41). Fix: raise SCORE_AMBER to 0.40, or detect codec in provenance and reduce phase weight.
- [ ] **AASIST weights absent:** `models/aasist_l.pth` not present. Pipeline always uses FallbackScorer.
- [ ] **Test suite status unknown:** `pytest -q` not run since last restructure. Verify before next commit.
- [ ] **E2E fixtures:** `tests/fixtures/*.wav` may not exist. E2E tests self-skip if absent.

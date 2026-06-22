# VoiceShield Stage 1 — Handoff Document

> **Who this is for:** Anyone picking up this repo for the first time, or returning after a break.  
> **What it covers:** What exists, why it was built that way, what works, what doesn't, and exactly where to start next.

---

## What is this project?

VoiceShield is a real-time AI voice fraud detector built for UCO Bank PSB Hackathon Problem Statement 2 ("Jigyasa"). It flags synthetic, cloned, or AI-generated voices during live bank calls — in under 10 seconds, on CPU, without recording the call.

The system is purely advisory: it produces a risk state (GREEN / AMBER / RED / GREY) and a recommended action for the bank agent. It never auto-blocks. This is by design — RBI compliance requires human-in-the-loop for any account action.

**Stage 1 only is implemented.** Stage 2 (ensemble models, replay detection, evidence export) is planned but out of scope for the hackathon.

---

## How to run it

```bash
# Install dependencies (Python 3.11+)
cd /home/beast/Documents/Hackathon/Jigyasa
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Start the API (file analysis mode — no microphone needed)
uvicorn voiceshield.api.app:create_app --factory --reload --port 8000

# Start the React dashboard
cd Interface
npm install
npm run dev
# open http://localhost:5173
```

Upload any WAV or MP3 file via the dashboard to see live analysis results.

**If the backend isn't running**, the dashboard shows "API Error" in the health badge. Verify the server URL in the NavBar matches where uvicorn is listening.

---

## What's been built (Phases 0–10 complete)

### Backend — `voiceshield/`

```
voiceshield/
├── config.py               # All constants (SR=16000, CHUNK_MS=500, SCORE_AMBER=0.30, SCORE_RED=0.70, ...)
├── logger.py               # Structured JSON logger (trace_id, run_id fields)
├── audio/
│   ├── source.py           # AudioSource ABC, FileSource (soundfile), MicSource (sounddevice)
│   ├── buffer.py           # RollingBuffer — 10s ring, 160k samples, latest_seconds(n)
│   └── resample.py         # numpy mono downmix + scipy.signal.resample_poly (no numba)
├── features/
│   ├── vad.py              # Energy-based VAD (25ms frames, 10ms hop) → bool array
│   ├── snr.py              # SNR estimate (dB) + compute_gate_decision() → GateState
│   ├── phase.py            # compute_phase_discontinuity() → float [0,1]
│   ├── pitch.py            # compute_f0() → (T,) ndarray in Hz (NaN=unvoiced) + compute_pitch_smoothness() → float
│   ├── flux.py             # compute_spectral_flux() → float (mean onset strength)
│   ├── subband.py          # compute_subband_energy() → dict with 5 dB floats (breath/fundamental/formants/consonants/artifact)
│   ├── mfcc.py             # compute_mfcc_full() → (39, T) ndarray — defined but NOT used in pipeline
│   ├── spectrogram.py      # linear STFT, mel, CQT — defined but NOT used in pipeline
│   └── artifact.py         # top_artifact_name(features) → str | None (highest z-score feature name)
├── classifier/
│   ├── protocol.py         # RiskState enum (GREEN/AMBER/RED/GREY) + Scorer Protocol
│   ├── state_mapper.py     # score_to_state(float) → RiskState (boundaries: 0.30=AMBER, 0.70=RED)
│   ├── fallback.py         # FallbackScorer — rule-based, score() + score_from_features()
│   ├── aasist_model.py     # AASIST-L architecture (stub, for loading weights)
│   ├── aasist_loader.py    # load_aasist(path) — raises FileNotFoundError if weights absent
│   ├── aasist_inference.py # AASISTScorer implementing Scorer Protocol
│   └── __init__.py         # get_scorer() factory — tries AASIST, falls back to FallbackScorer
├── pipeline/
│   ├── timeline.py         # TimelineEntry dataclass + Timeline (20-entry ring buffer)
│   ├── state_engine.py     # StateEngine — hysteresis, GREY override, 0.85 short-circuit, latching
│   └── runner.py           # PipelineRunner — ties everything together; run_once(), run_forever()
└── api/
    ├── app.py              # FastAPI factory, CORS, JIT warmup at startup, background WS loop
    ├── rest.py             # POST /v1/analyze, GET /v1/risk/current, GET /v1/risk/timeline, POST /v1/reset
    └── ws.py               # WebSocketManager, broadcast(), /v1/ws/risk
```

### Frontend — `Interface/`

```
Interface/src/
├── context/
│   └── VoiceShieldContext.jsx  # useReducer state (entries, current, summary, mode, wsStatus, ...)
├── components/
│   ├── NavBar.jsx              # Server URL editor, health badge, mode indicator (FILE / LIVE)
│   ├── SourcePanel.jsx         # File picker + WS stream toggle + export buttons
│   ├── RiskDisplay.jsx         # Risk badge, advisory text, chunk metrics, sparkline, summary grid
│   ├── Sparkline.jsx           # Canvas score history with threshold zones + state-coloured polyline
│   ├── TimelineTable.jsx       # Scrollable chunk log with filter by state
│   └── StatusBar.jsx           # Footer status bar
├── App.jsx                     # 3-column grid layout (SourcePanel | RiskDisplay | TimelineTable)
└── index.css                   # Cream paper design system (CSS vars: --color-paper-white, --color-ink-black, ...)
```

### API contract (current)

**POST /v1/analyze** — upload audio file, returns full analysis:
```json
{
  "entries": [
    {"time": 0.5, "score": 0.24, "state": "green", "snr_db": 18.2, "top_artifact": null, "first_amber_t": null, "first_red_t": null},
    ...
  ],
  "summary": {
    "total_chunks": 12, "duration_s": 6.0, "mean_score": 0.37,
    "final_state": "amber", "first_amber_t": 2.0, "first_red_t": null
  }
}
```

**WS /v1/ws/risk** — streams one `TimelineEntry` dict per 500ms chunk (same shape as each entry above).

---

## Key design decisions to preserve

### 1. Single feature extraction per chunk
`pipeline/runner.py:_extract_features()` is called once per chunk. The result dict is passed to both `FallbackScorer.score_from_features()` AND `top_artifact_name()`. Never call feature extraction twice — librosa functions have numba JIT overhead.

### 2. JIT warmup at startup
`api/app.py:_warmup_jit()` runs `_extract_features(zeros)` in a thread executor during FastAPI startup. This triggers numba JIT compilation before the first request arrives, preventing a 3–4s penalty on the first `/v1/analyze` call.

### 3. scipy for resampling, not librosa
`audio/resample.py` uses `scipy.signal.resample_poly` for non-16kHz input. `librosa.resample` uses numba internally — never use it here.

### 4. State engine is monotonic
The `StateEngine` never de-escalates: once AMBER is reached, it stays AMBER or higher. `first_amber_t` and `first_red_t` latch once and are never overwritten. This is intentional — the dashboard shows the *first* time risk appeared, not the current worst chunk.

### 5. GREY beats everything
If SNR < 8 dB, the state engine returns GREY regardless of the classifier score. This prevents low-quality audio from generating false RED alerts. The score is still recorded in the timeline for debugging.

### 6. Scorer is a Protocol (not a base class)
`classifier/protocol.py:Scorer` is a structural Protocol. Stage 2 models plug in without inheriting anything. The runner only checks `isinstance(scorer, FallbackScorer)` to decide whether to use `score_from_features()` — everything else is duck-typed.

---

## Known issues and calibration debt

### FallbackScorer AMBER false positives on MP3 input
**Observed:** Genuine voice uploaded as MP3, 6s file, scored AMBER (0.32–0.41 range).  
**Root cause:** MP3 compression creates phase artifacts that trigger `compute_phase_discontinuity()`. The FallbackScorer weights phase at 0.35 — the highest weight. WAV files don't have this problem.  
**Fix options (not yet applied):**
1. Detect codec from `soundfile.info()` and reduce phase weight for lossy formats
2. Raise `SCORE_AMBER` from 0.30 to 0.40 for FallbackScorer specifically
3. Compute phase discontinuity differently for lossy input (e.g., ignore high-frequency bins where MP3 artifacts concentrate)

### AASIST weights not present
`models/aasist_l.pth` is not in the repo (large binary, research-licensed). The pipeline always uses `FallbackScorer`. This is expected for a hackathon demo. Run `python scripts/download_model.py` if available.

### Test suite not verified recently
`pytest -q` has not been run since the performance optimisation changes. Run it before the next commit to verify nothing broke.

---

## What's next (Phase 11 — Forensic Dashboard)

Six features specified by the user, none yet implemented:

1. **"Why this state" caption** — one sentence under the risk badge explaining which feature triggered escalation. Requires `state_reason: str | None` field added to `TimelineEntry` and generated in `StateEngine.update()`.

2. **Audio provenance chip in NavBar** — codec, sample rate, duration, channels, peak amplitude, SHA-256. Requires `provenance` dict returned as top-level key in `/v1/analyze` response.

3. **Component score breakdown** (stacked bar) — normalized per-feature scores (phase / pitch / subband / flux). Requires `component_scores: dict` field in `TimelineEntry`.

4. **Pitch F0 contour** — canvas line chart of Hz over time. `compute_f0()` already returns the full array — just needs to be serialized (downsample to ~20 points per chunk before sending in JSON).

5. **Subband energy mini-bars** — 5 horizontal bars for breath/fundamental/formants/consonants/artifact in dB. `compute_subband_energy()` already returns this dict — just needs serialization.

6. **Phase heatmap** — the original spec included a freq×time waterfall canvas, but note: `compute_phase_discontinuity()` currently returns only a scalar, not the full matrix. To render a heatmap you'd need to extract the full unwrapped phase array from the STFT — modify `phase.py` to optionally return the full `(n_freq, n_frames)` matrix.

**Start here for Phase 11:**
- `pipeline/runner.py` → `_extract_features()`: extend to return `f0_contour` array and `subband` dict alongside the existing scalar dict
- `pipeline/timeline.py` → `TimelineEntry`: add optional fields (`f0_contour`, `subband_energies`, `component_scores`, `state_reason`)
- `api/rest.py` → `_run_analysis()`: populate new fields, add provenance, return as part of response
- Then: frontend components in `Interface/src/components/forensic/`

Full implementation plan is in `tasks/todo.md` Phase 11.

---

## Verification checklist

Before shipping any change:

```bash
# lint
uv run ruff check voiceshield --fix
uv run ruff format voiceshield

# type check
uv run mypy voiceshield --ignore-missing-imports

# tests
uv run pytest -q

# smoke: start server + curl a WAV
uvicorn voiceshield.api.app:create_app --factory --port 8000 &
curl -s -F "file=@tests/fixtures/genuine_male_16k.wav" http://localhost:8000/v1/analyze | python -m json.tool
```

---

## File map: where to look for what

| If you want to...                  | Look here |
|------------------------------------|-----------|
| Change risk thresholds             | `voiceshield/config.py` (SCORE_AMBER, SCORE_RED, SNR_NORMAL_DB, SNR_GREY_DB) |
| Change FallbackScorer weights      | `voiceshield/classifier/fallback.py` (_WEIGHTS, _BOUNDS) |
| Change hysteresis behaviour        | `voiceshield/pipeline/state_engine.py` (_HYSTERESIS_CHUNKS, _RED_OVERRIDE_THRESHOLD) |
| Add a new forensic feature         | `voiceshield/features/` + wire into `pipeline/runner.py:_extract_features()` |
| Add a new API endpoint             | `voiceshield/api/rest.py` |
| Add a new React component          | `Interface/src/components/` |
| Change the dashboard layout        | `Interface/src/App.jsx` (3-column grid) |
| Change shared frontend state       | `Interface/src/context/VoiceShieldContext.jsx` |
| Understand the architecture        | `docs/Stage_1.md`, `docs/ARCHITECTURE.md` |
| See what was planned               | `PLANS.md` |
| See accumulated lessons            | `tasks/lessons.md` |

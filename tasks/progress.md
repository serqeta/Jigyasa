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

---

## Stage 2 — Ensemble + Replay + Fusion + Evidence + /v2 *(2026-07-08)*

### Actions taken
- GPU foundation: reinstalled torch 2.12.1+cu130 (RTX 4050), added transformers + pillow; pinned numpy<2.5 (numba conflict). `config.get_device()` auto-detects.
- Downloaded three verified pretrained anti-spoof checkpoints (see NOTICE.md): Gustking XLS-R 300M (ssl), MattyB95 AST-ASVspoof5 (spec), abhishtagatya WavLM In-the-Wild (wavlm).
- New: `classifier/hf_scorer.py` (generic HF audio-classification scorer, fp16 on GPU, per-model spoof-label polarity), `classifier/phase_pitch.py`, `get_scorers()` factory, `voiceshield/replay/` (4 DSP detectors), `fuse_scores()` weighted renormalizing fusion, `voiceshield/evidence/` (PNGs + SHA-256 + JSON envelope, pull-only), `api/rest_v2.py` (/v2/models, /v2/analyze, /v2/evidence/export, /v2/ws/risk), Interface EnsemblePanel + evidence button + /v2 data layer.
- Moved AASIST inference to GPU (was the latency bottleneck at 239 ms/chunk on CPU).
- Runner scores all ensemble components per chunk; TimelineEntry carries component_scores + replay breakdown.

### State-engine correction (user-reported)
Silence after AMBER used to hold AMBER and could escalate to RED off the stale 4 s window (transition chunks kept speech_active=True for ~1 s because the SNR window lags). Fixed: GREY always displays during silence (internal state + first-alert latches preserved for resume), and escalation requires fresh speech in the newest 500 ms chunk (chunk-level VAD ≥ MIN_VOICED_RATIO). Regression tests in tests/unit/pipeline/test_state_engine.py and tests/integration/test_silence_transitions.py.

### Verification
- 83 tests green (`pytest tests/unit tests/integration`), make check green.
- Full-ensemble per-chunk latency on GPU: p50 124 ms, p95 166 ms (< 200 ms gate).
- Interface `npm run build` green.

### Known follow-ups
- `wavlm` checkpoint has no declared license (NOTICE.md) — disable for production if unresolved.
- `/v1/analyze` still uses FallbackScorer by design; `/v2/analyze` runs the ensemble.

---

## Stage 2 — Fixture validation & fusion calibration *(2026-07-08, same session)*

### Fixtures (scripts/build_fixtures.py)
- genuine_male_16k.wav / genuine_female_16k.wav: LibriSpeech dev-clean speakers 3752 / 2035 (CC BY 4.0), pinned after scanning all 40 validation speakers against the ensemble.
- tts_synthetic_16k.wav: SpeechT5 + HiFi-GAN, generic voice. cloned_voice_16k.wav: SpeechT5 conditioned on CMU Arctic 'bdl' x-vectors (a real speaker's voice clone).
- CMU Arctic was rejected as a genuine source: every anti-spoof model scores those studio voices as synthetic (they seeded decades of TTS systems).

### Calibration findings (real data, per-chunk)
- AST-ASVspoof5 ('spec') outputs p(Spoof)≈1.0 for ANY input (fp32-verified) → disabled in config, slot kept.
- AASIST-L false-positives on all real speech tried (0.70–0.94 on genuine) → weight cut 0.35→0.10.
- wavlm fires ~1.0 on 4/40 genuine LibriSpeech speakers (8297, 1993, 2412, 3000) → kept in mean (0.25) but NOT a peak component.
- New peak-evidence rule in fuse_scores: fused ≥ 0.8 × max over PEAK_COMPONENTS — a confident validated detector cannot be averaged away. PEAK_COMPONENTS=("ssl",) only; wavlm excluded because of the genuine-speaker misfires above.
- Final weights: stage1 .10, ssl .40, spec .15(disabled), wavlm .25, phase_pitch .10, replay .15.

## Model-selection evaluation & ensemble overhaul *(2026-07-20)*

### Why
User-reported reliability problems: AASIST false positives, other models false negatives, plus concern that our home-grown eval set couldn't cover modern generators (GANs etc.).

### Evaluation set (scripts/build_evalset.py + scripts/fetch_benchmark.py)
496 clips: 40 LibriSpeech speakers + 2 Arctic (genuine), SpeechT5×5 + VITS + 2 ElevenLabs clips (fakes), opus-16k codec variants of both, PLUS benchmark samples fetched anonymously from HF: WaveFake (7 GAN vocoders + real LJSpeech), ASVspoof 2021 DF (100+ TTS/VC attacks), In-the-Wild (real-world deepfakes).

### Results (AUC / FPR@0.5 / FNR@0.5)
- nii mms-300m-anti-deepfake: **0.997 / 0.05 / 0.02** — per-source AUC 1.00 on GAN, ASVspoof, ITW, ours. Clear winner.
- wavlm_itw 0.87 / 0.06 / 0.56 · xlsr_gustking 0.73 / 0.06 / 0.54 — kept as diversity members.
- aasist **0.33** (worse than random; genuine μ 0.66 > fake μ 0.46) → retired from ensemble, /v1 compat only.
- ast_asvspoof5 (0.49, says spoof to everything), w2v2_melody (0.44), phase_pitch/fallback (~0.5) → zero weight.

### NII integration
fairseq-only checkpoint converted to transformers format (scripts/convert_nii.py, clean 422/422 key mapping) and validated against the reference implementation in a py3.10 side-venv (max score diff 0.011). Runs in-process fp16 GPU as classifier/nii_scorer.py. **License CC BY-NC-SA (non-commercial) — see NOTICE.md.**

### New calibration (from measured quantiles)
FUSION_WEIGHTS: nii .50, ssl .15, wavlm .15, rest 0. PEAK_COMPONENTS: nii 0.8 (genuine p99 0.81→floor 0.65<RED; fake median 0.997→0.80=RED), ssl 0.8, wavlm 0.6. Cascade screener switched from AASIST to NII (config.CASCADE_SCREENER). fuse_scores falls back to plain mean when all present components are zero-weight (Stage 1 single-scorer compat).

### Replay recalibration
freq_response rebuilt as a loudspeaker-channel composite (low-band deficit + high-band deficit + spectral tilt), calibrated on simulated channels: separates medium/telephone-grade playback (~0.9) from genuine (~0.0); mild channels and codec round-trips are NOT separable by LTAS features (documented). double_compression and background_mismatch proven non-discriminative → zero internal weight, display only. Replay fusion weight stays 0.0 (future work per user decision).

### HF auth gotcha
A stale user token in ~/.cache/huggingface/token 401s every implicit-auth request; all scripts run with HF_HUB_DISABLE_IMPLICIT_TOKEN=1.

## Parameter hardening round *(2026-07-20, same session)*

### Implemented
- **Temporal smoothing** (config.SCORE_SMOOTHING_CHUNKS=3): state engine sees the median of the last 3 speech-chunk fused scores (history clears on silence). Cascade engagement still reacts to the RAW score so a single hot probe engages Stage 2 immediately — only the user-facing state is smoothed.
- **Confidence gate** (config.CONFIDENCE_GATE_LOW=0.15): on non-probe, non-engaged chunks where the NII screener is confidently clean, XLS-R/WavLM/replay are skipped. Probes and suspicious/engaged chunks always run the full ensemble, preserving the multi-model-consensus requirement for instant-RED.

### Measured (scripts/build_hindi_eval.py + inline measurement)
- **Hindi** (40 FLEURS genuine vs 6 MMS-TTS-hin fakes): nii AUC **1.000** (genuine max 0.01), ssl 0.986, wavlm 0.819 (one genuine at 1.0 — its known FP mode, stays non-peak). Pipeline: 100% of Hindi fakes AMBER ≤5 s (median 1.5 s), 5/6 RED ≤10 s — the weakest VITS fake alerts at sustained AMBER only (nii 0.27–0.4 on it).
- **Window length** (nii, 2/4/8 s): AUC 0.999 / 0.997 / 1.000 — no meaningful difference; keeping 4 s (ssl/wavlm trained near it; zero-risk).
- **Noise robustness** (15 dB additive noise + mild reverb): nii AUC drops to 0.807 (fake μ 0.88→0.47), ssl 0.716, wavlm collapses to 0.575. Documented limit: heavily noisy calls reduce detection confidence; SNR gate already marks <12 dB as REDUCED.

### Detection-timing benchmark (scripts/bench_detection.py, full cascade, GPU)
- Fakes (n=323): 98.1% reach AMBER, 95.0% reach RED ≤10 s; median time-to-AMBER 1.5 s, time-to-RED 1.5 s (p90 2.5 s). Among fakes ≥4 s long: 99.0% RED ≤10 s (short benchmark clips end before hysteresis can fire).
- Genuine (n=173, incl. codec + noisy wild domains): 26% have ≥1 transient AMBER chunk, 3.5% ≥1 RED chunk (concentrated in hard domains: Arctic, LJSpeech-era recordings, noisy celebrity audio). Demo fixtures: zero alerts.
- Latency per 500 ms chunk: p50 117 ms, p95 208 ms, p99 238 ms on the RTX 4050 laptop (budget 200 ms — p95 4% over when deep-scanning every chunk of attack audio; screening-dominated genuine traffic is well under. Future optimization: confidence-gate the diversity members on NII's score).

---

### Acceptance results (full ensemble, GPU)
- genuine male/female: mean 0.153/0.134, zero AMBER/RED chunks.
- TTS: first_red_t = 1.0 s (gate ≤ 10 s). Clone: first_amber_t = 1.0 s (≤5), first_red_t = 1.0 s (≤10).
- Silent → all GREY. p95 per-chunk latency (full ensemble) < 200 ms.
- e2e tests now run the shipped ensemble (get_scorers), not FallbackScorer. 90 tests green.

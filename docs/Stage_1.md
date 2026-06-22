# VoiceShield — Stage 1 Implementation Documentation

> **Companion to `planning.txt`.** This document explains *why* Stage 1 is scoped
> the way it is, how each component should be built, and how every acceptance
> criterion maps back to the source architecture and problem statement.

**Source documents this is derived from**
- `VoiceShield_Updated_Architecture_2_.md` — the full system architecture (referenced as **A§N**).
- `VoiceShield_Jigyasa.txt` — UCO PSB Hackathon Problem Statement 2 deck (referenced as **J p.N**).

---

## 1. Why "Stage 1 only" is the right slice

VoiceShield as a whole is a three-layer system (A§4):

```
Audio in
  → capture / chunking / buffer / SNR gate
  → spectrogram + forensic features
  → Stage 1 Fast Synthetic Voice Detector     ← we stop here
  → Stage 2 Ensemble + Replay Detection
  → Risk fusion + evidence export
```

Stage 1 is **the smallest vertical slice that delivers the headline promise of
the project**: *flag a synthetic / cloned voice as High Risk within the first
10 seconds of a live call* (A§2, J p.4). The rest of the system (ensemble,
replay, evidence export) raises confidence and explainability, but it is not
required to demonstrate the core claim.

So the implementation goal is a working pipeline that, end-to-end:

1. Takes a live or file audio stream.
2. Produces a single 0–1 synthetic-risk score per 500 ms chunk.
3. Maps that score (plus the SNR gate) into Green / Amber / Red / Grey.
4. Streams it to an advisory dashboard.

Everything else is **deliberately deferred**. The repository layout and the
`Scorer` protocol in `planning.txt` §10 are designed so Stage 2 plugs in
without rewriting Stage 1.

> **The narrower question Stage 1 must answer (A§3):**
> *"Does this live call contain acoustic evidence that the voice may be
> synthetic, replayed, or manipulated?"*

Stage 1 does the *synthetic* half of that question. Replay detection is
explicitly Stage 2.

---

## 2. End-to-end data flow (Stage 1 only)

```
              ┌──────────────────────────┐
              │  AudioSource             │   16 kHz mono float32
              │  (MicSource / FileSource)│   500 ms chunks
              └────────────┬─────────────┘
                           │ 8000 samples / chunk
                           ▼
              ┌──────────────────────────┐
              │  RollingBuffer           │   10 s ring (160 000 samples)
              └────────────┬─────────────┘
                           │ latest_seconds(N)
                ┌──────────┼──────────┐
                ▼                     ▼
         ┌──────────────┐     ┌──────────────────┐
         │ SNR Gate     │     │ Feature layer    │
         │ (VAD + dB)   │     │  - linear / mel  │
         │ NORMAL /     │     │  - CQT           │
         │ REDUCED /    │     │  - phase heatmap │
         │ GREY         │     │  - F0 + smooth   │
         └──────┬───────┘     │  - spectral flux │
                │             │  - MFCC Δ ΔΔ     │
                │             │  - subband energy│
                │             └─────────┬────────┘
                │                       │
                │                       ▼
                │             ┌──────────────────┐
                │             │ Stage 1 classifier│  AASIST-L
                │             │  score in [0, 1] │  (fallback: rules)
                │             └─────────┬────────┘
                │                       │
                ▼                       ▼
              ┌──────────────────────────┐
              │  State engine            │  Green / Amber / Red
              │  + hysteresis            │  Grey if SNR < 8 dB
              │  + first-alert tracker   │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  Artifact timeline       │  last 20 entries
              │  (one per 500 ms chunk)  │  { t, score, state, snr, … }
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  FastAPI                 │  WS /ws/risk
              │  + minimal HTML dashboard│  REST /risk/{current,timeline}
              └──────────────────────────┘
```

This is exactly the path from A§4 minus the boxes after "Stage 1 Fast Synthetic
Voice Detector".

---

## 3. Component-by-component design

The numbering matches the phases in `planning.txt` §6.

### 3.1 Audio capture (Phase 1)

**Why:** the system must work on a live stream and must not require a
pre-recorded call (A§5, J p.6 step 1).

**Design:**
- Single `AudioSource` ABC with one method: `read_chunk() -> np.ndarray` of
  shape `(CHUNK_SAMPLES,)`, dtype `float32`, mono, range `[-1, 1]`.
- Two concrete implementations:
  - `MicSource` for live demo (sounddevice + InputStream).
  - `FileSource` for deterministic tests and the hackathon demo
    ("laptop A plays synthetic voice, laptop B captures" — A§16).
- All audio is normalised to **16 kHz mono** before entering the pipeline.
  AASIST is trained on 16 kHz speech, so the rest of the stack standardises
  on that rate. Resampling lives in `audio/resample.py`.

**Why 500 ms chunks specifically?** A§5 and J p.6 fix this: it is the
granularity at which the system "recomputes the latest risk score from the
rolling buffer." With a 16 kHz sample rate that is exactly 8000 samples per
chunk.

### 3.2 Rolling buffer (Phase 2)

**Why:** "Old audio is discarded unless the call is flagged. This keeps the
design privacy-friendly." (A§5)

**Design:**
- A numpy ring buffer of fixed capacity `BUFFER_SECONDS * SAMPLE_RATE = 160 000`.
- Append-only API plus `latest_seconds(n)` for downstream feature windows.
- Each pushed chunk carries `(chunk_index, t_start, t_end)` so the timeline
  later has a consistent time axis.

This is the only memory the pipeline holds. There is no on-disk recording in
Stage 1 (acceptance criterion G10).

### 3.3 SNR Quality Gate (Phase 3)

**Why:** "Poor audio can create false artifacts. Background noise, packet loss,
microphone distortion, or poor network conditions may look suspicious even when
the caller is genuine." (A§6)

**Design — exactly matches A§6 + J p.6:**

| SNR (dB)        | Gate output | Pipeline effect                                  |
| --------------- | ----------- | ------------------------------------------------ |
| ≥ 12            | NORMAL      | Classifier score used as-is                      |
| 8 ≤ snr < 12    | REDUCED     | Classifier score used, but flagged as low-conf   |
| < 8             | GREY        | **Final state forced to Grey** regardless of score |

Two-step implementation:
1. **VAD** (`features/vad.py`) — energy-based, 25 ms frames, 10 ms hop. Decides
   which frames are speech vs silence.
2. **SNR estimator** (`features/snr.py`) — dB ratio between mean energy of
   speech frames and mean energy of silence frames over the latest 1 s window.

The override "GREY beats everything" is the single most important rule in the
state engine (T7.1, TEST-R7.1). It is what prevents the system from screaming
RED at a customer whose phone connection just dropped.

### 3.4 Forensic feature layer (Phase 4)

A§7 is the canonical list of features. Every feature in the architecture's
"Stage 1 inputs" section appears in Phase 4:

| Feature                       | Source ref | Why it matters                                                   |
| ----------------------------- | ---------- | ---------------------------------------------------------------- |
| Linear STFT spectrogram       | A§7.1      | High-freq artifacts, vocoder noise, codec distortions            |
| Mel spectrogram               | A§7.2      | Perceptual reference (closer to what an agent hears)             |
| CQT spectrogram               | A§7.3      | Harmonic drift, merged / unstable harmonics                      |
| Phase discontinuity heatmap   | A§7.4      | Neural-vocoder frame-boundary resets                             |
| Pitch contour (F0)            | A§7.5      | Tracks fundamental over time                                     |
| Pitch smoothness              | A§7.5      | Synthetic speech is unnaturally smooth                           |
| Spectral flux                 | A§7.6      | Consonant transitions, over-smoothed generated speech            |
| MFCC + Δ + ΔΔ                 | A§7.7      | Dynamic speech behaviour AASIST-style models consume             |
| Subband energy (5 bands)      | A§7.8      | Vocoder/compression clues in 6–8 kHz "artifact band"             |

**Note on responsibility.** The classifier (Phase 5) is what produces the
risk score. The feature panels exist to:

1. Feed the **fallback** rule-based classifier (T5.5) — so the pipeline can
   demo even if AASIST weights cannot be downloaded.
2. Drive the `top_artifact` field on each `TimelineEntry`, which is what makes
   the alerts *explainable* — the headline VoiceShield differentiator
   (J p.5 "Named artifacts", A§15).
3. Be available as evidence in Stage 2 without refactoring.

### 3.5 Stage 1 classifier (Phase 5)

**Choice: AASIST / AASIST-L.** The architecture explicitly recommends it
(A§8): "AASIST or AASIST-L style anti-spoofing classifier, CPU-friendly
deployment, runs on short rolling windows, outputs a synthetic-risk score
from 0.0 to 1.0."

**Why this model specifically:**
- It is the canonical baseline in the ASVspoof anti-spoofing literature.
- Open-source PyTorch implementation and pretrained weights exist
  (`clovaai/aasist` on GitHub).
- AASIST-L (the "lite" variant) is small enough to run on CPU within the
  < 100 ms per-call budget (G4).
- It takes raw waveform input — no hand-engineered preprocessing pipeline
  needed beyond resample + pad/trim.

**Implementation:**
- `classifier/aasist_loader.py` — loads `models/aasist_l.pth`.
- `classifier/aasist_inference.py` — exposes a `Scorer` protocol:
  `score(audio: np.ndarray) -> float in [0, 1]`.
- A typical inference call takes the last **4 seconds** of audio (the model's
  trained window), pads / trims to the expected sample count, runs forward,
  and returns the spoof probability via softmax. Within the first 4 s of a
  call the buffer is zero-padded to length, which is consistent with the
  "0–2 s: collect audio, low confidence" rule in A§5.

**Fallback (T5.5).** AASIST weights are research-licensed. If for any reason
the team cannot ship them in the hackathon demo, the fallback must still
produce a usable score. The fallback computes:

```
score = w1 * normalised(mean phase discontinuity)
      + w2 * normalised(1 - pitch variance)             # too smooth → high
      + w3 * normalised(high-band / low-band energy)    # vocoder artifact band
      + w4 * normalised(1 - spectral flux variance)     # over-smoothed
```

Weights start at `w1=0.35, w2=0.25, w3=0.20, w4=0.20` and can be tuned on the
fixture set. The fallback **must** score `tts_synthetic_16k.wav` strictly
higher than `genuine_male_16k.wav` (TEST-C5.6) before it is acceptable.

### 3.6 Score → state mapping (Phase 5 cont.)

From A§8 and J p.9:

| Score        | State | Agent action (advisory, A§15)                                |
| ------------ | ----- | ------------------------------------------------------------ |
| 0.00 – 0.30  | Green | Proceed normally with standard verification                  |
| 0.30 – 0.70  | Amber | Send OTP to registered mobile; preserve buffer; deeper check |
| 0.70 – 1.00  | Red   | Transfer to supervisor; do not process the transaction       |
| (any)        | Grey  | Audio quality insufficient; manual review post-call          |

The boundary at exactly 0.30 is **Amber**, and exactly 0.70 is **Red**
(TEST-C5.5). Pin this in the unit test or it will silently drift.

### 3.7 Risk state engine (Phase 7)

Three rules, in priority order:

1. **GREY override.** If the SNR gate is GREY, final state is GREY. The
   classifier score is ignored for state purposes (but is still emitted to
   the timeline for debugging).
2. **Hysteresis.** Going from Green → Amber requires two consecutive 500 ms
   chunks above 0.30. Going from Amber → Red requires two consecutive chunks
   above 0.70 — **except** a single chunk above 0.85 short-circuits straight
   to Red (so a clean, obvious deepfake doesn't get held up by hysteresis;
   see Risk R6 in planning.txt).
3. **Latching first-alert times.** `first_amber_t` and `first_red_t` are set
   once and never overwritten. They are what feeds the headline "trigger time"
   on the dashboard (A§15 example).

### 3.8 Artifact timeline (Phase 6)

Exactly mirrors A§12:

```json
[
  {"time": 0.5, "score": 0.18, "state": "green", "snr_db": 18.1, "top_artifact": null},
  {"time": 1.0, "score": 0.22, "state": "green", "snr_db": 17.9, "top_artifact": null},
  {"time": 4.5, "score": 0.56, "state": "amber", "snr_db": 17.4, "top_artifact": "phase_discontinuity"},
  {"time": 7.5, "score": 0.82, "state": "red",   "snr_db": 17.8, "top_artifact": "over_smooth_pitch_contour"}
]
```

The `top_artifact` field is the single highest-z-score feature this chunk —
this is what makes alerts *explainable* and is the bridge to the named-artifact
evidence the dashboard shows (J p.5, A§13). Stage 1 only needs the *name* —
Stage 2 will add the visual evidence (spectrogram PNG, phase heatmap PNG).

The timeline is a ring of 20 entries (10 s ÷ 500 ms) — older entries fall off.

### 3.9 Streaming API (Phase 8)

FastAPI + WebSocket. Three endpoints:

| Endpoint              | Method | Returns                                |
| --------------------- | ------ | -------------------------------------- |
| `/ws/risk`            | WS     | Live `TimelineEntry` per chunk         |
| `/risk/current`       | GET    | The latest `TimelineEntry`             |
| `/risk/timeline`      | GET    | The full 10 s rolling timeline (JSON)  |

API surface is versioned at `/v1/`. Stage 2 adds `/v2/` without breaking
Stage 1 clients (planning.txt §10).

### 3.10 Advisory dashboard (Phase 9)

Plain HTML + vanilla JS — **no React in Stage 1**. This is intentional:

- The hackathon demo is a single page that reflects current state.
- A 50-line file is auditable in seconds; a React/Vite build is not.
- It matches A§15: "Current risk colour, risk score, trigger time, short
  explanation, suggested action." Nothing more.

Visual contract:

```
┌────────────────────────────────────────────────────────────┐
│  VoiceShield — Call CALL-2026-001                          │
│                                                            │
│   ┌──────────┐    score: 0.82                              │
│   │   RED    │    trigger:  7.5 s                          │
│   └──────────┘    artifact: over_smooth_pitch_contour      │
│                                                            │
│   Recommended action:                                      │
│   "Do not process sensitive request; transfer to           │
│    supervisor."                                            │
│                                                            │
│   ┌─ Risk over last 10 s ────────────────────────────┐    │
│   │  ▁ ▁ ▂ ▂ ▃ ▅ ▆ ▇ █ █                              │    │
│   │  0s         5s                              10s    │    │
│   └─────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Test strategy

The acceptance bar is three layers deep:

**Unit tests** (Phase 0–9 per-module). Fast, run on every commit. Validate
shapes, dtypes, value ranges, boundary conditions. These catch "I changed the
buffer length and nothing else broke locally" silent regressions.

**Integration tests.** Wire FileSource through the whole pipeline to the
timeline. Catch interface mismatches (e.g. SNR estimator returning dB but
state engine expecting linear ratio).

**End-to-end acceptance tests** (`TEST-E2E10.1` … `10.5`). These are the
gate for Stage 1 being "done". They are the *direct, measurable form* of the
project promises:

| Promise from source docs                                              | E2E test         |
| --------------------------------------------------------------------- | ---------------- |
| "Flag suspicious calls as High Risk within the first 10 seconds" (A§2) | TEST-E2E10.2, 10.4 |
| "Does not auto-block" — Green stays Green on a genuine voice (A§15)    | TEST-E2E10.1     |
| "SNR < 8 dB → Grey / Monitor state" (A§6)                              | TEST-E2E10.3     |
| Stage 1 is "CPU-deployable, first alert in 5–10 s" (J p.9)             | TEST-E2E10.5     |

`TEST-E2E10.4` is the most important one. It uses the cloned voice fixture
and is the closest analog to a real attack — and it asserts both
`first_amber_t ≤ 5.0 s` *and* `first_red_t ≤ 10.0 s`, the exact escalation
curve specified in A§5 and visualised in J p.11.

**Fixtures.** A small `tests/fixtures/` directory ships with the repo
(see planning.txt §7). Get these recorded / generated early — they are the
single biggest blocker on running E2E tests. The `cloned_voice_16k.wav`
in particular must come from a known, reproducibly-detectable TTS source
(e.g. tortoise-tts, Coqui XTTS, or any consumer cloning service) and be
checked in once with a frozen hash; it must not be regenerated per run.

---

## 5. How to run (target developer experience)

```bash
# one-time
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py            # fetches models/aasist_l.pth

# live demo
python scripts/run_live.py
# open http://localhost:8000

# deterministic file demo (hackathon)
python scripts/run_file.py tests/fixtures/cloned_voice_16k.wav
# dashboard turns Amber by ~5 s, Red by ~8–10 s

# tests
pytest -q                                   # unit + integration
pytest tests/e2e -q                         # full E2E
```

If `download_model.py` fails or the user has no network access, set
`USE_FALLBACK_CLASSIFIER=True` in `voiceshield/config.py` and the pipeline
runs end-to-end on the rule-based scorer.

---

## 6. Privacy and compliance notes

These are baked into Stage 1 by design, not retrofitted:

- **No persistent recording.** The only audio storage is the 10 s rolling
  in-memory buffer (G10). Nothing is written to disk by default.
- **No auto-blocking.** The system never tells the agent to refuse a call.
  It produces an advisory state and a recommended action; the agent decides
  (A§15, J p.17). This is the RBI compliance requirement.
- **No voiceprints stored.** Stage 1 doesn't enroll anyone. It does not ask
  "who is calling"; it asks "is this audio physically produced by a human
  vocal tract?" (J p.4). There is no biometric database to leak.
- **Audio hash for evidence is Stage 2.** Stage 1 only logs the risk state
  and timeline.

---

## 7. Known limitations of Stage 1 (honest list)

1. **Single classifier.** Stage 1 is one model. A top-tier TTS that AASIST
   was not trained on may slip through. Mitigation: Stage 2 ensemble.
2. **No replay detection.** A fraudster who plays cloned audio through a
   speaker into a phone mic may evade Stage 1 if the underlying synthesis is
   high-fidelity. Replay detection is Phase-of-Stage-2 (A§11).
3. **No SIP/VoIP integration.** Stage 1 captures from the local mic only.
   Real-call ingestion is deployment work, not Stage 1 work.
4. **Hysteresis can delay alerts.** Two consecutive chunks above threshold
   means up to 1 s of latency *after* the model first becomes suspicious.
   The single-chunk-override at score > 0.85 mitigates this for obvious
   cases.
5. **Fixture-dependent E2E results.** Acceptance criteria G2/G3 are measured
   against curated fixtures, not arbitrary audio in the wild. This is
   appropriate for a hackathon MVP and is documented honestly.

---

## 8. Mapping back to the source documents

If a reviewer asks "where in the architecture does X come from?", here is
the lookup table:

| Stage 1 module / decision           | A§           | J p.   |
| ----------------------------------- | ------------ | ------ |
| Live capture, passive observer      | §4, §5       | p.4, 6 |
| 500 ms chunking                     | §5           | p.4, 6 |
| 10 s rolling buffer                 | §5           | p.6    |
| SNR thresholds (12 / 8 dB)          | §6           | p.6    |
| Linear/Mel/CQT spectrograms         | §7.1–7.3     | p.7    |
| Phase heatmap                       | §7.4         | p.7, 8 |
| Pitch contour + smoothness          | §7.5         | p.8    |
| Spectral flux                       | §7.6         | p.8    |
| MFCC Δ ΔΔ                           | §7.7         | —      |
| Subband energy (5 bands)            | §7.8         | —      |
| AASIST / AASIST-L choice            | §8           | p.9, 18|
| Score → state (0.30, 0.70)          | §8, §14      | p.9, 17|
| Artifact timeline format            | §12          | p.11   |
| Advisory output (4 states)          | §14, §15     | p.17   |
| Agent recommended actions           | §15          | p.17   |
| Tech stack                          | §18          | p.18   |
| Privacy (no persistence)            | §5           | p.4    |

---

## 9. Stage-2 hand-off (preserved interfaces)

Stage 1 implementers must not violate the following — they are the seams
Stage 2 plugs into without rewrites:

```python
# classifier/protocol.py
class Scorer(Protocol):
    def score(self, audio: np.ndarray) -> float: ...

# pipeline/state_engine.py
def compute_final(
    component_scores: dict[str, float],   # Stage 1 sends {"stage1": s}
    snr_gate: GateState,
) -> RiskState: ...

# api versioning
GET  /v1/risk/current
WS   /v1/ws/risk
# Stage 2 will add /v2/ alongside, never replace /v1/
```

The single most expensive mistake Stage 1 can make is to hard-code "stage1"
anywhere in the state engine, the timeline, or the API — they must all be
score-source-agnostic.

---

## 10. Quick checklist (print and tick off)

- [ ] All Phase 0 foundation tasks pass.
- [ ] `MicSource` and `FileSource` both emit 8000-sample float32 chunks at 16 kHz.
- [ ] `RollingBuffer` capped at 160 000 samples.
- [ ] SNR thresholds match A§6 *exactly* (12 / 8).
- [ ] All 9 forensic features compute without NaN/Inf on every fixture.
- [ ] AASIST checkpoint loads; fallback classifier also works with flag.
- [ ] Score → state boundaries pinned at 0.30 / 0.70.
- [ ] Grey overrides everything else in the state engine.
- [ ] Timeline ring buffer holds 20 entries; oldest drops on overflow.
- [ ] WebSocket pushes ≥ 1 update per 500 ms.
- [ ] Dashboard reflects state change within 1 chunk.
- [ ] `pytest -q` is green.
- [ ] `tests/e2e/test_pipeline.py` is green on the four fixtures.
- [ ] p95 per-chunk latency < 200 ms on the target laptop.
- [ ] README documents the `USE_FALLBACK_CLASSIFIER` flag.
- [ ] No audio persisted to disk in default config.

When every box is ticked, Stage 1 is done. Stage 2 starts.

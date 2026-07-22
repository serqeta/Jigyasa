# VoiceShield — Architecture (two entry paths, one detection core)

Both paths feed the **same detection core**; they differ only in how audio
arrives and whether the cascade runs (batch = score everything; live = screen
+ probe to stay real-time).

---

## A. Uploaded-file path  (`POST /v2/analyze`  →  batch, full ensemble every chunk)

```
 ┌───────────┐   .wav/.mp3    ┌──────────────┐   500 ms chunks   ┌───────────────┐
 │  Browser  │ ─────────────► │  FileSource  │ ────────────────► │ RollingBuffer │
 │ (upload)  │  multipart     │  (decode,    │   4 s window      │   (10 s ring) │
 └───────────┘                │   →16 kHz)   │                   └──────┬────────┘
                              └──────────────┘                          │
                                                                        ▼
                                                   ┌───────────────────────────────┐
                                                   │        DETECTION CORE          │
                                                   │   (cascade = OFF → deep-scan   │
                                                   │        EVERY chunk)            │
                                                   └───────────────┬───────────────┘
                                                                   │  per-chunk TimelineEntry
                                                                   ▼
                        ┌───────────────────────────────────────────────────────┐
                        │  Response: [entries...] + summary  ──► Dashboard render │
                        │  (score, state, per-model scores, WHY, spectrograms)    │
                        └───────────────────────────────────────────────────────┘
                                                                   │  on request
                                                                   ▼
                                            ┌──────────────────────────────────────┐
                                            │  /v2/report → forensic HTML (PDF):     │
                                            │  verdict + rationale + detector table  │
                                            │  + spectrograms + SHA-256 + evidence   │
                                            └──────────────────────────────────────┘
```

---

## B. Real-time path  (browser mic → WebSocket → cascade)

```
 ┌────────────────────┐  float32 PCM   ┌───────────────────┐   ┌───────────────┐
 │  Browser mic        │  16 kHz frames │  WS /v2/ws/ingest │──►│ RollingBuffer │
 │  getUserMedia       │ ──────────────►│   → StreamSource  │   │   (10 s ring) │
 │  (AEC/AGC/NS OFF —   │   (streaming)  └───────────────────┘   └──────┬────────┘
 │   keep artifacts)    │                                               │ 500 ms chunk
 └────────────────────┘                                                ▼
                                                   ┌───────────────────────────────┐
                                                   │        DETECTION CORE          │
                                                   │   (cascade = ON, see below)    │
                                                   └───────────────┬───────────────┘
                                                                   │ TimelineEntry per chunk
                                                                   ▼
 ┌────────────────────┐   push (JSON)    ┌─────────────────────────────────────────┐
 │  Dashboard (live)   │ ◄──────────────── │  WS /v2/ws/risk                          │
 │  badge + WHY + specs│   ~every 500 ms   │  (state, score, per-model, rationale)    │
 └────────────────────┘                   └─────────────────────────────────────────┘

   CASCADE (real-time budget):
     every chunk ─► NII screener only  ──clean──►  stay light (~25 ms)
                          │  suspicious (≥ trigger)         ▲
                          ▼                                 │ 4 clean chunks
                    engage FULL ensemble ─── periodic probe ┘ (cool down)
                    (also a forced deep probe every N chunks, so a
                     screener miss can't hide a fake indefinitely)
```

---

## The shared DETECTION CORE (both paths run this per chunk)

```
        4 s window (from RollingBuffer)
                    │
                    ▼
        ┌───────────────────────┐     GREY (no speech / low SNR)
        │  SNR gate  +  VAD      │ ─────────────────────────────────► state = GREY
        │  (speech_active?)      │                                     (no scoring)
        └───────────┬───────────┘
                    │ speech
                    ▼
        ┌───────────────────────┐
        │  feature extraction    │  (spectrograms, pitch, phase — reused by
        │  (once, shared)        │   rule scorers + visual evidence)
        └───────────┬───────────┘
                    ▼
   ┌──────────────────────── ENSEMBLE (score in parallel) ────────────────────────┐
   │                                                                               │
   │  SYNTHESIS (what was said is machine-made)      PHYSICAL        EXPLAIN-ONLY   │
   │  ┌─────────────┐ ┌──────┐ ┌───────┐ ┌───────┐   ┌──────────┐   ┌────────────┐ │
   │  │ NII MMS-300M│ │XLS-R │ │ WavLM │ │ codec │   │  replay  │   │ phase/pitch│ │
   │  │ w=0.45 pk.8 │ │0.15  │ │ 0.15  │ │*obs.* │   │ 0.25 pk.6│   │  w=0  cue  │ │
   │  │ primary+screen│ .8   │ │ pk.6  │ │ w=0   │   │ EchoFake │   │ (display)  │ │
   │  └─────────────┘ └──────┘ └───────┘ └───────┘   └──────────┘   └────────────┘ │
   │   AUC .997        SSL       ITW      Codecfake    LoRA, AUC       rule-based    │
   │   telephony .92-.95                  (root-cause,  .97 x-chan                   │
   │                                       calibrating)                             │
   └───────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────┐
        │  FUSION  (state_engine.fuse_scores)            │
        │  renormalized weighted mean                    │
        │  ⌊ floored by PEAK-EVIDENCE rule ⌋             │
        │  (one trusted detector isn't averaged away)    │
        └───────────────────────┬───────────────────────┘
                                ▼
        ┌───────────────────────────────────────────────┐
        │  STATE ENGINE  (hysteresis + smoothing)        │
        │  GREEN <0.30 · AMBER 0.30–0.70 · RED ≥0.70     │
        │  2 chunks to escalate · instant RED ≥0.85      │
        │  RED never auto-de-escalates                   │
        └───────────┬───────────────────────┬───────────┘
                    ▼                        ▼
        ┌───────────────────┐    ┌──────────────────────────────┐
        │  explain_verdict  │    │  TimelineEntry                │
        │  WHY: driver,     │    │  (score, state, per-model,    │
        │  consensus vs peak│    │   explanation, spectrograms)  │
        │  reasons, cue     │    └──────────────────────────────┘
        └───────────────────┘
```

**One line:** *same core, two front doors* — uploads deep-scan every chunk for a
thorough case file; the live mic screens cheaply and probes on suspicion to hold
the real-time budget (~<300 ms/chunk, well under the 500 ms ceiling).
```
```

*Notes:* `codec` is present but **weight 0 / observing** (root-cause layer, under
calibration); `phase_pitch` runs at weight 0 for explainability only; `stage1`
(AASIST) and `spec` (AST) are retired but slot-compatible.

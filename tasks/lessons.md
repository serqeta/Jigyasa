# Lessons

## Format

Each lesson is one line in the form: `[CATEGORY] Pattern — Rule`.

## Entries

- [SETUP] Antigravity uses `.agents/` rather than `.claude/` for custom rules and skills.
- [SCOPE] When fixing a bug, don't opportunistically refactor adjacent code — rule: one commit, one concern.
- [VERIFICATION] When claiming "tests pass," show the actual test output — rule: no unproven claims.
- [PERF] `librosa.to_mono` and `librosa.yin` trigger numba JIT compilation on first call (3–4s each) — replace `to_mono` with `audio.mean(axis=0)` and warm up `yin` via `_extract_features(zeros)` at server startup.
- [PERF] `scipy.signal.resample_poly` is a drop-in numba-free replacement for `librosa.resample` — use it whenever resampling non-16kHz input.
- [PERF] Never call feature extraction twice per chunk — extract once and pass the feature dict to both the scorer and the artifact name function.
- [CALIBRATION] FallbackScorer's `phase_discontinuity` weight (0.35) is too aggressive for MP3 inputs — MP3 codec compression creates phase artifacts identical to synthetic speech artifacts, causing false AMBER on genuine voices.
- [AUDIO] Real voices uploaded as MP3 will score 0.30–0.45 on FallbackScorer due to MP3 codec artifacts, not because they are synthetic — always note the codec when interpreting results.
- [DATA] `compute_f0()` returns an ndarray with NaN for unvoiced frames; `compute_subband_energy()` returns a dict of dB floats — both are already computed in the pipeline but not yet serialized to the API response.
- [API] `TimelineEntry.to_dict()` uses `dataclasses.asdict()` — adding new optional fields to the dataclass automatically serializes them; no custom serializer needed.
- [FRONTEND] LangGraph SDK messages must use `{role: "user"}` not `{type: "human"}` — the SDK coerces the latter to LangChain wire format the Python server rejects. (From Companion project but applies universally.)
- [FRONTEND] Never hardcode `pii_stripped: false` — identity node manages that flag. (From Companion project.)

## How to use this file

- Review at session start.
- Add after every correction from the user.
- Consolidate weekly: merge similar lessons, delete obsolete ones.
- When a lesson is followed consistently, consider moving it to `AGENTS.md` as a hard rule.

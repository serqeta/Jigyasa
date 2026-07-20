# Jigyasa
Real-time Audio Forensics for Voice Clone Detection

## VoiceShield

- **Stage 1**: 500 ms chunk pipeline, 10 s privacy-preserving rolling buffer,
  SNR quality gate, forensic features, AASIST-L scorer, GREEN/AMBER/RED/GREY
  risk states, `/v1/` API + advisory dashboard.
- **Stage 2**: pretrained ensemble (AASIST + XLS-R + AST + WavLM +
  rule-based Phase/Pitch), DSP replay detection, weighted risk fusion,
  audit evidence export (spectrogram/phase PNGs + SHA-256 + scores JSON),
  `/v2/` API. GPU (CUDA) used automatically when available.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/download_model.py     # AASIST-L + Stage 2 HF checkpoints (~2 GB)
python scripts/run_file.py path/to/audio.wav        # full ensemble demo
python scripts/run_file.py path/to/audio.wav --stage1   # Stage 1 only
python scripts/run_live.py                          # live microphone
```

Dashboard: `Interface/` (React/Vite) — `npm install && npm run dev`.

Config knobs live in `voiceshield/config.py` (`FUSION_WEIGHTS`, `HF_SCORERS`,
`USE_FALLBACK_CLASSIFIER`, `ENABLE_REPLAY_DETECTION`). Model licenses:
`NOTICE.md`. Harness: `make check`, `make test`, `make smoke`.

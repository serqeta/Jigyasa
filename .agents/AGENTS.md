# Project context

VoiceShield - Stage 1 Implementation (MVP): A fast, real-time synthetic voice detection pipeline designed to flag AI voice fraud (synthetic / cloned / TTS) within the first 10 seconds of a call.

## Stack

- Language: Python 3.11+
- Framework: FastAPI + Uvicorn, websockets
- Frontend: Vanilla HTML/JS/CSS in Interface/
- Signal/DSP: numpy, scipy, librosa
- ML: torch (CPU), torchaudio, onnxruntime
- Audio I/O: sounddevice, soundfile
- Testing: pytest, pytest-asyncio, hypothesis

## Code style

- All numeric computation uses float32.
- Functions over 20 lines need docstrings.
- Never add inline comments unless requested.
- Maintain strict modularity as per Phase 5 directory layout.
- Use explicit typing and structured logging.

## Do's and don'ts

- Check for existing implementations before adding new ones.
- Never create documentation unless explicitly requested.
- Never add console.log, print statements, or debug output to committed code.
- Ask before installing new dependencies.

## Workflow orchestration

1. **Plan mode default.** For any non-trivial task (3+ steps or architectural decisions), enter plan mode.
2. **Subagent strategy.** Use subagents for research, broad codebase audits, or separate component work.
3. **Self-improvement loop.** Update `tasks/lessons.md` after corrections.
4. **Verification before done.** Prove it works. Run tests (`pytest -q`), check logs.
5. **Demand elegance, balanced.** For non-trivial changes, choose elegance. Skip for simple fixes.
6. **Autonomous bug fixing.** When given a bug report, point at logs, errors, tests, then resolve.

## Task management

- Use `tasks/todo.md` for local project task tracking.
- Update `tasks/progress.md` with phase summaries.
- Track lessons learned in `tasks/lessons.md`.
- Keep findings in `tasks/findings.md`.

## Core principles

- **Simplicity first.** Every change as simple as possible. Minimal impact.
- **No laziness.** Find root causes. No temporary fixes.
- **Privacy.** No call recording persisted to disk beyond the 10-second buffer.

## Adversarial framing (self-reminder)

When asked "is this right," answer with the strongest counterargument first, then the assessment. Never give uniform confidence. If unsure, say so with a calibrated estimate.

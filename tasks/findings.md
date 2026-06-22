# Findings

## Research Log

### What I looked at
- SETUP.md - Antigravity adaptation of the setup instructions.
- Stage_1.md - VoiceShield architecture documentation.

### What I learned
- The project requires an audio pipeline running real-time inferences every 500ms chunks on a 10s rolling buffer.
- Antigravity uses `.agents/AGENTS.md` instead of `.claude/CLAUDE.md`.

### Technical decisions
- Decided to structure the project using `Interface/` alongside `voiceshield/` backend, based on user input.

### Open questions
- None at this time.

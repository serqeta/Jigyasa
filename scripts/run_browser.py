"""
Serve VoiceShield with browser-microphone input: the dashboard captures
your voice via getUserMedia and streams it to /v2/ws/ingest; the cascade
pipeline (Stage 1 screening → Stage 2 ensemble) scores it live.

Usage:
    python scripts/run_browser.py [--port 8000]

Then open the React Interface (cd Interface && npm run dev), pick the
"Microphone" mode, and speak.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceShield — browser microphone mode")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    from voiceshield.api.app import create_app
    from voiceshield.audio.stream_source import StreamSource
    from voiceshield.classifier import get_scorers
    from voiceshield.pipeline.runner import PipelineRunner

    source = StreamSource()
    # No cascade: the ensemble is two orthogonal detectors (NII synthesis +
    # replay physical), both cheap enough (~100 ms) to run on every chunk.
    # There is nothing to screen/defer, so score both directly — this also
    # removes the replay blind spot a synthesis-only screener would create.
    runner = PipelineRunner(source, ensemble=get_scorers(), cascade=False)
    app = create_app(runner)

    print(f"\nVoiceShield (browser-mic mode) on http://localhost:{args.port}")
    print("Open the Interface (cd Interface && npm run dev), choose 'Microphone', and speak.")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

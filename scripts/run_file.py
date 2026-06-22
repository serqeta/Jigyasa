"""
Feed a WAV file through the VoiceShield Stage 1 pipeline and serve the dashboard.

Usage:
    python scripts/run_file.py <wav_path> [--port 8000] [--fallback]
"""

import argparse
import os
import sys
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceShield Stage 1 — file demo")
    parser.add_argument("wav", help="Path to input WAV file")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fallback", action="store_true",
                        help="Force rule-based fallback classifier")
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print(f"Error: file not found: {args.wav}", file=sys.stderr)
        sys.exit(1)

    if args.fallback:
        import voiceshield.config as cfg
        cfg.USE_FALLBACK_CLASSIFIER = True

    import uvicorn

    from voiceshield.api.app import create_app
    from voiceshield.audio.source import FileSource
    from voiceshield.classifier import get_scorer
    from voiceshield.pipeline.runner import PipelineRunner

    source = FileSource(args.wav)
    scorer = get_scorer()
    runner = PipelineRunner(source, scorer)
    app = create_app(runner)

    url = f"http://localhost:{args.port}"
    print(f"\nVoiceShield serving at {url}")
    print(f"Input: {args.wav}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

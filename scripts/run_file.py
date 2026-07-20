"""
Feed a WAV file through the VoiceShield pipeline and serve the dashboard.
Runs the full Stage 2 ensemble by default; --stage1 restricts to the
single Stage 1 scorer.

Usage:
    python scripts/run_file.py <wav_path> [--port 8000] [--fallback] [--stage1]
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
    parser.add_argument("--stage1", action="store_true",
                        help="Stage 1 single-scorer mode (skip the ensemble)")
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
    from voiceshield.classifier import get_scorer, get_scorers
    from voiceshield.pipeline.runner import PipelineRunner

    source = FileSource(args.wav)
    if args.stage1:
        runner = PipelineRunner(source, get_scorer())
    else:
        # File-fed serving simulates a live call → same cascade as run_live
        runner = PipelineRunner(source, ensemble=get_scorers(), cascade=True)
    app = create_app(runner)

    url = f"http://localhost:{args.port}"
    print(f"\nVoiceShield serving at {url}")
    print(f"Input: {args.wav}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

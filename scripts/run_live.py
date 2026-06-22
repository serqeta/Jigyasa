"""
Capture live microphone audio through the VoiceShield Stage 1 pipeline.

Usage:
    python scripts/run_live.py [--port 8000] [--fallback] [--device DEVICE_ID]
"""

import argparse
import os
import sys
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceShield Stage 1 — live demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fallback", action="store_true",
                        help="Force rule-based fallback classifier")
    parser.add_argument("--device", type=int, default=None,
                        help="sounddevice input device index")
    args = parser.parse_args()

    if args.fallback:
        import voiceshield.config as cfg
        cfg.USE_FALLBACK_CLASSIFIER = True

    try:
        from voiceshield.audio.source import MicSource
    except Exception as exc:
        print(f"Cannot open microphone: {exc}", file=sys.stderr)
        print("Install PortAudio (apt install portaudio19-dev) or use run_file.py", file=sys.stderr)
        sys.exit(1)

    import uvicorn

    from voiceshield.api.app import create_app
    from voiceshield.classifier import get_scorer
    from voiceshield.pipeline.runner import PipelineRunner

    source = MicSource(device=args.device)
    scorer = get_scorer()
    runner = PipelineRunner(source, scorer)
    app = create_app(runner)

    url = f"http://localhost:{args.port}"
    print(f"\nVoiceShield listening on microphone → {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

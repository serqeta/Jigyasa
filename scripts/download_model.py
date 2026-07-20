"""
Download all pretrained checkpoints: AASIST-L (Stage 1) and the Stage 2
ensemble models from the Hugging Face Hub.

Usage:
    python scripts/download_model.py
"""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "aasist_l.pth")

# GitHub raw URL for the AASIST-L checkpoint.
# Source: https://github.com/clovaai/aasist  (MIT License)
_URL = "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth"


def download(url: str, dest: str) -> None:
    print("Downloading AASIST-L checkpoint…")
    print(f"  Source : {url}")
    print(f"  Dest   : {dest}")
    try:
        urllib.request.urlretrieve(url, dest, _progress)
        print()
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _progress(count: int, block: int, total: int) -> None:
    pct = min(100, count * block * 100 // max(total, 1))
    print(f"\r  {pct:3d}%", end="", flush=True)


def download_stage2_models() -> None:
    """Snapshot the Stage 2 ensemble checkpoints into config.HF_CACHE_DIR."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from huggingface_hub import snapshot_download

    from voiceshield import config

    for name, spec in config.HF_SCORERS.items():
        print(f"Fetching Stage 2 '{name}' checkpoint: {spec['model_id']} …")
        path = snapshot_download(spec["model_id"], cache_dir=config.HF_CACHE_DIR)
        print(f"  → {path}")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"Model already present: {MODEL_PATH} ({size_kb:.0f} KB)")
    else:
        print("Fetching AASIST-L checkpoint (clovaai/aasist, MIT License)…\n")
        download(_URL, MODEL_PATH)
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"Saved {size_kb:.0f} KB → {MODEL_PATH}")

    download_stage2_models()
    print("\nDone. Run:  python scripts/run_live.py")


if __name__ == "__main__":
    main()

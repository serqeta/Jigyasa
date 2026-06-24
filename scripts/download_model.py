"""
Download the AASIST-L pretrained checkpoint from the clovaai/aasist repository.

Usage:
    python scripts/download_model.py
"""

import hashlib
import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "aasist_l.pth")

# GitHub raw URL for the AASIST-L checkpoint.
# Source: https://github.com/clovaai/aasist  (MIT License)
_URL = "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth"


def download(url: str, dest: str) -> None:
    print(f"Downloading AASIST-L checkpoint…")
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


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"Model already present: {MODEL_PATH} ({size_kb:.0f} KB)")
        return

    print("Fetching AASIST-L checkpoint (clovaai/aasist, MIT License)…\n")
    download(_URL, MODEL_PATH)

    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"Saved {size_kb:.0f} KB → {MODEL_PATH}")
    print("\nDone. Run:  python scripts/run_live.py")


if __name__ == "__main__":
    main()

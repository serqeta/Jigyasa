"""
Download the AASIST-L pretrained checkpoint from the clovaai/aasist release.

Usage:
    python scripts/download_model.py
"""

import hashlib
import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "aasist_l.pth")

# Google Drive direct download URL for the AASIST-L checkpoint
# Source: https://github.com/clovaai/aasist (MIT License)
# File ID from the repo's README
_FILE_ID = "1-YLXv65WBYJm84h0CbEYuqWL8CAvXJSM"
_GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={_FILE_ID}"

# SHA-256 of the official checkpoint (update if the upstream file changes)
_EXPECTED_SHA256 = None  # set to the hash once you have the file


def download(url: str, dest: str) -> None:
    print(f"Downloading {url}")
    print(f"  → {dest}")
    try:
        urllib.request.urlretrieve(url, dest, _progress)
        print()
    except Exception as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        sys.exit(1)


def _progress(count: int, block: int, total: int) -> None:
    pct = min(100, count * block * 100 // max(total, 1))
    print(f"\r  {pct:3d}%", end="", flush=True)


def verify(path: str) -> None:
    if _EXPECTED_SHA256 is None:
        print("  (SHA-256 check skipped — hash not configured)")
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != _EXPECTED_SHA256:
        print(f"  SHA-256 MISMATCH!\n  Expected: {_EXPECTED_SHA256}\n  Got:      {got}", file=sys.stderr)
        sys.exit(1)
    print(f"  SHA-256 OK: {got[:16]}…")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        verify(MODEL_PATH)
        return

    print("Fetching AASIST-L checkpoint…")
    print("  License: MIT — clovaai/aasist")
    print()

    download(_GDRIVE_URL, MODEL_PATH)
    verify(MODEL_PATH)

    size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
    print(f"Saved {size_mb:.1f} MB → {MODEL_PATH}")
    print("\nDone. You can now run:")
    print("  python scripts/run_file.py <wav>")


if __name__ == "__main__":
    main()

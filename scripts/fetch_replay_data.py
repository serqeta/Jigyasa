"""
Sample a balanced replay-detection training set from
DynamicSuperb/SpoofDetection_ASVspoof2017 (real replay attacks: label
authentic vs spoofed) via the anonymous datasets-server rows API.

Writes <out>/authentic/*.wav and <out>/spoofed/*.wav (16 kHz mono).

Usage:
    python scripts/fetch_replay_data.py --out <dir> --per-class 300
"""

import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SR = 16000
DS = "DynamicSuperb/SpoofDetection_ASVspoof2017"
ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset={ds}"
    "&config=default&split=test&offset={off}&length={n}"
)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "voiceshield-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=300)
    args = ap.parse_args()

    import librosa

    dirs = {
        "authentic": os.path.join(args.out, "authentic"),
        "spoofed": os.path.join(args.out, "spoofed"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    saved = {"authentic": 0, "spoofed": 0}

    off = 0
    page = 100
    while min(saved.values()) < args.per_class and off < 20000:
        try:
            rows = json.loads(_get(ROWS.format(ds=DS, off=off, n=page)))["rows"]
        except Exception as e:
            print("page error", off, e)
            off += page
            continue
        if not rows:
            break
        for row in rows:
            r = row["row"]
            label = r["label"]  # authentic | spoofed
            if label not in saved or saved[label] >= args.per_class:
                continue
            try:
                raw = _get(r["audio"][0]["src"])
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
            if len(audio) < SR:
                continue
            fn = os.path.splitext(r["file"])[0]
            sf.write(os.path.join(dirs[label], f"{fn}.wav"), audio.astype(np.float32), SR)
            saved[label] += 1
        off += page
        if off % 1000 == 0:
            print(f"offset {off}: {saved}", flush=True)

    print("done:", saved, "→", os.path.abspath(args.out))


if __name__ == "__main__":
    main()

"""
Stream a balanced sample of ASVspoof 2019 PA (physical access = replay)
directly out of the 16 GB DataShare zip via HTTP range requests — no full
download. Writes <out>/bonafide/*.wav and <out>/spoof/*.wav (16 kHz mono).

PA spoof = replay attack (the target domain). Public data (ODC-BY).

Usage:
    python scripts/fetch_pa2019.py --out <dir> --per-class 300
"""

import argparse
import io
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
SR = 16000
URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/852ec9ad-fe51-48e7-93bd-5259b94e25d6/content"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--split", default="train")  # train | dev
    args = ap.parse_args()

    import librosa
    from remotezip import RemoteZip

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)

    for c in ("bonafide", "spoof"):
        os.makedirs(os.path.join(args.out, c), exist_ok=True)

    print("opening remote zip (reading central directory)…", flush=True)
    with RemoteZip(URL) as z:
        names = z.namelist()
        print(f"  {len(names)} entries", flush=True)
        # protocol: ASVspoof2019.PA.cm.<split>.tr[n/l].txt  cols: spk file - - label
        proto = next(
            n
            for n in names
            if "cm_protocols" in n and f".{args.split}." in n and n.endswith(".txt")
        )
        print("  protocol:", proto, flush=True)
        text = z.read(proto).decode()
        rows = [ln.split() for ln in text.strip().splitlines()]
        # label is last column: bonafide | spoof
        want = {"bonafide": [], "spoof": []}
        for r in rows:
            lab = r[-1]
            if lab in want:
                want[lab].append(r[1])  # file id
        import random

        random.seed(0)
        for lab in want:
            random.shuffle(want[lab])

        flac_by_id = {}
        for n in names:
            if n.endswith(".flac"):
                flac_by_id[os.path.splitext(os.path.basename(n))[0]] = n

        saved = {"bonafide": 0, "spoof": 0}
        for lab in ("bonafide", "spoof"):
            for fid in want[lab]:
                if saved[lab] >= args.per_class:
                    break
                path = flac_by_id.get(fid)
                if not path:
                    continue
                try:
                    raw = z.read(path)
                    a, sr = sf.read(io.BytesIO(raw), dtype="float32")
                except Exception:
                    continue
                if a.ndim > 1:
                    a = a.mean(axis=1)
                if sr != SR:
                    a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
                if len(a) < SR:
                    continue
                sf.write(os.path.join(args.out, lab, f"{fid}.wav"), a.astype(np.float32), SR)
                saved[lab] += 1
                if sum(saved.values()) % 50 == 0:
                    print(f"  saved {saved}", flush=True)
        print("done:", saved, "→", os.path.abspath(args.out), flush=True)


if __name__ == "__main__":
    main()

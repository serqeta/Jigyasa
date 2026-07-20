"""
Sample labeled clips from public anti-spoofing benchmarks into the eval set.

Sources (all anonymous, via the datasets-server rows API — no shard downloads):
  ajaykarthick/wavefake-audio            7 GAN vocoders + real LJSpeech
  SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF   100+ TTS/VC attack systems
  SpeechAntiSpoofingBenchmarks/InTheWild         real-world wild deepfakes

Writes 16 kHz mono WAVs into <out>/bench_genuine/ and <out>/bench_fake/,
filenames tagged with source + generator where available.

Usage:
    python scripts/fetch_benchmark.py --out <eval_set_dir>
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
ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length={n}"


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "voiceshield-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _rows(ds: str, split: str, offset: int, n: int = 100, cfg: str = "default"):
    data = json.loads(_get(ROWS.format(ds=ds, cfg=cfg, split=split, off=offset, n=n)))
    return data["rows"]


def _save(audio_cell, path: str) -> bool:
    """audio_cell: list of {src, type} — fetch, resample to 16k mono, save."""
    import librosa

    try:
        src = audio_cell[0]["src"]
        raw = _get(src)
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        try:
            audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        except Exception:
            return False
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    if len(audio) < SR:  # skip clips under 1 s
        return False
    sf.write(path, audio.astype(np.float32), SR)
    return True


def fetch_wavefake(gen_dir, fake_dir, n_rows=160):
    ds = "ajaykarthick/wavefake-audio"
    saved = {"R": 0}
    for off in range(0, n_rows, 100):
        for row in _rows(ds, "train", off, min(100, n_rows - off)):
            r = row["row"]
            tag = r["real_or_fake"]
            aid = r["audio_id"]
            out = gen_dir if tag == "R" else fake_dir
            name = f"wavefake_{tag}_{aid}.wav"
            if _save(r["audio"], os.path.join(out, name)):
                saved[tag] = saved.get(tag, 0) + 1
    print("  wavefake:", dict(sorted(saved.items())))


def fetch_asvspoof_df(gen_dir, fake_dir, offsets=(0, 7500), n=60):
    ds = "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF"
    saved = {"bonafide": 0, "spoof": 0}
    for off in offsets:
        for row in _rows(ds, "test", off, n):
            r = row["row"]
            label = "bonafide" if r["label"] == 0 else "spoof"
            notes = json.loads(r.get("notes") or "{}")
            attack = str(notes.get("attack_id", "NA")).replace("/", "-")
            out = gen_dir if label == "bonafide" else fake_dir
            name = f"asvspoof21_{attack}_{os.path.splitext(r['path'])[0]}.wav"
            if _save(r["audio"], os.path.join(out, name)):
                saved[label] += 1
    print("  asvspoof2021-DF:", saved)


def fetch_inthewild(gen_dir, fake_dir, offsets=(0, 15000), n=60):
    ds = "SpeechAntiSpoofingBenchmarks/InTheWild"
    saved = {"bonafide": 0, "spoof": 0}
    for off in offsets:
        for row in _rows(ds, "test", off, n):
            r = row["row"]
            label = "bonafide" if r["label"] == 0 else "spoof"
            out = gen_dir if label == "bonafide" else fake_dir
            name = f"itw_{label}_{os.path.splitext(r['path'])[0]}.wav"
            if _save(r["audio"], os.path.join(out, name)):
                saved[label] += 1
    print("  in-the-wild:", saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gen = os.path.join(args.out, "bench_genuine")
    fake = os.path.join(args.out, "bench_fake")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(fake, exist_ok=True)
    print("Sampling benchmarks…")
    fetch_wavefake(gen, fake)
    fetch_asvspoof_df(gen, fake)
    fetch_inthewild(gen, fake)
    print("Done →", os.path.abspath(args.out))


if __name__ == "__main__":
    main()

"""
Does the SYNTHESIS ensemble survive a real phone channel?

Applies genuine PSTN/telephony degradation to a real eval set via ffmpeg
(8 kHz + G.711 mu-law round-trip = exactly what a landline call does; also
AMR-NB cellular and G.722 wideband for contrast), then scores the shipped
ensemble (nii / ssl / wavlm + fusion) on clean vs degraded and reports the
AUC/EER drop. This is the free, honest answer to "will it work over the
phone?" — no telco needed.

Usage:
    python scripts/telephony_eval.py [--per-class 80]
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from voiceshield import config  # noqa: E402
from voiceshield.pipeline.state_engine import fuse_scores  # noqa: E402

SR = 16000
WIN = 4 * SR
# Real dataset from EchoFake (cached): bonafide = genuine, fake = modern
# synthesis (XTTS-v2 / StyleTTS2). Replay shards excluded — this tests
# whether SYNTHESIS detection survives the phone channel.
BASE = "https://huggingface.co/datasets/EchoFake/EchoFake/resolve/main/data/"
ECHO = {
    "data/train-00000-of-00004.parquet": 0,  # bonafide (genuine)
    "data/train-00002-of-00004.parquet": 1,
}  # fake (digital synthesis)

# codec round-trips: (encode args, note). Each decodes back to 16 kHz mono.
CODECS = {
    "clean": None,
    "g711_ulaw_8k": ["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"],
    "amr_nb_8k": ["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"],
    "g722_16k": ["-ar", "16000", "-ac", "1", "-c:a", "g722"],  # wideband telephony
}


def degrade(a16, enc_args):
    """Round-trip a 16 kHz waveform through a codec, return 16 kHz mono."""
    if enc_args is None:
        return a16
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t0:
        sf.write(t0.name, a16, SR)
        path = t0.name
    ext = ".amr" if "amrnb" in " ".join(enc_args) else ".wav"
    with (
        tempfile.NamedTemporaryFile(suffix=ext, delete=False) as t1,
        tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t2,
    ):
        enc, dec = t1.name, t2.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path, *enc_args, enc], check=True
        )
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-ar", str(SR), "-ac", "1", dec],
            check=True,
        )
        return sf.read(dec, dtype="float32")[0]
    finally:
        for f in (path, enc, dec):
            if os.path.exists(f):
                os.unlink(f)


def clip_score(scorer, a):
    if len(a) <= WIN:
        return scorer.score(a)
    return float(
        np.mean([scorer.score(a[i : i + WIN]) for i in range(0, len(a) - WIN + 1, 2 * SR)])
    )


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((pos > n).sum() + 0.5 * (pos == n).sum() for n in neg) / (len(pos) * len(neg)))


def eer(pos, neg):
    ths = np.unique(np.concatenate([pos, neg]))
    d = [
        ((neg >= t).mean() + (pos < t).mean()) / 2
        for t in ths
        if abs((neg >= t).mean() - (pos < t).mean()) < 0.05
    ]
    return float(min(d)) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=80)
    args = ap.parse_args()

    import io

    import librosa
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from voiceshield.classifier.hf_scorer import HFScorer
    from voiceshield.classifier.nii_scorer import NIIScorer

    rng = np.random.default_rng(0)
    scorers = {
        "nii": NIIScorer(),
        "ssl": HFScorer(config.HF_SCORERS["ssl"]["model_id"], 1),
        "wavlm": HFScorer(config.HF_SCORERS["wavlm"]["model_id"], 1),
    }

    # decode a real genuine+synthesis sample from EchoFake ONCE (clean 16 kHz)
    clips = []
    for shard, lab in ECHO.items():
        df = pd.read_parquet(hf_hub_download("EchoFake/EchoFake", shard, repo_type="dataset"))
        for i in rng.permutation(len(df))[: args.per_class]:
            try:
                a, sr = sf.read(io.BytesIO(df.iloc[i]["path"]["bytes"]), dtype="float32")
            except Exception:
                continue
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr != SR:
                a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
            if len(a) >= SR:
                clips.append((a, lab))
    print(
        f"eval clips: {len(clips)} (genuine={sum(1 for _, lab in clips if lab == 0)})", flush=True
    )

    for codec, enc in CODECS.items():
        rows = []
        for a16, lab in clips:
            try:
                a = degrade(a16, enc)
            except Exception:
                continue
            cs = {n: clip_score(s, a) for n, s in scorers.items()}
            rows.append((lab, cs, fuse_scores(cs)))
        y = np.array([r[0] for r in rows])
        print(f"\n=== {codec} (n={len(rows)}) ===")
        for n in ["nii", "ssl", "wavlm"]:
            sc = np.array([r[1][n] for r in rows])
            print(
                f"  {n:6s} AUC={auc(sc[y == 1], sc[y == 0]):.3f} EER={eer(sc[y == 1], sc[y == 0]):.3f}"
            )
        fu = np.array([r[2] for r in rows])
        print(
            f"  FUSED  AUC={auc(fu[y == 1], fu[y == 0]):.3f} EER={eer(fu[y == 1], fu[y == 0]):.3f}"
        )


if __name__ == "__main__":
    main()

"""
Pilot: replay-aware detector trained on EchoFake (real physical-replay
corpus, MIT), validated on our own 12 browser-mic recordings (the
deployment-channel test) FIRST, then EchoFake's held-out open set.

Replay task (detect the loudspeaker channel, content-agnostic):
  class 0 (not replay):  bonafide + fake         (live / digital synthesis)
  class 1 (replay):      replay_bonafide + replay_fake

Robust: per-file cached downloads (resumable, timeout) instead of
pandas-over-HTTP (which hangs); embeddings cached to disk; the
deployment-channel verdict is computed and printed BEFORE the open-set
eval so a slow download can never block the number that matters.

Usage:
    python scripts/echofake_pilot.py --per-class 800
"""

import argparse
import glob
import io
import os
import sys

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
SR = 16000
WIN = 4 * SR
REAL = "/home/manicguy/Projects/Jigyasa/eval_recordings"
CACHE = "/tmp/claude-1000/-home-manicguy-Projects-Jigyasa/9ffc4fa1-d014-4b1c-ac2c-4a04a3e3de71/scratchpad/echofake_emb"
TRAIN_SHARDS = {
    "data/train-00000-of-00004.parquet": 0,  # bonafide
    "data/train-00002-of-00004.parquet": 0,  # fake (digital synthesis)
    "data/train-00001-of-00004.parquet": 1,  # replay_bonafide
    "data/train-00003-of-00004.parquet": 1,  # replay_fake
}
OPEN_SHARDS = [f"data/open_set_eval-0000{i}-of-00003.parquet" for i in range(3)]


def get_shard(fn):
    from huggingface_hub import hf_hub_download

    return hf_hub_download("EchoFake/EchoFake", fn, repo_type="dataset")


def decode(cell):
    import librosa

    a, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
    return a


def clip_embed(nii, a):
    ws = [a[i : i + WIN] for i in range(0, max(1, len(a) - WIN + 1), 2 * SR)] or [a[:WIN]]
    return np.mean([nii.embed(w) for w in ws], axis=0)


def embed_shard(nii, fn, replay_label, per, rng, tag):
    """Embed `per` sampled rows from one cached shard; cache result to disk."""
    import pandas as pd

    cf = os.path.join(CACHE, f"{tag}_n{per}.npz")  # per-count in key (no stale reuse)
    if os.path.exists(cf):
        d = np.load(cf)
        return d["X"], d["y"]
    df = pd.read_parquet(get_shard(fn))
    idx = rng.permutation(len(df))[:per]
    X, y = [], []
    for i in idx:
        a = decode(df.iloc[i]["path"])
        if len(a) < SR:
            continue
        X.append(clip_embed(nii, a))
        lab = df.iloc[i]["label"]
        y.append(1 if lab.startswith("replay_") else 0)
    X, y = np.array(X), np.array(y)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cf, X=X, y=y)
    print(f"    {tag}: {len(y)} embedded (replay={int(y.sum())})", flush=True)
    return X, y


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
    return float(min(d)) if d else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=800)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from voiceshield.classifier.nii_scorer import NIIScorer

    rng = np.random.default_rng(0)
    nii = NIIScorer()
    half = args.per_class // 2

    print("embedding EchoFake train (cached per shard)…", flush=True)
    Xtr, ytr = [], []
    for i, (fn, lab) in enumerate(TRAIN_SHARDS.items()):
        X, y = embed_shard(nii, fn, lab, half, rng, f"train{i}")
        Xtr.append(X)
        ytr.append(y)
    Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
    print(f"  train: {len(ytr)} (replay={int(ytr.sum())} not={int((ytr == 0).sum())})", flush=True)

    print("embedding OUR recordings…", flush=True)
    files = sorted(glob.glob(f"{REAL}/*.wav"))
    Xours = np.array([clip_embed(nii, sf.read(f, dtype="float32")[0]) for f in files])
    yours = np.array([0 if os.path.basename(f).startswith("live_") else 1 for f in files])

    sc = StandardScaler().fit(Xtr)
    heads = {
        "logreg": LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, early_stopping=True),
    }
    fitted = {}
    print("\n########## DEPLOYMENT-CHANNEL VERDICT (our 12 recordings) ##########")
    for name, clf in heads.items():
        clf.fit(sc.transform(Xtr), ytr)
        fitted[name] = clf
        our = clf.predict_proba(sc.transform(Xours))[:, 1]
        print(f"\n=== {name} — OUR channel  AUC={auc(our[yours == 1], our[yours == 0]):.3f} ===")
        for f, yy, pp in sorted(zip(files, yours, our), key=lambda t: t[2]):
            print(f"   {os.path.basename(f):34s} {'REPLAY' if yy else 'live  '} {pp:.3f}")

    print("\nembedding EchoFake open_set_eval (in-corpus sanity, last)…", flush=True)
    Xoe, yoe = [], []
    for i, fn in enumerate(OPEN_SHARDS):
        X, y = embed_shard(nii, fn, None, 200, rng, f"open{i}")
        Xoe.append(X)
        yoe.append(y)
    Xoe, yoe = np.vstack(Xoe), np.concatenate(yoe)
    for name, clf in fitted.items():
        oe = clf.predict_proba(sc.transform(Xoe))[:, 1]
        print(
            f"  {name}: EchoFake open-set AUC={auc(oe[yoe == 1], oe[yoe == 0]):.3f} "
            f"EER={eer(oe[yoe == 1], oe[yoe == 0]):.3f}"
        )

    import pickle

    out = os.path.join(os.path.dirname(__file__), "..", "models", "replay_head.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "scaler": sc,
                "mlp": fitted["mlp"],
                "logreg": fitted["logreg"],
                "feat": "nii_embed",
                "trained_on": "EchoFake",
                "per_class": args.per_class,
            },
            f,
        )
    print("\nsaved", os.path.abspath(out))


if __name__ == "__main__":
    main()

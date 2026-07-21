"""
Train a channel-robust replay detector on the 300+300 ASVspoof2019 PA
sample with heavy channel augmentation (RawBoost-style), so the model
cannot overfit to the PA recording channel and must learn replay-intrinsic
structure. Features = LFCC mean+std + pop-noise/airflow cues.

Held-out test: our 12 browser-mic recordings (the deployment-channel test).

Usage:
    python scripts/train_replay_aug.py --pa <dir> [--naug 8]
"""

import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from voiceshield.features.lfcc import compute_lfcc  # noqa: E402

SR = 16000
WIN = 4 * SR
SCRATCH = "/tmp/claude-1000/-home-manicguy-Projects-Jigyasa/9ffc4fa1-d014-4b1c-ac2c-4a04a3e3de71/scratchpad"
REAL = "/home/manicguy/Projects/Jigyasa/eval_recordings"
_SOS_LO = butter(4, [20, 150], btype="bandpass", fs=SR, output="sos")
_SOS_SP = butter(4, [300, 3000], btype="bandpass", fs=SR, output="sos")


def pop_cues(a):
    lo, sp = sosfilt(_SOS_LO, a), sosfilt(_SOS_SP, a)
    win = int(0.02 * SR)
    env = np.sqrt(np.convolve(lo**2, np.ones(win) / win, mode="same")) + 1e-9
    return np.array([
        10 * np.log10((np.mean(lo**2) + 1e-12) / (np.mean(sp**2) + 1e-12)),
        float(np.max(env) / np.median(env)),
        float(((env - env.mean()) ** 4).mean() / (env.var() ** 2 + 1e-12)),
    ])


def feat(a):
    a = a[:WIN]
    L = compute_lfcc(a)
    return np.concatenate([L.mean(1), L.std(1), pop_cues(a)])


def augment(a, rng):
    """RawBoost-style channel randomization: gain, coloration EQ, band-limit,
    reverb, additive noise — applied to BOTH classes so channel is not a cue."""
    x = a.copy()
    # random coloration (spectral tilt via one biquad-ish shelf)
    if rng.random() < 0.8:
        lo = rng.uniform(50, 400)
        hi = rng.uniform(3000, 7500)
        sos = butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos")
        x = sosfilt(sos, x).astype(np.float32)
    # random reverb
    if rng.random() < 0.6:
        t60 = rng.uniform(0.05, 0.5)
        n = int(t60 * SR)
        ir = (rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / SR / t60)).astype(np.float32) * 0.3
        ir[0] = 1.0
        x = np.convolve(x, ir)[: len(a)].astype(np.float32)
    # additive noise
    if rng.random() < 0.7:
        snr = rng.uniform(5, 30)
        p = np.mean(x**2) + 1e-9
        nz = rng.standard_normal(len(x)).astype(np.float32)
        nz *= np.sqrt(p / (10 ** (snr / 10)) / (np.mean(nz**2) + 1e-12))
        x = x + nz
    # random gain
    x = x * rng.uniform(0.5, 1.5)
    m = np.max(np.abs(x)) + 1e-9
    return (0.9 * x / m).astype(np.float32)


def build(paths, label, naug, rng):
    X, y = [], []
    for p in paths:
        a = sf.read(p, dtype="float32")[0]
        X.append(feat(a)); y.append(label)
        for _ in range(naug):
            X.append(feat(augment(a[:WIN], rng))); y.append(label)
    return X, y


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(sum((pos > n).sum() + 0.5 * (pos == n).sum() for n in neg) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", required=True)
    ap.add_argument("--naug", type=int, default=8)
    args = ap.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    print(f"building augmented training set (naug={args.naug}/clip)…", flush=True)
    Xs, ys = build(sorted(glob.glob(f"{args.pa}/spoof/*.wav")), 1, args.naug, rng)
    Xb, yb = build(sorted(glob.glob(f"{args.pa}/bonafide/*.wav")), 0, args.naug, rng)
    X = np.array(Xs + Xb); y = np.array(ys + yb)
    print(f"  train vectors: {len(y)} (replay={int((y==1).sum())} bonafide={int((y==0).sum())})")

    files = sorted(glob.glob(f"{REAL}/*.wav"))
    Xte = np.array([feat(sf.read(f, dtype="float32")[0]) for f in files])
    yte = np.array([0 if os.path.basename(f).startswith("live_") else 1 for f in files])

    sc = StandardScaler().fit(X)
    for name, clf in [
        ("logreg", LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
        ("gbm", GradientBoostingClassifier(n_estimators=200, max_depth=3, subsample=0.8)),
    ]:
        clf.fit(sc.transform(X), y)
        pr = clf.predict_proba(sc.transform(Xte))[:, 1]
        pos, neg = pr[yte == 1], pr[yte == 0]
        print(f"\n=== {name} — OUR channel (held out) ===  AUC={auc(pos, neg):.3f}")
        for f, yy, pp in sorted(zip(files, yte, pr), key=lambda t: t[2]):
            print(f"   {os.path.basename(f):34s} {'REPLAY' if yy else 'live  '} {pp:.3f}")


if __name__ == "__main__":
    main()

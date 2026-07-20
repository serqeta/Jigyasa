"""
Train the LFCC replay classifier on ASVspoof 2019 PA (real replay corpus,
27 acoustic configs) and evaluate generalization to:
  1. a held-out slice of PA itself (in-corpus sanity),
  2. ASVspoof 2017 real-replay 200-sample set (cross-corpus real replay),
  3. our own 12 browser-mic recordings (the deployment-channel test).

The #3 number is the one that decides whether replay ships.

Usage:
    python scripts/eval_replay_pa.py --pa <dir> [--save]
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from voiceshield.features.lfcc import compute_lfcc  # noqa: E402

SR = 16000
WIN = 4 * SR
SCRATCH = "/tmp/claude-1000/-home-manicguy-Projects-Jigyasa/9ffc4fa1-d014-4b1c-ac2c-4a04a3e3de71/scratchpad"
REAL = "/home/manicguy/Projects/Jigyasa/eval_recordings"
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "replay_head.pkl")


def feat(p):
    a = sf.read(p, dtype="float32")[0][:WIN]
    L = compute_lfcc(a)
    return np.concatenate([L.mean(1), L.std(1)])


def load(d, label):
    return [(feat(p), label) for p in sorted(glob.glob(os.path.join(d, "*.wav")))]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((pos > n).sum() + 0.5 * (pos == n).sum() for n in neg) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pa", required=True)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = load(os.path.join(args.pa, "spoof"), 1) + load(os.path.join(args.pa, "bonafide"), 0)
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])
    # 80/20 in-corpus split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    print(f"PA train: replay={int((y[tr] == 1).sum())} bonafide={int((y[tr] == 0).sum())}")

    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced").fit(
        scaler.transform(X[tr]), y[tr]
    )

    def probs(paths_labels):
        Xt = np.array([p for p, _ in paths_labels])
        yt = np.array([lb for _, lb in paths_labels])
        pr = clf.predict_proba(scaler.transform(Xt))[:, 1]
        return pr[yt == 1], pr[yt == 0]

    p, n = (
        clf.predict_proba(scaler.transform(X[te]))[:, 1][y[te] == 1],
        clf.predict_proba(scaler.transform(X[te]))[:, 1][y[te] == 0],
    )
    print(f"\n1) PA held-out (in-corpus):      AUC={auc(p, n):.3f}")

    # ASVspoof2017 real replay
    a17 = os.path.join(SCRATCH, "replay_train")
    if os.path.isdir(a17):
        p, n = probs(
            load(os.path.join(a17, "spoofed"), 1) + load(os.path.join(a17, "authentic"), 0)
        )
        print(
            f"2) ASVspoof2017 real replay:     AUC={auc(p, n):.3f}  "
            f"(replay μ={np.mean(p):.2f}, genuine μ={np.mean(n):.2f})"
        )

    # OUR recordings — the deployment-channel test
    files = sorted(glob.glob(os.path.join(REAL, "*.wav")))
    rec = [(feat(f), 0 if os.path.basename(f).startswith("live_") else 1) for f in files]
    pr = clf.predict_proba(scaler.transform(np.array([r[0] for r in rec])))[:, 1]
    yr = np.array([r[1] for r in rec])
    print("\n3) OUR real recordings (held-out deployment channel):")
    for f, yy, pp in sorted(zip(files, yr, pr), key=lambda t: t[2]):
        print(f"   {os.path.basename(f):34s} {'REPLAY' if yy else 'live  '} prob={pp:.3f}")
    pos, neg = pr[yr == 1], pr[yr == 0]
    print(
        f"\n   OUR-CHANNEL AUC: {auc(pos, neg):.3f}   live max={neg.max():.2f}  replay min={pos.min():.2f}"
    )

    if args.save:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        # refit on all data before shipping
        scaler_f = StandardScaler().fit(X)
        clf_f = LogisticRegression(max_iter=3000, C=0.2, class_weight="balanced").fit(
            scaler_f.transform(X), y
        )
        with open(OUT, "wb") as f:
            pickle.dump({"scaler": scaler_f, "clf": clf_f, "feat": "lfcc_meanstd"}, f)
        print("\nsaved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()

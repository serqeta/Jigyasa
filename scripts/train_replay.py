"""
Train a learned replay/physical-access detector on top of the NII SSL
embedding (reused from the deepfake scorer → ~free at inference).

Train on real replay data (ASVspoof 2017, scripts/fetch_replay_data.py),
then VALIDATE on the held-out real recordings captured through our own
browser-mic path (eval_recordings/) — the true test of whether it
generalizes where the hand-tuned DSP detector failed.

Ships models/replay_head.pkl only if the real-recording separation is
convincing.

Usage:
    python scripts/train_replay.py --train <dir> --real eval_recordings
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SR = 16000
WIN = 4 * SR
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "replay_head.pkl")


def windows(a):
    if len(a) <= WIN:
        return [a]
    return [a[i : i + WIN] for i in range(0, len(a) - WIN + 1, 2 * SR)]


def embed_clip(nii, a):
    """Mean SSL embedding over a clip's windows."""
    return np.mean([nii.embed(w) for w in windows(a)], axis=0)


def load_dir(nii, d):
    X = []
    for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
        a = sf.read(p, dtype="float32")[0]
        X.append(embed_clip(nii, a))
    return np.array(X)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((pos > n).sum() + 0.5 * (pos == n).sum() for n in neg) / (len(pos) * len(neg)))


def eer(pos, neg):
    ths = np.unique(np.concatenate([pos, neg]))
    return float(
        min(
            ((neg >= t).mean() + (pos < t).mean()) / 2
            for t in ths
            if abs((neg >= t).mean() - (pos < t).mean()) < 0.05
        )
        if len(ths)
        else 0.5
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--real", default="eval_recordings")
    ap.add_argument("--save", action="store_true", help="write models/replay_head.pkl")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler

    from voiceshield.classifier.nii_scorer import NIIScorer

    nii = NIIScorer()

    print("Embedding training data (ASVspoof2017 replay)…")
    Xa = load_dir(nii, os.path.join(args.train, "authentic"))  # label 0
    Xs = load_dir(nii, os.path.join(args.train, "spoofed"))  # label 1 (replay)
    X = np.vstack([Xa, Xs])
    y = np.array([0] * len(Xa) + [1] * len(Xs))
    print(f"  authentic={len(Xa)}  spoofed(replay)={len(Xs)}  dim={X.shape[1]}")

    scaler = StandardScaler().fit(X)
    Xs_ = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")

    # honest in-corpus estimate via 5-fold CV
    cv_prob = cross_val_predict(clf, Xs_, y, cv=5, method="predict_proba")[:, 1]
    print(
        f"\nASVspoof2017 5-fold CV: AUC={auc(cv_prob[y == 1], cv_prob[y == 0]):.3f}  "
        f"EER={eer(cv_prob[y == 1], cv_prob[y == 0]):.3f}"
    )

    clf.fit(Xs_, y)

    # THE REAL TEST: our own recordings (never seen in training)
    print("\n=== Held-out REAL recordings (eval_recordings/) ===")
    real = sorted(glob.glob(os.path.join(args.real, "*.wav")))
    live_scores, replay_scores = [], []
    for p in real:
        name = os.path.basename(p)
        if name.startswith("live_"):
            grp = "live (genuine)"
        elif name.startswith("replay_"):
            grp = "replay (loudspeaker)"
        else:
            continue
        prob = float(
            clf.predict_proba(
                scaler.transform(embed_clip(nii, sf.read(p, dtype="float32")[0])[None])
            )[0, 1]
        )
        (replay_scores if grp.startswith("replay") else live_scores).append(prob)
        print(f"  {name:32s} {grp:20s} replay_prob={prob:.3f}")

    live_scores, replay_scores = np.array(live_scores), np.array(replay_scores)
    print(f"\nREAL live  (want low):   mean={live_scores.mean():.3f}  max={live_scores.max():.3f}")
    print(
        f"REAL replay(want high):  mean={replay_scores.mean():.3f}  min={replay_scores.min():.3f}"
    )
    print(f"REAL-recording AUC: {auc(replay_scores, live_scores):.3f}")
    gap = replay_scores.min() - live_scores.max()
    print(
        f"separation (min replay - max live): {gap:+.3f}  → {'SEPARABLE' if gap > 0 else 'OVERLAP'}"
    )

    if args.save:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "wb") as f:
            pickle.dump({"scaler": scaler, "clf": clf}, f)
        print("\nsaved", os.path.abspath(OUT))


if __name__ == "__main__":
    main()

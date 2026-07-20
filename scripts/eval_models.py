"""
Model-selection evaluation: score every candidate detector on a labeled
eval set (see scripts/build_evalset.py) and report per-model metrics.

Usage:
    python scripts/eval_models.py --set <dir> [--json out.json]

Directory layout expected: <dir>/{genuine,fake,genuine_codec,fake_codec}/*.wav
Clip score = mean over 4 s windows (2 s hop). Higher = more likely fake.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceshield import config  # noqa: E402

SR = config.SAMPLE_RATE
WIN, HOP = 4 * SR, 2 * SR


def clip_score(scorer, audio: np.ndarray) -> float:
    if len(audio) <= WIN:
        return scorer.score(audio)
    scores = [scorer.score(audio[i:i + WIN]) for i in range(0, len(audio) - WIN + 1, HOP)]
    return float(np.mean(scores))


def load_candidates() -> dict:
    """name -> scorer. Loads every candidate, enabled or not."""
    from voiceshield.classifier import get_scorer
    from voiceshield.classifier.fallback import FallbackScorer
    from voiceshield.classifier.hf_scorer import HFScorer
    from voiceshield.classifier.phase_pitch import PhasePitchScorer

    candidates = {
        "aasist": get_scorer(),
        "phase_pitch": PhasePitchScorer(),
        "fallback_rules": FallbackScorer(),
    }
    hf = {
        "xlsr_gustking": ("Gustking/wav2vec2-large-xlsr-deepfake-audio-classification", 1),
        "wavlm_itw": ("abhishtagatya/wavlm-base-960h-itw-deepfake", 1),
        "ast_asvspoof5": ("MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection", 1),
        "w2v2_melody": ("MelodyMachine/Deepfake-audio-detection-V2", 0),  # 0 = fake!
    }
    for name, (mid, spoof_label) in hf.items():
        try:
            candidates[name] = HFScorer(mid, spoof_label)
        except Exception as e:
            print(f"  ! {name} failed to load: {e}")
    return candidates


def collect(set_dir: str) -> list[tuple[str, str, int, np.ndarray]]:
    """[(group, filename, label, audio)] — label 1 = fake. Any subdirectory
    is a group; 'genuine' in its name means bonafide."""
    items = []
    for group in sorted(os.listdir(set_dir)):
        gdir = os.path.join(set_dir, group)
        if not os.path.isdir(gdir):
            continue
        label = 0 if "genuine" in group else 1
        for p in sorted(glob.glob(os.path.join(gdir, "*.wav"))):
            audio, sr = sf.read(p, dtype="float32")
            assert sr == SR
            items.append((group, os.path.basename(p), label, audio))
    return items


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC (probability a fake outscores a genuine)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((pos > n).sum() + 0.5 * (pos == n).sum() for n in neg)
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    items = collect(args.set)
    n_genuine = sum(1 for item in items if item[2] == 0)
    print(f"eval set: {len(items)} clips ({n_genuine} genuine)")

    candidates = load_candidates()
    results: dict[str, dict] = {}

    for name, scorer in candidates.items():
        rows = []
        for group, fname, label, audio in items:
            rows.append((group, fname, label, clip_score(scorer, audio)))
        scores = np.array([r[3] for r in rows])
        labels = np.array([r[2] for r in rows])
        pos, neg = scores[labels == 1], scores[labels == 0]
        fpr = float((neg >= 0.5).mean())
        fnr = float((pos < 0.5).mean())
        results[name] = {
            "auc": auc(pos, neg),
            "fpr@0.5": fpr,
            "fnr@0.5": fnr,
            "genuine_mean": float(neg.mean()),
            "fake_mean": float(pos.mean()),
            "per_clip": [
                {"group": g, "file": f, "label": lbl, "score": round(s, 4)}
                for g, f, lbl, s in rows
            ],
        }
        print(f"{name:16s} AUC={results[name]['auc']:.3f}  "
              f"FPR@0.5={fpr:.2f}  FNR@0.5={fnr:.2f}  "
              f"genuine μ={neg.mean():.2f}  fake μ={pos.mean():.2f}")

    # per-group breakdown for the top models
    print("\nPer-group mean scores:")
    groups = sorted({g for g, *_ in items})
    hdr = "model".ljust(16) + "".join(g.rjust(15) for g in groups)
    print(hdr)
    for name, res in results.items():
        by = {g: [] for g in groups}
        for row in res["per_clip"]:
            by[row["group"]].append(row["score"])
        print(name.ljust(16) + "".join(
            (f"{np.mean(v):.2f}" if v else "—").rjust(15) for v in
            (by[g] for g in groups)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()

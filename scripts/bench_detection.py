"""
Detection-timing + latency benchmark for the shipped pipeline.

Runs the full ensemble (cascade mode, as served live) over every clip in
the eval set and reports the numbers the problem statement asks for:

  - time-to-AMBER / time-to-RED distributions on fake clips
  - % of fakes flagged within 5 s and 10 s (vs the 10 s requirement)
  - false-alert rate on genuine clips (any AMBER / any RED chunk)
  - per-chunk processing latency percentiles

Usage:
    python scripts/bench_detection.py --set <dir> [--json out.json]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voiceshield import config  # noqa: E402

SR = config.SAMPLE_RATE


def run_clip(ensemble, path):
    from voiceshield.audio.source import FileSource
    from voiceshield.pipeline.runner import PipelineRunner

    runner = PipelineRunner(FileSource(path), ensemble=ensemble, cascade=True)
    latencies, states = [], []
    while True:
        t0 = time.perf_counter()
        try:
            entry = runner.run_once()
        except EOFError:
            break
        latencies.append((time.perf_counter() - t0) * 1000)
        states.append(entry.state)
    eng = runner.state_engine
    return eng.first_amber_t, eng.first_red_t, states, latencies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from voiceshield.classifier import get_scorers

    ensemble = get_scorers()
    print("ensemble:", ", ".join(ensemble))

    clips = []
    for group in sorted(os.listdir(args.set)):
        gdir = os.path.join(args.set, group)
        if os.path.isdir(gdir):
            label = 0 if "genuine" in group else 1
            clips += [(group, p, label) for p in sorted(glob.glob(os.path.join(gdir, "*.wav")))]

    fake_rows, genuine_rows, all_lat = [], [], []
    for i, (group, path, label) in enumerate(clips):
        dur = len(sf.read(path, dtype="float32")[0]) / SR
        fa, fr, states, lat = run_clip(ensemble, path)
        all_lat += lat[1:]  # drop per-clip first-chunk warmup
        row = {"group": group, "file": os.path.basename(path), "dur": round(dur, 1),
               "first_amber_t": fa, "first_red_t": fr,
               "any_amber": any(s in ("amber", "red") for s in states),
               "any_red": "red" in states}
        (fake_rows if label else genuine_rows).append(row)
        if i % 50 == 0:
            print(f"{i}/{len(clips)}", flush=True)

    lat = np.array(all_lat)
    print("\n=== Latency (per 500 ms chunk, full cascade pipeline) ===")
    print(f"p50={np.percentile(lat, 50):.0f} ms  p95={np.percentile(lat, 95):.0f} ms  "
          f"p99={np.percentile(lat, 99):.0f} ms  (budget: 200 ms)")

    print(f"\n=== Genuine clips (n={len(genuine_rows)}) ===")
    fa_rate = np.mean([r["any_amber"] for r in genuine_rows])
    fr_rate = np.mean([r["any_red"] for r in genuine_rows])
    print(f"any-AMBER rate: {fa_rate:.1%}   any-RED rate: {fr_rate:.1%}")

    print(f"\n=== Fake clips (n={len(fake_rows)}) ===")
    det = [r for r in fake_rows if r["first_red_t"] is not None]
    amber = [r for r in fake_rows if r["first_amber_t"] is not None]
    print(f"reached AMBER: {len(amber)/len(fake_rows):.1%}   reached RED: {len(det)/len(fake_rows):.1%}")
    if amber:
        t = np.array([r["first_amber_t"] for r in amber])
        print(f"time-to-AMBER: median={np.median(t):.1f}s  p90={np.percentile(t, 90):.1f}s")
    if det:
        t = np.array([r["first_red_t"] for r in det])
        print(f"time-to-RED:   median={np.median(t):.1f}s  p90={np.percentile(t, 90):.1f}s")
    within5 = np.mean([r["first_amber_t"] is not None and r["first_amber_t"] <= 5.0
                       for r in fake_rows])
    within10 = np.mean([r["first_red_t"] is not None and r["first_red_t"] <= 10.0
                        for r in fake_rows])
    print(f"flagged AMBER ≤5 s: {within5:.1%}   RED ≤10 s: {within10:.1%}")
    # many benchmark clips are shorter than 10 s — report the fair number too
    long_fakes = [r for r in fake_rows if r["dur"] >= 4.0]
    w10_long = np.mean([r["first_red_t"] is not None and r["first_red_t"] <= 10.0
                        for r in long_fakes])
    print(f"RED ≤10 s among fakes ≥4 s (n={len(long_fakes)}): {w10_long:.1%}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"latency_ms": {"p50": float(np.percentile(lat, 50)),
                                      "p95": float(np.percentile(lat, 95))},
                       "genuine": genuine_rows, "fake": fake_rows}, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()

"""
Evidence package export (Stage 2).

Privacy contract (G10): nothing here runs automatically. An evidence
package is produced only on an explicit save-for-audit request, and it
captures exactly what the rolling buffer holds at that moment — the last
10 seconds, nothing more.

A package directory contains:
    spectrogram.png     mel spectrogram of the buffered audio
    phase_heatmap.png   per-(freq, frame) phase discontinuity map
    audio_sha256.txt    SHA-256 of the raw float32 buffer bytes
    evidence.json       envelope: scores, timeline, config snapshot, hash
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import librosa
import numpy as np

from voiceshield import config

# 256-step blue→yellow perceptual ramp (viridis endpoints, linear blend) —
# enough for forensic legibility without a matplotlib dependency.
_C0 = np.array([68, 1, 84], dtype=np.float64)
_C1 = np.array([253, 231, 37], dtype=np.float64)


def _colormap(norm: np.ndarray) -> np.ndarray:
    """(H, W) in [0,1] → (H, W, 3) uint8."""
    ramp = norm[..., None]
    return (_C0 * (1.0 - ramp) + _C1 * ramp).astype(np.uint8)


def _to_png(matrix: np.ndarray, path: str, min_width: int = 640) -> None:
    """Render a 2D feature matrix (freq × frames, low freq at row 0) to PNG."""
    from PIL import Image

    m = np.asarray(matrix, dtype=np.float64)
    lo, hi = float(np.min(m)), float(np.max(m))
    norm = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    img = Image.fromarray(_colormap(np.flipud(norm)))  # low frequencies at the bottom
    if img.width < min_width:
        scale = int(np.ceil(min_width / img.width))
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    img.save(path)


def _phase_discontinuity_map(audio: np.ndarray) -> np.ndarray:
    """Per-(freq, frame) phase discontinuity in [0, 1] (2D sibling of
    features.phase.compute_phase_discontinuity, which returns the mean)."""
    stft = librosa.stft(audio, n_fft=512, hop_length=160)
    unwrapped = np.unwrap(np.angle(stft), axis=1)
    d2 = np.abs(np.diff(unwrapped, n=2, axis=1))
    return np.clip(d2 / (2 * np.pi), 0.0, 1.0)


def export_evidence(
    audio: np.ndarray,
    timeline: list[dict[str, Any]],
    component_scores: dict[str, float] | None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """
    Write one evidence package and return its manifest (also saved as
    evidence.json). `audio` is the rolling-buffer contents; `timeline` the
    serialized TimelineEntry list; `component_scores` the latest fusion
    inputs.
    """
    from voiceshield.features.spectrogram import compute_mel

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    package_dir = os.path.join(out_dir or config.EVIDENCE_DIR, f"evidence_{stamp}")
    os.makedirs(package_dir, exist_ok=True)

    audio = np.asarray(audio, dtype=np.float32)
    sha256 = hashlib.sha256(audio.tobytes()).hexdigest()

    spectrogram_path = os.path.join(package_dir, "spectrogram.png")
    phase_path = os.path.join(package_dir, "phase_heatmap.png")
    _to_png(compute_mel(audio), spectrogram_path)
    _to_png(_phase_discontinuity_map(audio), phase_path)

    with open(os.path.join(package_dir, "audio_sha256.txt"), "w") as f:
        f.write(sha256 + "\n")

    manifest = {
        "created_utc": stamp,
        "audio_sha256": sha256,
        "audio_seconds": round(len(audio) / config.SAMPLE_RATE, 3),
        "sample_rate": config.SAMPLE_RATE,
        "component_scores": component_scores or {},
        "fusion_weights": config.FUSION_WEIGHTS,
        "timeline": timeline,
        "files": {
            "spectrogram": os.path.basename(spectrogram_path),
            "phase_heatmap": os.path.basename(phase_path),
        },
    }
    with open(os.path.join(package_dir, "evidence.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    manifest["package_dir"] = package_dir
    return manifest

"""Evidence package export (Stage 2, privacy-gated)."""

import hashlib
import json
import os

import numpy as np

from voiceshield import config
from voiceshield.evidence import export_evidence

SR = config.SAMPLE_RATE


def _audio() -> np.ndarray:
    t = np.arange(4 * SR) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_export_writes_full_package(tmp_path):
    audio = _audio()
    timeline = [{"time": 0.5, "score": 0.4, "state": "amber"}]
    scores = {"stage1": 0.4, "ssl": 0.3}

    manifest = export_evidence(audio, timeline, scores, out_dir=str(tmp_path))
    package = manifest["package_dir"]

    assert os.path.isfile(os.path.join(package, "spectrogram.png"))
    assert os.path.isfile(os.path.join(package, "phase_heatmap.png"))
    assert os.path.isfile(os.path.join(package, "audio_sha256.txt"))
    assert os.path.isfile(os.path.join(package, "evidence.json"))

    # PNG magic bytes
    with open(os.path.join(package, "spectrogram.png"), "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_sha256_matches_buffer_bytes(tmp_path):
    audio = _audio()
    manifest = export_evidence(audio, [], None, out_dir=str(tmp_path))
    expected = hashlib.sha256(audio.tobytes()).hexdigest()
    assert manifest["audio_sha256"] == expected

    with open(os.path.join(manifest["package_dir"], "audio_sha256.txt")) as f:
        assert f.read().strip() == expected


def test_manifest_round_trips_and_carries_scores(tmp_path):
    scores = {"stage1": 0.9, "replay": 0.2}
    manifest = export_evidence(_audio(), [{"time": 0.5}], scores, out_dir=str(tmp_path))

    with open(os.path.join(manifest["package_dir"], "evidence.json")) as f:
        on_disk = json.load(f)

    assert on_disk["component_scores"] == scores
    assert on_disk["fusion_weights"] == config.FUSION_WEIGHTS
    assert on_disk["audio_seconds"] == 4.0
    assert on_disk["timeline"] == [{"time": 0.5}]
    # package_dir is runtime info, not part of the stored envelope
    assert "package_dir" not in on_disk

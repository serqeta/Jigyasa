"""
Real pretrained ensemble members (Stage 2). Skipped when the HF cache is
absent so CI without the ~2 GB of weights stays green; run
`python scripts/download_model.py` to enable.
"""

import os

import numpy as np
import pytest

import voiceshield.config as cfg

SR = cfg.SAMPLE_RATE


def _cached(model_id: str) -> bool:
    leaf = "models--" + model_id.replace("/", "--")
    return os.path.isdir(os.path.join(cfg.HF_CACHE_DIR, leaf))


def _synthetic_window() -> np.ndarray:
    t = np.arange(4 * SR) / SR
    return (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


@pytest.mark.parametrize("name", list(cfg.HF_SCORERS))
def test_pretrained_scorer_loads_and_scores(name):
    spec = cfg.HF_SCORERS[name]
    if not _cached(spec["model_id"]):
        pytest.skip(f"{spec['model_id']} not in {cfg.HF_CACHE_DIR}")
    # disabled models (e.g. 'spec', see config) must still load mechanically

    from voiceshield.classifier.hf_scorer import HFScorer

    scorer = HFScorer(spec["model_id"], spec["spoof_label"])
    score = scorer.score(_synthetic_window())
    assert 0.0 <= score <= 1.0

    # empty input contract shared by all Scorer implementations
    assert scorer.score(np.array([], dtype=np.float32)) == 0.0


def test_get_scorers_returns_full_ensemble():
    if not all(_cached(s["model_id"]) for s in cfg.HF_SCORERS.values()):
        pytest.skip("Stage 2 weights not fully cached")

    from voiceshield.classifier import get_scorers

    scorers = get_scorers()
    enabled = {k for k, s in cfg.HF_SCORERS.items() if s.get("enabled", True)}
    assert set(scorers) >= {"nii", "phase_pitch", *enabled}
    assert "stage1" not in scorers, "AASIST was retired from the ensemble"
    disabled = set(cfg.HF_SCORERS) - enabled
    assert not disabled & set(scorers), "disabled models must not join the ensemble"

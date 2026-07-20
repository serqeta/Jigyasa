"""Stage 2 risk fusion (pipeline/state_engine.fuse_scores, compute_final)."""

import pytest

from voiceshield.classifier.protocol import RiskState
from voiceshield.features import GateState
from voiceshield.pipeline.state_engine import compute_final, fuse_scores


def test_single_component_is_identity():
    # zero-weighted single component → plain-mean fallback keeps identity
    # (Stage 1 single-scorer compatibility path)
    assert fuse_scores({"stage1": 0.62}) == pytest.approx(0.62)


def test_weighted_mean_dominates_when_above_floors():
    # nii 0.50, ssl 0.15 → (0.5*0.8 + 0.15*0.2) / 0.65 = 0.6615…
    # peak floors (0.8*0.8=0.64, 0.8*0.2=0.16) sit below the mean.
    fused = fuse_scores({"nii": 0.8, "ssl": 0.2})
    assert fused == pytest.approx((0.5 * 0.8 + 0.15 * 0.2) / 0.65)


def test_peak_evidence_floor():
    # A confident validated detector must not be averaged away: fused is
    # floored at each peak component's factor × its score.
    fused = fuse_scores({"nii": 0.0, "ssl": 0.95, "phase_pitch": 0.0, "replay": 0.0})
    assert fused == pytest.approx(0.8 * 0.95)
    # nii floor: fake-median 0.997 → RED-range, but never the 0.85
    # instant short-circuit on its own
    fused = fuse_scores({"nii": 0.997, "ssl": 0.0, "wavlm": 0.0})
    assert fused == pytest.approx(0.8 * 0.997)
    assert fused < 0.85
    # wavlm solo ceiling stays AMBER (1% of genuine speakers hit 0.999)
    fused = fuse_scores({"nii": 0.0, "ssl": 0.0, "wavlm": 1.0})
    assert fused == pytest.approx(0.6)
    # retired components get no floor and no weight, however confident
    fused = fuse_scores({"nii": 0.05, "stage1": 0.99, "spec": 0.99})
    assert fused < 0.10


def test_missing_components_renormalize():
    # A missing model must not dilute: all-ones stays 1.0 for any subset.
    assert fuse_scores({"stage1": 1.0, "phase_pitch": 1.0}) == pytest.approx(1.0)
    assert fuse_scores({"stage1": 1.0, "ssl": 1.0, "spec": 1.0, "wavlm": 1.0,
                        "phase_pitch": 1.0, "replay": 1.0}) == pytest.approx(1.0)


def test_unknown_component_gets_default_weight():
    # Score-source-agnostic: an unconfigured key still participates.
    fused = fuse_scores({"stage3_future": 0.8})
    assert fused == pytest.approx(0.8)


def test_empty_components():
    assert fuse_scores({}) == 0.0


def test_result_clamped_to_unit_interval():
    assert 0.0 <= fuse_scores({"stage1": 1.0, "ssl": 1.0}) <= 1.0


def test_compute_final_grey_overrides_everything():
    assert compute_final({"stage1": 0.99, "ssl": 0.99}, GateState.GREY) is RiskState.GREY


def test_compute_final_maps_fused_score():
    assert compute_final({"stage1": 0.1}, GateState.NORMAL) is RiskState.GREEN
    assert compute_final({"stage1": 0.5}, GateState.NORMAL) is RiskState.AMBER
    assert compute_final({"stage1": 0.9}, GateState.NORMAL) is RiskState.RED


def test_compute_final_no_components_green():
    assert compute_final({}, GateState.NORMAL) is RiskState.GREEN

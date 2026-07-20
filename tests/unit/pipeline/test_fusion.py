"""Stage 2 risk fusion (pipeline/state_engine.fuse_scores, compute_final)."""

import pytest

from voiceshield.classifier.protocol import RiskState
from voiceshield.features import GateState
from voiceshield.pipeline.state_engine import compute_final, fuse_scores


def test_single_component_is_identity():
    assert fuse_scores({"stage1": 0.62}) == pytest.approx(0.62)


def test_weighted_mean_two_components():
    # stage1: 0.10, ssl: 0.40 → (0.10*1.0 + 0.40*0.0) / 0.50; ssl peak is 0
    fused = fuse_scores({"stage1": 1.0, "ssl": 0.0})
    assert fused == pytest.approx(0.10 / 0.50)


def test_peak_evidence_floor():
    # A confident validated detector must not be averaged away: fused is
    # floored at each peak component's factor × its score.
    fused = fuse_scores({"stage1": 0.0, "ssl": 0.95, "phase_pitch": 0.0, "replay": 0.0})
    assert fused == pytest.approx(0.8 * 0.95)
    # wavlm gets a softer floor than ssl (demo calibration: 0.75 —
    # RED-capable via hysteresis but never the 0.85 instant short-circuit)
    fused = fuse_scores({"stage1": 0.0, "ssl": 0.0, "wavlm": 1.0, "phase_pitch": 0.0,
                         "replay": 0.0})
    assert fused == pytest.approx(0.75 * 1.0)
    assert fused < 0.85  # instant-RED requires more than one model's word
    # ...but non-peak components get no floor, however confident.
    fused = fuse_scores({"stage1": 0.95, "phase_pitch": 0.0, "replay": 0.0})
    assert fused < 0.5


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

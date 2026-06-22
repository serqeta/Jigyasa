"""TEST-C5.5: Score → state boundary tests."""

from voiceshield.classifier.protocol import RiskState
from voiceshield.classifier.state_mapper import score_to_state


def test_green_below_amber():
    assert score_to_state(0.0) is RiskState.GREEN
    assert score_to_state(0.29) is RiskState.GREEN


def test_amber_at_boundary():
    assert score_to_state(0.30) is RiskState.AMBER


def test_amber_range():
    assert score_to_state(0.50) is RiskState.AMBER
    assert score_to_state(0.69) is RiskState.AMBER


def test_red_at_boundary():
    assert score_to_state(0.70) is RiskState.RED


def test_red_above():
    assert score_to_state(0.85) is RiskState.RED
    assert score_to_state(1.0) is RiskState.RED

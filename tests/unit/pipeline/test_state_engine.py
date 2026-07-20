"""TEST-R7.1, R7.2, R7.3, R7.4: State engine rules."""

from voiceshield.classifier.protocol import RiskState
from voiceshield.features import GateState
from voiceshield.pipeline.state_engine import StateEngine


def test_grey_override():
    """TEST-R7.1: GREY gate forces final state to GREY regardless of score."""
    eng = StateEngine()
    state = eng.update(score=0.99, gate=GateState.GREY, t_end=0.5)
    assert state is RiskState.GREY


def test_grey_does_not_latch_amber():
    eng = StateEngine()
    eng.update(score=0.99, gate=GateState.GREY, t_end=0.5)
    assert eng.first_amber_t is None
    assert eng.first_red_t is None


def test_hysteresis_single_chunk_no_escalation():
    """TEST-R7.2: Single chunk above AMBER threshold does not escalate."""
    eng = StateEngine()
    eng.update(0.10, GateState.NORMAL, 0.5)  # green
    state = eng.update(0.50, GateState.NORMAL, 1.0)  # one chunk amber
    # Should still be GREEN (needs 2 consecutive)
    assert state is RiskState.GREEN


def test_hysteresis_two_chunks_escalates():
    eng = StateEngine()
    eng.update(0.40, GateState.NORMAL, 0.5)
    state = eng.update(0.40, GateState.NORMAL, 1.0)
    assert state is RiskState.AMBER


def test_first_red_t_latched():
    """TEST-R7.3: first_red_t set on first RED chunk, not overwritten."""
    eng = StateEngine()
    for i in range(2):
        eng.update(0.80, GateState.NORMAL, float(i + 1) * 0.5)
    first = eng.first_red_t
    assert first is not None
    eng.update(0.80, GateState.NORMAL, 2.0)
    assert eng.first_red_t == first  # latched


def test_single_chunk_override_at_085():
    """TEST-R7.4: Score ≥ 0.85 in one chunk → immediate RED."""
    eng = StateEngine()
    state = eng.update(0.90, GateState.NORMAL, 0.5)
    assert state is RiskState.RED
    assert eng.first_red_t == 0.5


def test_no_de_escalation():
    """Once RED, stays RED even when score drops."""
    eng = StateEngine()
    eng.update(0.90, GateState.NORMAL, 0.5)  # immediate RED
    state = eng.update(0.10, GateState.NORMAL, 1.0)
    assert state is RiskState.RED


def test_first_amber_latched_before_red():
    eng = StateEngine()
    # Two consecutive AMBER-range chunks → AMBER
    eng.update(0.40, GateState.NORMAL, 0.5)
    eng.update(0.40, GateState.NORMAL, 1.0)
    assert eng.first_amber_t == 1.0
    assert eng.first_red_t is None


def test_grey_wins_even_after_amber():
    """Silence after AMBER must display GREY, not hold AMBER (spec T7.1)."""
    eng = StateEngine()
    eng.update(0.50, GateState.NORMAL, 0.5)
    assert eng.update(0.50, GateState.NORMAL, 1.0) is RiskState.AMBER
    assert eng.update(0.00, GateState.GREY, 1.5) is RiskState.GREY
    assert eng.update(0.00, GateState.GREY, 2.0) is RiskState.GREY


def test_grey_preserves_escalation_for_resume():
    """Suspicion survives a silent stretch: speech resuming suspicious
    continues from AMBER (with hysteresis), and first_amber_t stays latched."""
    eng = StateEngine()
    eng.update(0.50, GateState.NORMAL, 0.5)
    eng.update(0.50, GateState.NORMAL, 1.0)  # AMBER
    eng.update(0.00, GateState.GREY, 1.5)  # silence → GREY
    first_amber = eng.first_amber_t
    eng.update(0.75, GateState.NORMAL, 2.0)
    state = eng.update(0.75, GateState.NORMAL, 2.5)  # 2 red-scoring chunks
    assert state is RiskState.RED  # resumed from AMBER
    assert eng.first_amber_t == first_amber


def test_grey_never_escalates():
    """No chunk processed under a GREY gate may raise the risk state."""
    eng = StateEngine()
    eng.update(0.50, GateState.NORMAL, 0.5)
    eng.update(0.50, GateState.NORMAL, 1.0)  # AMBER
    for i in range(6):
        state = eng.update(0.99, GateState.GREY, 1.5 + i * 0.5)
        assert state is RiskState.GREY
    assert eng.first_red_t is None

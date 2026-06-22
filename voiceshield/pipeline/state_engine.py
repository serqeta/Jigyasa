from voiceshield.classifier.protocol import RiskState
from voiceshield.classifier.state_mapper import score_to_state
from voiceshield.features import GateState

# Single-chunk score that bypasses hysteresis and forces immediate RED
_RED_OVERRIDE_THRESHOLD = 0.85
# Consecutive chunks required to escalate
_HYSTERESIS_CHUNKS = 2


class StateEngine:
    """
    Tracks risk state across chunks with hysteresis, Grey override, and
    first-alert-time latching.
    """

    def __init__(self) -> None:
        self.first_amber_t: float | None = None
        self.first_red_t: float | None = None
        self._consec_amber = 0
        self._consec_red = 0
        self._current: RiskState = RiskState.GREEN

    def update(self, score: float, gate: GateState, t_end: float) -> RiskState:
        # Rule 1: GREY gate overrides score, but respects latched escalated states
        if gate is GateState.GREY:
            self._consec_amber = 0
            self._consec_red = 0
            if self._current in (RiskState.AMBER, RiskState.RED):
                return self._current
            return RiskState.GREY

        # Rule 2: single-chunk short-circuit to RED at very high confidence
        if score >= _RED_OVERRIDE_THRESHOLD:
            self._consec_amber = _HYSTERESIS_CHUNKS
            self._consec_red = _HYSTERESIS_CHUNKS
            self._latch(RiskState.RED, t_end)
            self._current = RiskState.RED
            return RiskState.RED

        # Rule 3: hysteresis counters
        raw = score_to_state(score)

        if raw in (RiskState.AMBER, RiskState.RED):
            self._consec_amber += 1
        else:
            self._consec_amber = 0

        if raw is RiskState.RED:
            self._consec_red += 1
        else:
            self._consec_red = 0

        # Effective state — never de-escalate once reached
        if self._consec_red >= _HYSTERESIS_CHUNKS:
            final = RiskState.RED
        elif self._consec_amber >= _HYSTERESIS_CHUNKS:
            final = RiskState.AMBER
        else:
            if self._current is RiskState.RED:
                final = RiskState.RED
            elif self._current is RiskState.AMBER:
                final = RiskState.AMBER
            else:
                final = RiskState.GREEN

        self._latch(final, t_end)
        self._current = final
        return final

    def _latch(self, state: RiskState, t: float) -> None:
        if state in (RiskState.AMBER, RiskState.RED) and self.first_amber_t is None:
            self.first_amber_t = t
        if state is RiskState.RED and self.first_red_t is None:
            self.first_red_t = t


def compute_final(
    component_scores: dict[str, float],
    gate: GateState,
) -> RiskState:
    """Stage 2 hand-off seam. Stage 1 calls with {"stage1": score}."""
    if gate is GateState.GREY:
        return RiskState.GREY
    if not component_scores:
        return RiskState.GREEN
    return score_to_state(max(component_scores.values()))

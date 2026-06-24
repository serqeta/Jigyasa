from voiceshield.classifier.protocol import RiskState
from voiceshield.classifier.state_mapper import score_to_state
from voiceshield.features import GateState

# Score threshold to bypass hysteresis and jump straight to RED
_RED_OVERRIDE_THRESHOLD = 0.85
# Consecutive suspicious chunks needed to escalate GREEN → AMBER / AMBER → RED
_ESCALATE_CHUNKS = 2
# Consecutive clean (green-scoring) chunks needed to de-escalate AMBER → GREEN.
# RED never de-escalates automatically — it requires an explicit reset.
# 4 chunks = 2 seconds of sustained clean audio, which is enough to clear a
# brief mic noise spike without masking a real attack.
_DEESCALATE_AMBER_CHUNKS = 4


class StateEngine:
    """
    Tracks risk state across chunks with hysteresis, grey override, and
    first-alert-time latching.

    Escalation: GREEN → AMBER after _ESCALATE_CHUNKS suspicious chunks.
                AMBER → RED  after _ESCALATE_CHUNKS suspicious chunks.
                RED    instant at score >= _RED_OVERRIDE_THRESHOLD.

    De-escalation: AMBER → GREEN after _DEESCALATE_AMBER_CHUNKS consecutive
                   clean chunks. RED never de-escalates (requires reset).
    """

    def __init__(self) -> None:
        self.first_amber_t: float | None = None
        self.first_red_t: float | None = None
        self._consec_suspicious = 0  # amber-or-red streak
        self._consec_red = 0
        self._consec_clean = 0       # green streak (for de-escalation)
        self._current: RiskState = RiskState.GREEN

    def update(self, score: float, gate: GateState, t_end: float) -> RiskState:
        # Grey gate: no speech or poor SNR — pause scoring, hold current state
        # if already escalated, reset counters but don't de-escalate.
        if gate is GateState.GREY:
            self._consec_suspicious = 0
            self._consec_red = 0
            self._consec_clean = 0
            if self._current in (RiskState.AMBER, RiskState.RED):
                return self._current
            return RiskState.GREY

        # Instant RED at very high single-chunk confidence
        if score >= _RED_OVERRIDE_THRESHOLD:
            self._consec_suspicious = _ESCALATE_CHUNKS
            self._consec_red = _ESCALATE_CHUNKS
            self._consec_clean = 0
            self._latch(RiskState.RED, t_end)
            self._current = RiskState.RED
            return RiskState.RED

        raw = score_to_state(score)
        is_suspicious = raw in (RiskState.AMBER, RiskState.RED)

        if is_suspicious:
            self._consec_suspicious += 1
            self._consec_red = self._consec_red + 1 if raw is RiskState.RED else 0
            self._consec_clean = 0
        else:
            self._consec_suspicious = 0
            self._consec_red = 0
            self._consec_clean += 1

        # --- Escalation ---
        if self._consec_red >= _ESCALATE_CHUNKS:
            final = RiskState.RED
        elif self._consec_suspicious >= _ESCALATE_CHUNKS:
            final = RiskState.AMBER if self._current is RiskState.GREEN else RiskState.RED \
                if self._current is RiskState.RED else RiskState.AMBER
        # --- De-escalation ---
        elif self._current is RiskState.RED:
            # RED never auto-de-escalates
            final = RiskState.RED
        elif self._current is RiskState.AMBER:
            # AMBER → GREEN after sustained clean audio
            final = RiskState.GREEN if self._consec_clean >= _DEESCALATE_AMBER_CHUNKS else RiskState.AMBER
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

from voiceshield import config
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
        self._consec_clean = 0  # green streak (for de-escalation)
        self._current: RiskState = RiskState.GREEN

    def update(self, score: float, gate: GateState, t_end: float) -> RiskState:
        # Grey gate: no speech or poor SNR — GREY wins regardless of score
        # or prior escalation (spec T7.1). Internal state and first-alert
        # latches are preserved, so when speech resumes hysteresis continues
        # from where it left off (an earlier AMBER isn't forgotten), but the
        # display never claims risk while there is nothing to analyze.
        if gate is GateState.GREY:
            self._consec_suspicious = 0
            self._consec_red = 0
            self._consec_clean = 0
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
            final = (
                RiskState.AMBER
                if self._current is RiskState.GREEN
                else RiskState.RED
                if self._current is RiskState.RED
                else RiskState.AMBER
            )
        # --- De-escalation ---
        elif self._current is RiskState.RED:
            # RED never auto-de-escalates
            final = RiskState.RED
        elif self._current is RiskState.AMBER:
            # AMBER → GREEN after sustained clean audio
            final = (
                RiskState.GREEN
                if self._consec_clean >= _DEESCALATE_AMBER_CHUNKS
                else RiskState.AMBER
            )
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
    """Map fused component scores to a risk state. GREY gate wins outright."""
    if gate is GateState.GREY:
        return RiskState.GREY
    if not component_scores:
        return RiskState.GREEN
    return score_to_state(fuse_scores(component_scores))


def fuse_scores(component_scores: dict[str, float]) -> float:
    """
    Stage 2 risk fusion: weighted mean over the components present, with
    weights from config.FUSION_WEIGHTS renormalized so missing components
    never dilute the result. Components without a configured weight share
    the mean of the configured weights, keeping the fusion
    score-source-agnostic per the Stage-2 hand-off contract.

    The result is floored by the peak-evidence rule: when one of the
    validated high-precision detectors (config.PEAK_COMPONENTS) is highly
    confident, averaging must not wash that evidence out.
    """
    if not component_scores:
        return 0.0

    configured = config.FUSION_WEIGHTS
    default_w = sum(configured.values()) / max(len(configured), 1)
    weights = {k: configured.get(k, default_w) for k in component_scores}
    total = sum(weights.values())
    if total <= 0.0:
        # every present component is zero-weighted — the Stage 1
        # single-scorer compatibility path: plain mean keeps score=score
        fused = sum(component_scores.values()) / len(component_scores)
    else:
        fused = sum(weights[k] * component_scores[k] for k in component_scores) / total

    peak = max(
        (
            factor * component_scores[k]
            for k, factor in config.PEAK_COMPONENTS.items()
            if k in component_scores
        ),
        default=0.0,
    )
    fused = max(fused, peak)
    return float(min(max(fused, 0.0), 1.0))

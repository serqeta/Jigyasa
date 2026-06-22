from voiceshield import config
from voiceshield.classifier.protocol import RiskState


def score_to_state(score: float) -> RiskState:
    """Map a [0, 1] risk score to a RiskState. Boundaries per arch §8."""
    if score >= config.SCORE_RED:
        return RiskState.RED
    elif score >= config.SCORE_AMBER:
        return RiskState.AMBER
    else:
        return RiskState.GREEN

import os

from voiceshield import config
from voiceshield.classifier.protocol import RiskState, Scorer

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "aasist_l.pth")


def get_scorer() -> Scorer:
    """
    Factory: returns AASISTScorer if weights exist and USE_FALLBACK_CLASSIFIER
    is False, otherwise returns FallbackScorer.
    """
    from voiceshield.logger import get_logger

    log = get_logger("voiceshield.classifier")

    if not config.USE_FALLBACK_CLASSIFIER:
        try:
            from voiceshield.classifier.aasist_inference import AASISTScorer
            from voiceshield.classifier.aasist_loader import load_aasist

            model = load_aasist(_MODEL_PATH)
            log.info("AASIST-L loaded from %s", _MODEL_PATH)
            return AASISTScorer(model)
        except FileNotFoundError as e:
            log.warning("AASIST weights missing (%s); falling back to rule-based scorer.", e)
        except Exception as e:
            log.warning("AASIST load failed (%s); falling back to rule-based scorer.", e)

    from voiceshield.classifier.fallback import FallbackScorer

    log.info("Using FallbackScorer (rule-based).")
    return FallbackScorer()


__all__ = ["get_scorer", "RiskState", "Scorer"]

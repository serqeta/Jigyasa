import os
import threading

from voiceshield import config
from voiceshield.classifier.protocol import RiskState, Scorer

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "aasist_l.pth")

# get_scorers() singleton state — see its docstring.
_ENSEMBLE: dict[str, Scorer] | None = None
_ENSEMBLE_LOCK = threading.Lock()


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


def get_scorers() -> dict[str, Scorer]:
    """
    Stage 2 factory: name → scorer for every available ensemble component.

    Always contains "stage1" (AASIST or fallback). Pretrained HF components
    and the rule-based phase/pitch scorer join when enabled and loadable;
    a component that fails to load is logged and excluded — fusion weights
    renormalize over whatever this returns.

    Process-wide singleton: the models are GPU-resident, and loading a
    second copy (live runner + /v2/analyze both calling this) exhausts
    a 6 GB card and poisons the CUDA context for the whole process.
    """
    global _ENSEMBLE
    with _ENSEMBLE_LOCK:
        if _ENSEMBLE is not None:
            return _ENSEMBLE

        from voiceshield.logger import get_logger

        log = get_logger("voiceshield.classifier")
        scorers: dict[str, Scorer] = {"stage1": get_scorer()}

        for name, spec in config.HF_SCORERS.items():
            if not spec.get("enabled", True):
                continue
            try:
                from voiceshield.classifier.hf_scorer import HFScorer

                scorers[name] = HFScorer(spec["model_id"], spec["spoof_label"])
                log.info("Ensemble scorer '%s' loaded (%s).", name, spec["model_id"])
            except Exception as e:
                log.warning(
                    "Ensemble scorer '%s' unavailable (%s); excluded from fusion.", name, e
                )

        if config.ENABLE_PHASE_PITCH_SCORER:
            from voiceshield.classifier.phase_pitch import PhasePitchScorer

            scorers["phase_pitch"] = PhasePitchScorer()

        _ENSEMBLE = scorers
        return scorers


__all__ = ["get_scorer", "get_scorers", "RiskState", "Scorer"]

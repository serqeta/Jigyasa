import numpy as np
import torch

# AASIST-L was trained on exactly 64600 samples (≈4.04 s @ 16 kHz)
_NB_SAMP = 64600


class AASISTScorer:
    """Scorer backed by a pretrained AASIST-L model."""

    def __init__(self, model: "torch.nn.Module"):
        self._model = model

    def _preprocess(self, audio: np.ndarray) -> torch.Tensor:
        n = len(audio)
        if n >= _NB_SAMP:
            audio = audio[-_NB_SAMP:]
        else:
            audio = np.pad(audio, (0, _NB_SAMP - n))
        return torch.from_numpy(audio).float().unsqueeze(0)  # (1, NB_SAMP)

    def score(self, audio: np.ndarray) -> float:
        x = self._preprocess(audio)
        with torch.no_grad():
            # Model returns (last_hidden, logits); logits shape: (1, 2) = [bonafide, spoof]
            _, logits = self._model(x)
        probs = torch.softmax(logits, dim=-1)
        spoof_prob = float(probs[0, 1].item())
        return float(np.clip(spoof_prob, 0.0, 1.0))

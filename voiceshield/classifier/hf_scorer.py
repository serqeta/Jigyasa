"""
Generic scorer over pretrained Hugging Face audio-classification
anti-spoofing checkpoints (wav2vec2/XLS-R, WavLM, AST, ...).

Each hub model has its own label polarity (which class index means
"spoof"), so the index is pinned per model in config.HF_SCORERS and the
score is normalized here to a single convention: higher == more likely
synthetic.
"""

from __future__ import annotations

import numpy as np

from voiceshield import config


class HFScorer:
    """Scorer protocol implementation backed by AutoModelForAudioClassification."""

    def __init__(self, model_id: str, spoof_label: int) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.model_id = model_id
        self._spoof_label = spoof_label
        self._device = config.get_device()
        self._torch = torch
        def _load(local_only: bool):
            kwargs = {"cache_dir": config.HF_CACHE_DIR, "local_files_only": local_only}
            return (
                AutoFeatureExtractor.from_pretrained(model_id, **kwargs),
                AutoModelForAudioClassification.from_pretrained(model_id, **kwargs),
            )

        try:
            self._extractor, model = _load(local_only=True)
        except Exception:
            # not cached yet — fetch from the Hub once
            self._extractor, model = _load(local_only=False)
        model = model.to(self._device).eval()
        # fp16 on GPU: ~2x faster inference, and classification softmax is
        # insensitive to half-precision logits at our thresholds.
        self._half = self._device == "cuda"
        if self._half:
            model = model.half()
        self._model = model

    def score(self, audio: np.ndarray) -> float:
        if audio is None or len(audio) == 0:
            return 0.0
        torch = self._torch
        inputs = self._extractor(
            audio.astype(np.float32),
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="pt",
        ).to(self._device)
        if self._half:
            inputs = {
                k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()
            }
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[0, self._spoof_label].item())

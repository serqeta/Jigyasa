"""
Learned replay / physical-access scorer — LoRA-fine-tuned wav2vec2 (NII
backbone) trained on EchoFake with RawBoost channel augmentation
(scripts/finetune_replay.py).

Detects the loudspeaker-replay channel content-agnostically (higher =
more likely played through a speaker). Validated CROSS-channel: trained
only on EchoFake, held-out AUC 0.97 on our own unseen mic recordings.

SCOPE (honest): covers consumer-device WIDEBAND replay (the coverage of
EchoFake + augmentation). Weak on far-field; does NOT cover narrowband
telephony (replay cues sit at 6-8 kHz, above the phone passband). See
docs/REPLAY_FINDINGS.md. Loads only if models/replay_lora/ exists.
"""

from __future__ import annotations

import os

import numpy as np

from voiceshield import config
from voiceshield.classifier._infer_lock import run_on_gpu

_LORA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "replay_lora")
_NII_W = os.path.join(os.path.dirname(__file__), "..", "..", "models", "nii_mms300m_converted.pt")


class ReplayScorer:
    """Scorer protocol implementation; higher = more likely loudspeaker replay."""

    def __init__(self, lora_dir: str = _LORA_DIR) -> None:
        import torch
        from peft import PeftModel
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"{lora_dir} missing — run scripts/finetune_replay.py --save")
        self._torch = torch
        self._device = config.get_device()

        blob = torch.load(_NII_W, map_location="cpu", weights_only=False)
        cfg = Wav2Vec2Config(**dict(blob["config"]))
        base = Wav2Vec2Model(cfg)
        base.load_state_dict(blob["wav2vec2"], strict=True)
        ssl = PeftModel.from_pretrained(base, lora_dir)
        ssl = ssl.merge_and_unload()  # fold LoRA into weights → plain forward, faster
        self._ssl = ssl.to(self._device).eval()

        head = torch.nn.Linear(1024, 2)
        head.load_state_dict(torch.load(os.path.join(lora_dir, "head.pt"), map_location="cpu"))
        self._head = head.to(self._device).eval()

        self._half = self._device == "cuda"
        if self._half:
            self._ssl = self._ssl.half()
            self._head = self._head.half()

    def score(self, audio: np.ndarray) -> float:
        if audio is None or len(audio) == 0:
            return 0.0
        return run_on_gpu(self._score_gpu, audio)

    def _score_gpu(self, audio: np.ndarray) -> float:
        torch = self._torch
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
        wav = wav.unsqueeze(0).to(self._device)
        if self._half:
            wav = wav.half()
        with torch.no_grad():
            h = self._ssl(wav).last_hidden_state.mean(dim=1)
            logits = self._head(h)
            probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[0, 1].item())  # index 1 = replay

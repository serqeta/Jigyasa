"""
NII anti-deepfake scorer (nii-yamagishilab/mms-300m-anti-deepfake).

Evaluation winner (2026-07-20, 496-clip multi-source benchmark): AUC 0.997,
FPR@0.5 0.05, FNR@0.5 0.02 — near-perfect across GAN vocoders (WaveFake),
ASVspoof 2021 DF, In-the-Wild, and our generated clones.

Runs in-process on transformers: the original fairseq checkpoint is
converted by scripts/convert_nii.py into models/nii_mms300m_converted.pt
(validated to reproduce the reference implementation's scores).

LICENSE: CC BY-NC-SA 4.0 — research/hackathon use only; see NOTICE.md.
"""

from __future__ import annotations

import os

import numpy as np

from voiceshield import config

_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "models", "nii_mms300m_converted.pt")


class NIIScorer:
    """Scorer protocol implementation; higher = more likely synthetic."""

    def __init__(self, weights_path: str = _WEIGHTS) -> None:
        import torch
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"{weights_path} missing — run scripts/convert_nii.py first")
        self._torch = torch
        self._device = config.get_device()

        blob = torch.load(weights_path, map_location="cpu", weights_only=False)
        cfg = Wav2Vec2Config(**dict(blob["config"]))
        ssl = Wav2Vec2Model(cfg)
        ssl.load_state_dict(blob["wav2vec2"], strict=True)
        self._ssl = ssl.to(self._device).eval()  # type: ignore[arg-type]
        self._fc_w = blob["proj_fc"]["weight"].to(self._device)
        self._fc_b = blob["proj_fc"]["bias"].to(self._device)
        self._half = self._device == "cuda"
        if self._half:
            self._ssl = self._ssl.half()
            self._fc_w = self._fc_w.half()
            self._fc_b = self._fc_b.half()

    def _pooled(self, audio: np.ndarray):
        """Mean-pooled SSL embedding (1024-d torch tensor) for a window."""
        torch = self._torch
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        # reference implementation normalizes the raw waveform with LayerNorm
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
        wav = wav.unsqueeze(0).to(self._device)
        if self._half:
            wav = wav.half()
        with torch.no_grad():
            emb = self._ssl(wav).last_hidden_state  # (1, T, 1024)
            return emb.mean(dim=1)  # (1, 1024)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Public SSL embedding (float32, 1024-d) — reused by the replay
        head so it costs no extra forward pass beyond scoring."""
        if audio is None or len(audio) == 0:
            return np.zeros(1024, dtype=np.float32)
        return self._pooled(audio).float().cpu().numpy().reshape(-1)

    def score(self, audio: np.ndarray) -> float:
        if audio is None or len(audio) == 0:
            return 0.0
        torch = self._torch
        with torch.no_grad():
            pooled = self._pooled(audio)
            logits = pooled @ self._fc_w.T + self._fc_b  # (1, 2)
            probs = torch.softmax(logits.float(), dim=-1)
        return float(probs[0, 0].item())  # index 0 = fake

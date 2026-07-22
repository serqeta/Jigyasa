"""
Codecfake countermeasure scorer.

wav2vec2 (XLS-R-300m) front-end → W2VAASIST head, from the *co-trained CSAM*
checkpoint of the Codecfake work (arXiv:2405.04880, https://github.com/
xieyuankun/Codecfake). It targets neural-codec / vocoder synthesis artifacts —
the family modern commercial TTS (e.g. ElevenLabs) is built on — and so
complements the ASVspoof-trained ensemble on generators it never saw.

Inference mirrors the upstream generate_score.py exactly:
  - waveform @ 16 kHz, padded/truncated to 64600 samples (~4.04 s);
  - Wav2Vec2FeatureExtractor → Wav2Vec2Model, take hidden_states[5];
  - reshape to (1, 1, 1024, T); W2VAASIST returns (feats, logits).

Score convention (calibrated on our fixtures — genuine 0.0002 vs TTS/clone
0.999): softmax index 0 = real, index 1 = fake → we return P(fake).

Loads only if models/codecfake/cotrain_w2v2aasist_CSAM/ exists.

LICENSE: upstream is CC BY-NC-ND 4.0 (NonCommercial, NoDerivatives). The
vendored model.py is a verbatim copy; the weights are not redistributed
(gitignored). Research / demo use only — not for a shipped product.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import types

import numpy as np

from voiceshield import config

_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "codecfake", "cotrain_w2v2aasist_CSAM"
)
_CKPT = os.path.join(_DIR, "anti-spoofing_feat_model.pt")
_FRONTEND = "facebook/wav2vec2-xls-r-300m"
_HIDDEN_LAYER = 5
_CUT = 64600  # ~4.04 s @ 16 kHz — upstream pad_dataset()


def model_available() -> bool:
    return os.path.exists(_CKPT)


def _install_vendor_aliases() -> None:
    """Make the vendored Codecfake code importable for unpickling.

    The checkpoint is a full pickled ``W2VAASIST`` whose class lives in a
    module named ``model``; expose our vendored copy under that name. Its
    ``from pytorch_model_summary import summary`` (used only in the upstream
    __main__) is satisfied with a stub so the file loads verbatim.
    """
    if "pytorch_model_summary" not in sys.modules:
        stub = types.ModuleType("pytorch_model_summary")
        stub.summary = lambda *a, **k: None  # type: ignore[attr-defined]  # noqa: E731
        sys.modules["pytorch_model_summary"] = stub
    import voiceshield.classifier._codecfake.model as _cfm

    sys.modules.setdefault("model", _cfm)


class CodecScorer:
    """P(fake) from the Codecfake co-trained W2VAASIST detector."""

    def __init__(self) -> None:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self._torch = torch
        self._device = config.get_device()
        _install_vendor_aliases()

        # The pickled module prints tensor shapes on load/forward — swallow it.
        with contextlib.redirect_stdout(io.StringIO()):
            add = torch.load(_CKPT, map_location=self._device, weights_only=False)
        self._add = add.to(self._device).eval()

        # Front-end from cache only: a stale HF token in the environment 401s
        # the online repo probe, so never touch the network here.
        self._proc = Wav2Vec2FeatureExtractor.from_pretrained(_FRONTEND, local_files_only=True)
        w2v = Wav2Vec2Model.from_pretrained(_FRONTEND, local_files_only=True)
        w2v.config.output_hidden_states = True
        self._w2v = w2v.to(self._device).eval()  # type: ignore[arg-type]

    def _prep(self, audio: np.ndarray):
        w = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(w) >= _CUT:
            w = w[:_CUT]
        elif len(w) > 0:
            w = np.tile(w, int(_CUT / len(w)) + 1)[:_CUT]
        else:
            w = np.zeros(_CUT, dtype=np.float32)
        iv = self._proc(w, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_values
        return iv.to(self._device)

    def score(self, audio: np.ndarray) -> float:
        torch = self._torch
        iv = self._prep(audio)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            hidden = self._w2v(iv).hidden_states[_HIDDEN_LAYER]
            x = hidden.unsqueeze(0).transpose(2, 3)  # (1, 1, 1024, T)
            _, logits = self._add(x)
            p_fake = torch.softmax(logits, dim=1)[0, 1].item()
        return float(min(max(p_fake, 0.0), 1.0))

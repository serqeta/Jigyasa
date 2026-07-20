"""
Score the eval set with nii-yamagishilab/mms-300m-anti-deepfake.

Runs in a Python 3.10 side-environment (fairseq requirement); CPU-only.
One 4 s window per clip (head of clip) to keep CPU runtime practical.
Output JSON: {filename: {"group": ..., "label": ..., "score": p_fake}}.

Usage:
    <venv310>/bin/python scripts/eval_nii.py --set <dir> --json out.json
"""

import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
import torch
from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model
from huggingface_hub import PyTorchModelHubMixin

SR = 16000
WIN = 4 * SR
DEVICE = "cpu"


class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            feature_grad_mult=1.0,
        )
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        with torch.no_grad():
            return self.model(x.to(DEVICE), mask=False, features_only=True)["x"]


class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(1024, 2)

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav).transpose(1, 2)
        return self.proj_fc(self.adap_pool1d(emb).squeeze(-1))


def score(model, audio: np.ndarray) -> float:
    a = audio[:WIN]
    wav = torch.from_numpy(a).float()
    wav = torch.nn.functional.layer_norm(wav, wav.shape)
    with torch.no_grad():
        probs = torch.softmax(model(wav.unsqueeze(0)), dim=1)
    return float(probs[0, 0])  # index 0 = fake (per model card)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    model = (
        DeepfakeDetector.from_pretrained(
            "nii-yamagishilab/mms-300m-anti-deepfake", cache_dir="models/hf"
        )
        .to(DEVICE)
        .eval()
    )

    out = {}
    files = []
    for group in sorted(os.listdir(args.set)):
        gdir = os.path.join(args.set, group)
        if os.path.isdir(gdir):
            files += [(group, p) for p in sorted(glob.glob(os.path.join(gdir, "*.wav")))]
    for i, (group, p) in enumerate(files):
        audio, _ = sf.read(p, dtype="float32")
        out[os.path.basename(p)] = {
            "group": group,
            "label": 0 if "genuine" in group else 1,
            "score": round(score(model, audio), 4),
        }
        if i % 25 == 0:
            print(f"{i}/{len(files)}", flush=True)

    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.json)


if __name__ == "__main__":
    main()

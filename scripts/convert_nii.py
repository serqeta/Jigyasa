"""
Convert nii-yamagishilab/mms-300m-anti-deepfake from its fairseq-keyed
checkpoint to a transformers-loadable state dict, so it runs in the main
(Python 3.12) process without fairseq.

Output: models/nii_mms300m_converted.pt containing
  {"wav2vec2": <transformers Wav2Vec2Model state dict>,
   "proj_fc": {"weight": ..., "bias": ...}}

Validate afterwards against scripts/eval_nii.py reference scores.
"""

import os
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.join(os.path.dirname(__file__), "..", "models", "nii_mms300m_converted.pt")


def convert_key(k: str) -> str | None:
    """fairseq Wav2Vec2Model key -> transformers Wav2Vec2Model key."""
    if any(s in k for s in ("quantizer", "project_q", "final_proj", "_ema")):
        return None  # pretraining-only heads
    k = k.replace("mask_emb", "masked_spec_embed")
    k = k.replace("post_extract_proj", "feature_projection.projection")
    if k.startswith("layer_norm."):
        return "feature_projection.layer_norm." + k.split(".", 1)[1]
    # feature extractor convs (layer_norm extractor mode: conv=.0, ln=.2)
    if k.startswith("feature_extractor.conv_layers."):
        parts = k.split(".")
        i, sub, rest = parts[2], parts[3], ".".join(parts[4:])
        if sub == "0":
            return f"feature_extractor.conv_layers.{i}.conv.{rest}"
        if sub == "2":
            # fairseq wraps the LN: conv_layers.{i}.2.1.{weight,bias}
            rest = rest.removeprefix("1.")
            return f"feature_extractor.conv_layers.{i}.layer_norm.{rest}"
        return None
    if k.startswith("encoder.pos_conv.0."):
        return k.replace("encoder.pos_conv.0.", "encoder.pos_conv_embed.conv.")
    if k.startswith("encoder.layers."):
        k = k.replace(".self_attn.", ".attention.")
        k = k.replace(".self_attn_layer_norm.", ".layer_norm.")
        k = k.replace(".fc1.", ".feed_forward.intermediate_dense.")
        k = k.replace(".fc2.", ".feed_forward.output_dense.")
        return k
    return k  # encoder.layer_norm, masked_spec_embed, etc.


def main() -> None:
    path = hf_hub_download(
        "nii-yamagishilab/mms-300m-anti-deepfake",
        "model.safetensors",
        cache_dir="models/hf",
    )
    sd = load_file(path)

    w2v, head = {}, {}
    skipped = []
    for k, v in sd.items():
        if k.startswith("proj_fc."):
            head[k.split(".", 1)[1]] = v
            continue
        if not k.startswith("m_ssl.model."):
            skipped.append(k)
            continue
        nk = convert_key(k[len("m_ssl.model.") :])
        if nk is not None:
            w2v[nk] = v

    print(f"converted {len(w2v)} SSL tensors, head={list(head)}, skipped={len(skipped)}")

    # sanity: load into a transformers model strictly
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    cfg = Wav2Vec2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        conv_bias=True,
        num_feat_extract_layers=7,
        feat_proj_dropout=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layerdrop=0.0,
    )
    model = Wav2Vec2Model(cfg)
    missing, unexpected = model.load_state_dict(w2v, strict=False)
    real_missing = [m for m in missing if "position" not in m and "rotary" not in m]
    print("missing:", real_missing[:8], f"(+{max(0, len(real_missing) - 8)} more)")
    print("unexpected:", unexpected[:8], f"(+{max(0, len(unexpected) - 8)} more)")
    if real_missing or unexpected:
        print("WARNING: non-clean load — validate carefully")

    torch.save({"wav2vec2": w2v, "proj_fc": head, "config": cfg.to_dict()}, OUT)
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()

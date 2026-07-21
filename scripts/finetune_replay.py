"""
LoRA fine-tune the NII wav2vec2 backbone for REPLAY detection on EchoFake,
with on-the-fly RawBoost-style channel augmentation. Fits a 6 GB GPU
(LoRA + fp16 + gradient checkpointing). Few trainable params → resists the
source-domain overfitting seen with frozen-embedding + more data.

Task: class 1 = replay (played through a loudspeaker), class 0 = not.
Model-selection on EchoFake dev; the honest metric is held-out AUC on our
12 browser-mic recordings.

Usage:
    python scripts/finetune_replay.py --per-class 4000 --epochs 3 [--smoke]
"""

import argparse
import glob
import io
import os
import sys

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
SR = 16000
WIN = 4 * SR
REAL = "/home/manicguy/Projects/Jigyasa/eval_recordings"
NII_W = os.path.join(os.path.dirname(__file__), "..", "models", "nii_mms300m_converted.pt")
TRAIN_SHARDS = [
    "data/train-00000-of-00004.parquet",
    "data/train-00002-of-00004.parquet",
    "data/train-00001-of-00004.parquet",
    "data/train-00003-of-00004.parquet",
]
DEV_SHARD = "data/dev-00000-of-00001.parquet"


def get_shard(fn):
    from huggingface_hub import hf_hub_download

    return hf_hub_download("EchoFake/EchoFake", fn, repo_type="dataset")


def load_wavs(shard, per, rng):
    """Return list of (int16 waveform[<=WIN], replay_label) sampled from a shard."""
    import librosa
    import pandas as pd

    df = pd.read_parquet(get_shard(shard))
    idx = rng.permutation(len(df))[:per]
    out = []
    for i in idx:
        r = df.iloc[i]
        try:
            a, sr = sf.read(io.BytesIO(r["path"]["bytes"]), dtype="float32")
        except Exception:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR:
            a = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=SR)
        a = a[:WIN]
        if len(a) < SR:
            continue
        lab = 1 if r["label"].startswith("replay_") else 0
        out.append(((a * 32768).astype(np.int16), lab))
    return out


def rawboost(x, rng):
    """Lightweight RawBoost: convolutive coloration + reverb + additive noise + gain."""
    from scipy.signal import butter, sosfilt

    if rng.random() < 0.8:
        lo, hi = rng.uniform(40, 400), rng.uniform(3000, 7800)
        x = sosfilt(butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos"), x).astype(
            np.float32
        )
    if rng.random() < 0.5:
        t60 = rng.uniform(0.05, 0.5)
        n = int(t60 * SR)
        ir = (rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / SR / t60)).astype(
            np.float32
        ) * 0.3
        ir[0] = 1.0
        x = np.convolve(x, ir)[: len(x)].astype(np.float32)
    if rng.random() < 0.7:
        snr = rng.uniform(5, 30)
        p = np.mean(x**2) + 1e-9
        nz = rng.standard_normal(len(x)).astype(np.float32)
        nz *= np.sqrt(p / (10 ** (snr / 10)) / (np.mean(nz**2) + 1e-12))
        x = x + nz
    return (x * rng.uniform(0.6, 1.4)).astype(np.float32)


def fix_len(x):
    """Pad/truncate to exactly WIN samples so a batch can be stacked."""
    if len(x) >= WIN:
        return x[:WIN]
    return np.pad(x, (0, WIN - len(x)))


def build_model(device):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    blob = torch.load(NII_W, map_location="cpu", weights_only=False)
    cfg = Wav2Vec2Config(**dict(blob["config"]))
    ssl = Wav2Vec2Model(cfg)
    ssl.load_state_dict(blob["wav2vec2"], strict=True)
    ssl.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
    )
    ssl = get_peft_model(ssl, lora)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ssl = ssl
            self.head = torch.nn.Linear(1024, 2)

        def forward(self, wav):
            wav = torch.nn.functional.layer_norm(wav, wav.shape[1:])
            h = self.ssl(wav).last_hidden_state.mean(dim=1)
            return self.head(h)

    return Net().to(device)


def wav_to_tensor(a_i16, torch):
    return torch.from_numpy((a_i16.astype(np.float32) / 32768.0))


def embed_score(model, wavs, device, torch):
    model.eval()
    probs = []
    with torch.no_grad():
        for a in wavs:
            w = wav_to_tensor(a, torch).unsqueeze(0).to(device)
            p = torch.softmax(model(w).float(), -1)[0, 1].item()
            probs.append(p)
    return np.array(probs)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(sum((pos > n).sum() + 0.5 * (pos == n).sum() for n in neg) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save", action="store_true", help="save LoRA adapter + head to models/")
    args = ap.parse_args()

    import torch

    from voiceshield import config

    device = config.get_device()
    rng = np.random.default_rng(0)
    half = args.per_class // 2

    print("loading EchoFake train waveforms…", flush=True)
    train = []
    for sh in TRAIN_SHARDS:
        train += load_wavs(sh, half, rng)
    rng.shuffle(train)
    print(f"  train clips: {len(train)} (replay={sum(lb for _, lb in train)})", flush=True)

    print("loading EchoFake dev + our recordings…", flush=True)
    dev = load_wavs(DEV_SHARD, 400, rng)
    dev_w = [w for w, _ in dev]
    dev_y = np.array([lb for _, lb in dev])
    our_f = sorted(glob.glob(f"{REAL}/*.wav"))
    our_w = [(sf.read(f, dtype="float32")[0][:WIN] * 32768).astype(np.int16) for f in our_f]
    our_y = np.array([0 if os.path.basename(f).startswith("live_") else 1 for f in our_f])

    model = build_model(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {trainable / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    lossf = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    n = 2 * args.batch if args.smoke else len(train)
    best = {"our_auc": 0.0}
    for ep in range(1 if args.smoke else args.epochs):
        model.train()
        order = rng.permutation(len(train))[:n]
        tot = 0.0
        for b in range(0, len(order), args.batch):
            idx = order[b : b + args.batch]
            xs = np.stack(
                [fix_len(rawboost(train[i][0].astype(np.float32) / 32768.0, rng)) for i in idx]
            )
            xb = torch.from_numpy(xs).to(device)
            yb = torch.tensor([train[i][1] for i in idx]).to(device)
            opt.zero_grad()
            if scaler:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = lossf(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = lossf(model(xb), yb)
                loss.backward()
                opt.step()
            tot += loss.item()
            if args.smoke:
                print(
                    f"  smoke batch {b // args.batch}: loss={loss.item():.3f} "
                    f"GPU={torch.cuda.max_memory_allocated() / 1e9:.2f}GB",
                    flush=True,
                )
        oep = embed_score(model, our_w, device, torch)
        dep = embed_score(model, dev_w, device, torch)
        oa = auc(oep[our_y == 1], oep[our_y == 0])
        da = auc(dep[dev_y == 1], dep[dev_y == 0])
        print(
            f"epoch {ep}: loss={tot / (len(order) // args.batch):.3f}  "
            f"dev_AUC={da:.3f}  OUR_AUC={oa:.3f}",
            flush=True,
        )
        if not args.smoke and da >= best.get("dev_auc", 0):
            best = {
                "dev_auc": da,
                "our_auc": oa,
                "our_probs": list(
                    zip([os.path.basename(f) for f in our_f], our_y.tolist(), oep.round(3).tolist())
                ),
            }

    if args.smoke:
        print("SMOKE OK — memory fit confirmed")
        return
    print("\n=== BEST (selected on EchoFake dev) ===")
    print(f"dev_AUC={best['dev_auc']:.3f}  OUR_AUC={best['our_auc']:.3f}")
    for nme, yy, pp in sorted(best["our_probs"], key=lambda t: t[2]):
        print(f"   {nme:34s} {'REPLAY' if yy else 'live  '} {pp:.3f}")

    if args.save:
        out = os.path.join(os.path.dirname(__file__), "..", "models", "replay_lora")
        os.makedirs(out, exist_ok=True)
        model.ssl.save_pretrained(out)  # LoRA adapter only (small)
        torch.save(model.head.state_dict(), os.path.join(out, "head.pt"))
        print("\nsaved fine-tuned replay model →", os.path.abspath(out))


if __name__ == "__main__":
    main()

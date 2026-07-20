"""
Build the model-selection evaluation set.

Writes labeled WAVs (16 kHz mono float32) into an output directory:

  genuine/    all LibriSpeech dev-clean validation speakers (~16 s each)
              + 2 CMU Arctic speakers (hard genuine — studio TTS-source voices)
  fake/       SpeechT5 clones with several CMU Arctic x-vectors
              + MMS-TTS (VITS) sentences
              + the ElevenLabs WhatsApp clip if present in ~/Downloads
  *_codec/    opus-compressed (16 kbps, WhatsApp-like) variants of both
              classes — the "hard channel" condition

Usage:
    python scripts/build_evalset.py [--out DIR]
"""

import argparse
import glob
import io
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SR = 16000
TARGET_SECONDS = 16.0

SENTENCES = [
    "Please transfer the funds to my new account before the end of the day.",
    "I have forgotten my password, but you can verify me by my voice.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Kindly update my registered mobile number immediately after this call.",
    "My account balance seems wrong and I would like to raise a dispute.",
    "Yes, I authorize this transaction of fifty thousand rupees right now.",
]

ARCTIC = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/arctic_a{i:04d}.wav"


def _fade(a, ms=10.0):
    n = int(ms / 1000 * SR)
    if len(a) < 2 * n:
        return a
    r = np.linspace(0, 1, n, dtype=np.float32)
    a = a.copy()
    a[:n] *= r
    a[-n:] *= r[::-1]
    return a


def _assemble(pieces, gap_s=0.2):
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    out = []
    for p in pieces:
        out.extend([_fade(p), gap])
    a = np.concatenate(out)[: int(TARGET_SECONDS * SR)]
    return (0.8 * a / (np.max(np.abs(a)) + 1e-9)).astype(np.float32)


def build_genuine(out_dir):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    pq = hf_hub_download("openslr/librispeech_asr", "clean/validation/0000.parquet",
                         repo_type="dataset", revision="refs/convert/parquet")
    df = pd.read_parquet(pq)
    n = 0
    for spk, grp in df.groupby("speaker_id"):
        clips = [sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0]
                 for _, r in grp.head(10).iterrows()]
        if sum(len(c) for c in clips) < TARGET_SECONDS * SR:
            continue
        sf.write(os.path.join(out_dir, f"libri_{spk}.wav"), _assemble(clips), SR)
        n += 1
    print(f"  genuine: {n} LibriSpeech speakers")

    import urllib.request
    for spk in ["bdl", "slt"]:
        pieces, total, i = [], 0.0, 1
        while total < TARGET_SECONDS and i < 40:
            with urllib.request.urlopen(ARCTIC.format(spk=spk, i=i)) as r:
                a, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
            pieces.append(a)
            total += len(a) / sr
            i += 1
        sf.write(os.path.join(out_dir, f"arctic_{spk}.wav"), _assemble(pieces), SR)
    print("  genuine: +2 CMU Arctic (hard domain)")


def build_speecht5(out_dir):
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    from voiceshield import config

    device = config.get_device()
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    pq = hf_hub_download("Matthijs/cmu-arctic-xvectors", "default/validation/0000.parquet",
                         repo_type="dataset", revision="refs/convert/parquet")
    xv = pd.read_parquet(pq)

    for spk in ["bdl", "slt", "rms", "clb", "ksp"]:
        vecs = xv[xv["filename"].str.contains(spk)]["xvector"].tolist()
        emb = torch.tensor(np.mean(np.asarray(vecs), axis=0),
                           dtype=torch.float32).unsqueeze(0).to(device)
        chunks = []
        for text in SENTENCES:
            inp = processor(text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                speech = model.generate_speech(inp["input_ids"], emb, vocoder=vocoder)
            chunks.append(speech.cpu().numpy())
        sf.write(os.path.join(out_dir, f"speecht5_{spk}.wav"),
                 _assemble(chunks, gap_s=0.3), SR)
    print("  fake: 5 SpeechT5 voice clones")


def build_vits(out_dir):
    import torch
    from transformers import AutoTokenizer, VitsModel

    from voiceshield import config

    device = config.get_device()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng", cache_dir=config.HF_CACHE_DIR)
    model = VitsModel.from_pretrained("facebook/mms-tts-eng",
                                      cache_dir=config.HF_CACHE_DIR).to(device)
    chunks = []
    for text in SENTENCES:
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            wav = model(**inp).waveform[0].cpu().numpy()
        # MMS-TTS also outputs 16 kHz
        chunks.append(wav.astype(np.float32))
    sf.write(os.path.join(out_dir, "vits_mms.wav"), _assemble(chunks, gap_s=0.3), SR)
    print("  fake: 1 MMS-TTS (VITS)")


def add_elevenlabs(out_dir):
    import librosa

    hits = glob.glob(os.path.expanduser("~/Downloads/WhatsApp Audio*.mp3"))
    for i, p in enumerate(sorted(hits)):
        a, _ = librosa.load(p, sr=SR, mono=True)
        sf.write(os.path.join(out_dir, f"elevenlabs_{i}.wav"), a.astype(np.float32), SR)
    print(f"  fake: {len(hits)} ElevenLabs clip(s)")


def codec_variants(src_dir, dst_dir):
    """Opus 16 kbps round-trip — the WhatsApp/telephony channel condition."""
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for wav in sorted(glob.glob(os.path.join(src_dir, "*.wav"))):
        base = os.path.basename(wav)
        tmp = os.path.join(dst_dir, base + ".opus")
        out = os.path.join(dst_dir, base)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav,
                        "-c:a", "libopus", "-b:a", "16k", tmp], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-ar", str(SR), "-ac", "1", out], check=True)
        os.unlink(tmp)
        n += 1
    print(f"  codec variants: {n} in {os.path.basename(dst_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("EVALSET_DIR", "eval_set"))
    args = ap.parse_args()

    gen = os.path.join(args.out, "genuine")
    fake = os.path.join(args.out, "fake")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(fake, exist_ok=True)

    print("Building genuine…")
    build_genuine(gen)
    print("Building fakes…")
    build_speecht5(fake)
    build_vits(fake)
    add_elevenlabs(fake)
    print("Building codec variants…")
    codec_variants(gen, os.path.join(args.out, "genuine_codec"))
    codec_variants(fake, os.path.join(args.out, "fake_codec"))
    print("Done →", os.path.abspath(args.out))


if __name__ == "__main__":
    main()

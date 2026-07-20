"""
Build the real-audio acceptance fixtures in tests/fixtures/:

  genuine_male_16k.wav    LibriSpeech dev-clean (real human male, CC BY 4.0)
  genuine_female_16k.wav  LibriSpeech dev-clean (real human female)
  tts_synthetic_16k.wav   SpeechT5 TTS, generic female voice
  cloned_voice_16k.wav    SpeechT5 TTS conditioned on CMU Arctic 'bdl'
                          x-vectors — a voice clone of a real speaker,
                          mirroring the fraud scenario end-to-end.

Genuine audio comes from LibriSpeech (real audiobook speech — the bonafide
domain the anti-spoofing checkpoints were trained against). CMU Arctic was
deliberately NOT used for the genuine fixtures: those studio voices seeded
decades of TTS systems and every anti-spoof model we validated scores them
as synthetic. Synthetic audio is generated locally (GPU if available).

Usage:
    python scripts/build_fixtures.py
"""

import io
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
TARGET_SECONDS = 16.0
SR = 16000

# Harvard-style sentences; enough text for ~16 s of TTS output.
SENTENCES = [
    "Please transfer the funds to my new account before the end of the day.",
    "I have forgotten my password, but you can verify me by my voice.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Kindly update my registered mobile number immediately after this call.",
    "My account balance seems wrong and I would like to raise a dispute.",
    "Yes, I authorize this transaction of fifty thousand rupees right now.",
]


def _fade(audio: np.ndarray, ms: float = 10.0) -> np.ndarray:
    n = int(ms / 1000.0 * SR)
    if len(audio) < 2 * n:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio = audio.copy()
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def _assemble(pieces: list[np.ndarray], gap_s: float = 0.20) -> np.ndarray:
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    out: list[np.ndarray] = []
    for p in pieces:
        out.extend([_fade(p), gap])
    audio = np.concatenate(out)[: int(TARGET_SECONDS * SR)]
    return 0.8 * audio / (np.max(np.abs(audio)) + 1e-9)


def _librispeech_rows():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    pq = hf_hub_download(
        "openslr/librispeech_asr",
        "clean/validation/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    return pd.read_parquet(pq)


def _median_f0(audio: np.ndarray) -> float:
    import librosa

    f0 = librosa.yin(audio[: 4 * SR], fmin=60, fmax=400, sr=SR)
    return float(np.median(f0))


# Pinned after scanning every dev-clean validation speaker against the
# ensemble (2026-07-08): both speakers score near-zero on ssl AND wavlm.
# Note ~4/40 genuine speakers trip wavlm to ~1.0 (e.g. 8297, 1993, 2412,
# 3000) — real-speech false-positive channels of that checkpoint; they make
# poor acceptance fixtures but are useful robustness probes.
GENUINE_SPEAKERS = {"genuine_male_16k.wav": 3752, "genuine_female_16k.wav": 2035}


def _build_genuine(df) -> None:
    """One male and one female LibriSpeech speaker, ~16 s each."""
    for name, spk in GENUINE_SPEAKERS.items():
        grp = df[df["speaker_id"] == spk]
        clips = [
            sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0] for _, r in grp.iterrows()
        ]
        audio = _assemble(clips)
        sf.write(os.path.join(FIXTURES, name), audio, SR)
        f0 = _median_f0(audio)
        print(
            f"  {name}: {len(audio) / SR:.1f}s (LibriSpeech speaker {spk}, median F0 {f0:.0f} Hz)"
        )


def _build_tts(out_name: str, speaker_filter: str | None) -> None:
    """
    Generate speech with SpeechT5. speaker_filter=None uses a generic
    female x-vector; a filter like 'bdl' averages that CMU Arctic
    speaker's x-vectors — i.e., clones a real speaker's voice.
    """
    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    from voiceshield import config

    device = config.get_device()
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    # The x-vectors dataset ships a legacy loading script; read its
    # auto-converted parquet directly instead.
    pq = hf_hub_download(
        "Matthijs/cmu-arctic-xvectors",
        "default/validation/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    xv = pd.read_parquet(pq)
    if speaker_filter is None:
        emb = torch.tensor(xv.iloc[7306]["xvector"])  # 'slt' female, the reference demo voice
    else:
        vecs = xv[xv["filename"].str.contains(speaker_filter)]["xvector"].tolist()
        emb = torch.tensor(np.mean(np.asarray(vecs), axis=0), dtype=torch.float32)
    emb = emb.unsqueeze(0).to(device)

    chunks = []
    for text in SENTENCES:
        inputs = processor(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], emb, vocoder=vocoder)
        chunks.append(speech.cpu().numpy())

    audio = _assemble(chunks, gap_s=0.30)
    sf.write(os.path.join(FIXTURES, out_name), audio.astype(np.float32), SR)
    who = f"cloned '{speaker_filter}'" if speaker_filter else "generic TTS voice"
    print(f"  {out_name}: {len(audio) / SR:.1f}s ({who})")


def main() -> None:
    os.makedirs(FIXTURES, exist_ok=True)
    print("Downloading genuine speech (LibriSpeech dev-clean)…")
    _build_genuine(_librispeech_rows())
    print("Generating synthetic speech (SpeechT5)…")
    _build_tts("tts_synthetic_16k.wav", speaker_filter=None)
    _build_tts("cloned_voice_16k.wav", speaker_filter="bdl")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Hindi evaluation set: genuine Hindi speech (google/fleurs hi_in, validation
split via the datasets-server rows API) + Hindi fakes generated with
facebook/mms-tts-hin (VITS).

Writes <out>/hindi_genuine/*.wav and <out>/hindi_fake/*.wav (16 kHz mono).

Usage:
    python scripts/build_hindi_eval.py --out <dir> [--n-genuine 40]
"""

import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SR = 16000
ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=google/fleurs"
    "&config=hi_in&split=validation&offset={off}&length={n}"
)

# Hindi banking/fraud-scenario sentences (Devanagari)
SENTENCES = [
    "कृपया आज शाम से पहले मेरे नए खाते में पैसे ट्रांसफर कर दीजिए।",
    "मैं अपना पासवर्ड भूल गया हूं, लेकिन आप मेरी आवाज़ से मेरी पहचान कर सकते हैं।",
    "मेरे खाते की शेष राशि गलत लग रही है और मैं शिकायत दर्ज करना चाहता हूं।",
    "कृपया इस कॉल के तुरंत बाद मेरा पंजीकृत मोबाइल नंबर बदल दीजिए।",
    "हां, मैं पचास हज़ार रुपये के इस लेनदेन को अभी अधिकृत करता हूं।",
    "मुझे तुरंत एक नया डेबिट कार्ड चाहिए, पुराना कार्ड खो गया है।",
]


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "voiceshield-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_genuine(out_dir, n_total):
    import librosa

    saved = 0
    off = 0
    while saved < n_total and off < 300:
        rows = json.loads(_get(ROWS.format(off=off, n=50)))["rows"]
        if not rows:
            break
        for row in rows:
            if saved >= n_total:
                break
            r = row["row"]
            try:
                raw = _get(r["audio"][0]["src"])
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
            if len(audio) < 4 * SR:
                continue
            sf.write(
                os.path.join(out_dir, f"fleurs_hi_{r['id']}.wav"), audio.astype(np.float32), SR
            )
            saved += 1
        off += 50
    print(f"  hindi genuine: {saved} FLEURS clips")


def build_fakes(out_dir):
    import torch
    from transformers import AutoTokenizer, VitsModel

    from voiceshield import config

    device = config.get_device()
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-hin", cache_dir=config.HF_CACHE_DIR)
    model = VitsModel.from_pretrained("facebook/mms-tts-hin", cache_dir=config.HF_CACHE_DIR).to(
        device
    )
    n = 0
    for i, text in enumerate(SENTENCES):
        inputs = tok(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            print(f"  ! tokenizer produced empty ids for sentence {i} (uroman needed?)")
            continue
        inputs = inputs.to(device)
        with torch.no_grad():
            wav = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
        wav = 0.8 * wav / (np.max(np.abs(wav)) + 1e-9)
        # pad/tile to ≥ 6 s so the pipeline has enough chunks
        while len(wav) < 6 * SR:
            wav = np.concatenate([wav, np.zeros(int(0.3 * SR), dtype=np.float32), wav])
        sf.write(os.path.join(out_dir, f"mms_hin_{i}.wav"), wav[: 16 * SR], SR)
        n += 1
    print(f"  hindi fake: {n} MMS-TTS-hin clips")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-genuine", type=int, default=40)
    args = ap.parse_args()
    gen = os.path.join(args.out, "hindi_genuine")
    fake = os.path.join(args.out, "hindi_fake")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(fake, exist_ok=True)
    fetch_genuine(gen, args.n_genuine)
    build_fakes(fake)
    print("Done →", os.path.abspath(args.out))


if __name__ == "__main__":
    main()

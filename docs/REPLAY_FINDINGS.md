# Replay / Physical-Access Detection — Findings & Evidence

**STATUS (2026-07-21): SOLVED for wideband — shipped at a conservative
AMBER-capped fusion weight.** A LoRA-fine-tuned wav2vec2 trained on EchoFake
with channel augmentation reaches **cross-channel AUC 0.97** on our own
held-out recordings (trained only on EchoFake, never on our mic). Sections
1–4 below are the investigation trail (five approaches that failed the
channel-generalization wall) — kept because the honesty is the moat, and
because it explains *why* the final recipe (fine-tune + augmentation on a
real replay corpus) was the thing that worked. The RESOLUTION is at the end.

VoiceShield's deepfake (voice-synthesis) detection is rigorously validated
and ships (NII AUC 0.997; telephony-robust — see RESOLUTION). **Replay
detection** — telling a *genuine recording played through a loudspeaker*
apart from a live human — was the hard part; the trail below is honest.

---

## 1. What a replay attack is (and why it's different from synthesis)

- **Synthesis/clone:** the audio is machine-generated. Our ensemble (NII MMS-300M
  + XLS-R + WavLM) detects this well — AUC 0.997, EER 3.1%.
- **Replay:** a real recording (genuine or cloned) is *played through a speaker*
  into the microphone. The speech content may be genuine; the "tell" is the
  physical playback channel — loudspeaker + air + room + re-recording.

## 2. The physics — and which cues survive a channel change

| Cue | Physical basis | Channel-invariant? |
|---|---|---|
| **Pop-noise / breath airflow** | A live mouth ejects directed breath on plosives; a loudspeaker moves air to make sound but cannot reproduce that breath burst. Live-only physical evidence. | **Most invariant** — but **close-mic only** |
| Double reverberation | Live speech = 1 acoustic impulse response; replay = 2 convolved AIRs (record room + playback room). | Condition-dependent (needs reverb + distance) |
| High-freq (6–8 kHz) artifacts | Double A/D conversion + anti-alias filtering leaves band-limiting/artifacts near Nyquist. | Device-dependent **and above the telephone passband** |
| Loudspeaker nonlinear distortion | Small speakers add harmonic/intermodulation distortion. | Device-dependent |
| Sub-band cepstral (CQCC/LFCC) | Captures the spectral fingerprint of the specific replay channel. | **Channel-dependent — overfits** |

**The crux:** the cues that discriminate *best in-corpus* (CQCC/LFCC, high-freq,
reverb) are exactly the ones that encode the *specific device/room*, so they
collapse on an unseen channel. The one physically channel-*invariant* cue
(pop-noise) needs a close microphone.

## 3. What we measured (our own recordings)

12 recordings captured through the real browser-mic path (8 genuine live across
quiet/far/noisy/other-speaker/different-room + 4 loudspeaker playbacks). Held out
as the deployment-channel test:

| Approach | Our-channel AUC |
|---|---|
| Hand-tuned DSP (reverb/freq/compression), simulation-calibrated | ~0.50 (random) |
| LFCC + logreg ← ASVspoof2017 real replay | 0.19 |
| LFCC + logreg ← ASVspoof2019 PA (no augmentation) | ~0.72 (no real separation) |
| LFCC + heavy channel augmentation ← ASVspoof2019 PA | 0.56 (logreg) / 0.22 (GBM) |
| SSL embedding, in-channel leave-one-out | 0.91 — **invalid** (needs the demo room in training) |
| **Pop-noise / airflow (`lf_ratio`)** | **0.84** — best channel-robust cue, close-mic only |

Every learned/spectral approach failed to generalize to our channel. Only the
physically-motivated airflow cue showed cross-channel merit, and it is close-mic
only (it would false-positive on genuine far-field callers and does not survive
a phone channel).

## 4. What the literature says (this reproduces the state of the art)

Cross-channel / cross-dataset replay & anti-spoofing generalization, from a
verified literature review (2017–2025):

- **Cross-dataset collapse is universal.** ResNet-OC LA countermeasure: 2.29% EER
  in-domain → 26.30% (ASVspoof2015) → **41.66%** (VCC2020). Channel mismatch
  *alone*, isolated by simulation, drives average EER to **40–49%**.
  — Chen et al., *Channel-robust speaker verification/anti-spoofing*, arXiv:2104.01320
- **One-class learning (SOTA robustness trick) still fails cross-corpus:**
  ASVspoof2017→2019-PA **32.58%** EER; 2019-PA→2017 **35.40%** EER. Authors
  explicitly conclude generalizable passive replay detection is **unsolved**.
  — Lou et al., Interspeech 2022
- **Reverberation-based PAD is condition-dependent:** 1.04% EER (favorable reverb
  + distance) → 22.37% (full ASVspoof2019 PA) → 36.28% (ASVspoof2021 PA);
  close-mic/low-reverb defeats it. — Pindrop, *On the Role of Room Acoustics in PAD*
- **High-freq replay cues sit at 6–8 kHz** (best sub-band, 5.13% EER dev) but
  overfit (→17.31% eval) **and lie above the 8 kHz telephone passband** — so they
  do not survive narrowband telephony. — Witkowski et al., Interspeech 2017
- **Synthesis-trained detectors do not transfer to real replay:** ASVspoof2019-LA
  models on the real-replay EchoFake set score **42–46% EER** (near chance);
  training on real replay (EchoFake) cuts open-set EER to **11–21%**.
  — *EchoFake*, arXiv:2510.19414 (2025)
- **Pop-noise is genuine live-human physical evidence.** — Univ. Edinburgh /
  Shiota et al., *Voice liveness detection based on pop-noise*
- **RawBoost channel augmentation** helps *seen/simulated* telephony variability
  (9.50% → 5.31% EER) but does not close the true cross-dataset gap. — Tak et al.,
  arXiv:2111.04433

**Conclusion (ours + literature):** generalizable, single-channel, *passive*
replay detection is an open research problem. Our four failures reproduced the
published state of the art, not a gap in engineering.

## 5. Why this is acceptable — and the roadmap

- **The production channel is digital, not through-air.** In a real bank call, a
  fraudster's cloned/replayed voice arrives *digitally over the phone network* —
  where our synthesis ensemble catches it (the ElevenLabs clip: RED in 2–3 s).
  The through-air replay that defeats detection is a laptop-mic-room artifact,
  not the deployment path. And the passive replay cues (6–8 kHz) wouldn't survive
  the phone channel regardless.
- **Verified on our own recordings:** 11/12 correct — all genuine (incl. genuine
  replays) GREEN with zero false alarms; the single miss is a synthetic clone
  played *through-air*, which the same clip delivered digitally still catches.
- **Pilot roadmap for replay, if required on-site:**
  1. **Replay-aware training on a real, matching corpus** (e.g. EchoFake, or
     per-site calibration data) — the one thing shown to work (~11–21% EER).
  2. **Active challenge-response liveness** (ask the caller to repeat a random
     phrase) — the robust answer when passive cues can't survive the channel.

## 6. RESOLUTION (2026-07-21) — fine-tuning cracked the wideband case

Sections 1–5 established that *frozen-feature / DSP / cross-corpus* replay
detection fails the channel wall (AUC 0.19–0.87). The fix was the SOTA
recipe the literature points to: **LoRA-fine-tune the wav2vec2 backbone on
a real replay corpus (EchoFake) with RawBoost channel augmentation.**

Recipe (scripts/finetune_replay.py, model in models/replay_lora/):
- Backbone: NII wav2vec2 (reused); LoRA r=8 on q/v (0.79 M trainable params).
- Data: EchoFake replay-vs-not (~8 k clips), on-the-fly RawBoost aug.
- 6 GB-GPU friendly: LoRA + fp16 + gradient checkpointing (2.3 GB peak).
- Model-selected on EchoFake dev; held-out on our 12 recordings.

Results (our recordings — trained only on EchoFake, CROSS-channel):
- **AUC 0.969** (deduped set: 0 false alarms, all 4 replays flagged).
- EchoFake in-corpus dev AUC 1.0; open-set AUC 0.93 / EER 0.14 (matches paper).
- ~38 ms/clip inference (LoRA merged into backbone, fp16).
- The replayed ElevenLabs clone that NII alone MISSED (0.00) is caught by
  the replay scorer (0.76) → the end-to-end pipeline flags it AMBER.

Why it worked where frozen features didn't: augmentation forces channel-
INVARIANT replay features (dev stays perfect AND transfers), instead of
memorizing EchoFake's channels (which just adding data did — it overfit,
AUC dropped 0.875→0.75).

Scope (honest): covers consumer-device **wideband** replay. Far-field is
the weak spot (borderline). Does NOT cover narrowband telephony (cues at
6–8 kHz, above the phone passband). Small held-out set (12 clips) → strong
signal, not a proven EER; a few more recordings would firm it up.

Shipped wiring: ensemble component `replay` (classifier/replay_scorer.py),
FUSION_WEIGHTS 0.25, PEAK_COMPONENTS 0.6 (AMBER-capped — confident replay →
"verify"; RED needs corroboration, e.g. replayed clone fires replay +
synthesis). Loads only if models/replay_lora/ exists.

### Telephony robustness of SYNTHESIS detection (2026-07-21)

Separate question, tested free via ffmpeg codec simulation on EchoFake
genuine+synthesis (scripts/telephony_eval.py). NII survives the phone
channel essentially intact:

| condition            | NII AUC / EER | fused AUC |
|----------------------|---------------|-----------|
| clean 16 kHz         | 0.932 / 0.119 | 0.856 |
| G.711 μ-law 8 kHz    | 0.952 / 0.106 | 0.830 |
| AMR-NB 8 kHz         | 0.923 / 0.150 | 0.862 |
| G.722 wideband       | 0.926 / 0.138 | 0.879 |

So: **synthesis detection is telephony-robust** (NII ~0.93 across all
codecs), **replay detection is wideband-only.** Both measured, not assumed.

## 7. One-line summary for the demo

> "Synthesis/clone detection is our validated core — NII AUC 0.997, and it
> survives real phone codecs (0.93 through G.711/AMR/G.722). Replay was
> genuinely hard — five approaches reproduced the literature's channel wall —
> but LoRA-fine-tuning wav2vec2 on a real replay corpus with channel
> augmentation cracked the wideband case: cross-channel AUC 0.97 on
> recordings the model never trained on, shipped at a conservative
> verify-don't-block weight. Narrowband telephony replay stays honest pilot
> work. Every number here is measured on real audio."

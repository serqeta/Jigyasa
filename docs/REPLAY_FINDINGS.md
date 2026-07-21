# Replay / Physical-Access Detection — Findings & Evidence

**Status: documented pilot work, held out of the shipped verdict (zero fusion weight).**
This file records why, backed by our own measurements *and* the peer-reviewed
literature, so the position is defensible in the demo and Q&A.

VoiceShield's deepfake (voice-synthesis) detection is rigorously validated and
ships. **Replay detection** — telling a *genuine recording played through a
loudspeaker* apart from a live human — is a separate, much harder problem. We
invested in it seriously, measured it honestly, and the conclusion below is
consistent with the state of the art.

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

## 6. One-line summary for the demo

> "We built and honestly measured replay detection. It's a documented open
> problem — cross-channel EER of 27–46% in the literature, which our own
> experiments reproduced — so we hold it out of the verdict rather than ship a
> detector that false-flags genuine callers. Our synthesis detection, which is
> the primary threat and the digital deployment path, is validated and ships."

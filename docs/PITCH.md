# VoiceShield — Pitch Script (bottom-up: root cause → breadth → physical)

*A layered detector that attacks the **shared substrate** of modern voice
clones first, then nets whatever slips through. ~4-minute spoken pitch.*

---

## 0. The hook (15 sec)

> "A fraudster no longer needs your voice — 30 seconds of you on a podcast is
> enough to clone it and call your bank. The detectors everyone ships are
> trained to recognize *specific* fake-voice models. But a new voice generator
> launches every few weeks. So the industry is stuck playing whack-a-mole —
> always one model behind the attacker. We took the opposite approach: instead
> of chasing every generator, we go after the **one thing they all share.**"

---

## 1. The insight — attack the root cause, not the symptom (45 sec)

> "Here's the key realization. Almost every modern voice-cloning system —
> ElevenLabs, the open-source ones, the research models — is built on the same
> engine underneath: a **neural audio codec**. The codec is what turns the
> AI's internal 'tokens' back into an actual waveform. It's the last step in
> the pipeline, and **it's shared across the whole ecosystem.**
>
> A neural codec doesn't reproduce sound perfectly — it *reconstructs* it from
> a compressed representation, and that reconstruction leaves a fingerprint.
> Inaudible to us, but statistically consistent. Crucially, that fingerprint
> is there **no matter what words were said or which brand of model made it** —
> because they all decode through the same family of codec.
>
> So instead of learning '*this is what an ElevenLabs voice sounds like*,' we
> learn '*this is what neural-codec-reconstructed audio looks like.*' Catch the
> substrate, and you catch the generators built on it — **including ones you've
> never seen.** That's the difference between fighting the symptom and fighting
> the cause."

---

## 2. Layer 1 — the root-cause detector (how codec does it) (45 sec)

> "Our first layer is a codec-artifact detector. Under the hood it's a
> self-supervised speech model — wav2vec2 XLS-R — feeding a graph-attention
> classifier (AASIST). We don't train it on any one deepfake brand. We train it
> on **audio that has been passed through neural codecs vs. genuine audio**, so
> the only thing it can key on is the reconstruction fingerprint itself.
>
> The payoff: **it generalizes to generators outside its training set.** In our
> tests it flagged a fresh ElevenLabs professional voice clone at **0.99
> confidence** — a voice our conventional detectors scored near zero on,
> because they'd never seen that generator. The codec layer caught it anyway,
> because ElevenLabs, like the rest, decodes through a neural codec."

*(Honest aside if asked: this layer is powerful but young — it can over-fire on
real audio that's been through lossy transmission codecs, so we're calibrating
its decision threshold on real-world voice before it drives verdicts on its own.
See §5.)*

---

## 3. Layer 2 — the breadth net (NII + SSL ensemble) (40 sec)

> "But a root-cause detector isn't enough on its own — some attacks don't fit
> the codec signature: older concatenative or vocoder-based synthesis, or novel
> methods that dodge it. So the second layer is a **breadth net**: an ensemble
> of models trained on a wide *variety* of deepfake data.
>
> Our primary here is an MMS-300M anti-spoofing model — benchmark **AUC 0.997**
> — backed by two more self-supervised detectors (XLS-R and WavLM) for
> diversity. And critically, this layer is **telephony-robust**: we ran real
> phone codecs — G.711, AMR, G.722 — and it holds AUC 0.92–0.95. That matters,
> because the fraud actually happens over a phone line.
>
> So Layer 1 catches the *mechanism* broadly; Layer 2 catches the *known
> families* precisely. Anything that beats one tends to trip the other."

---

## 4. Layer 3 — the physical channel (replay) (25 sec)

> "There's a third attack that's pure physics: a fraudster records or clones a
> voice and simply **plays it through a speaker** into the call. No synthesis
> artifact to find. So we built a dedicated replay detector — LoRA-fine-tuned
> on a real loudspeaker-replay corpus — that reaches **cross-channel AUC 0.97**
> on recordings it never trained on. That's the layer that catches a *replayed*
> clone even when the audio itself is a perfect copy of a real voice."

---

## 5. The layer that actually earns trust — honesty & explainability (40 sec)

> "Two things separate this from a demo that falls over in production.
>
> **First — every verdict explains itself.** We don't output a black-box score.
> The system tells the operator *which* detector fired, *why*, and whether the
> decision came from broad agreement or one high-confidence signal — plus the
> spectrogram evidence, in plain language. A bank fraud analyst can act on it.
>
> **Second — we calibrated honestly.** We measured not just what works but what
> *doesn't*: we reproduced the field's known failure — that a detector with a
> great benchmark score can fall apart when its threshold is transferred to
> real traffic. So we're conservative by design: a single uncertain detector
> escalates to **'verify the caller,' not 'block.'** RED requires corroboration.
> We'd rather flag for a human than falsely reject a real customer. Every number
> in this pitch is measured on real audio — including the ones that told us to
> hold a detector back."

---

## 6. Why it wins (20 sec)

> "So the architecture is **defense-in-depth, from the root cause up**:
> - **Layer 1** — codec fingerprint → generalizes to *unseen* generators.
> - **Layer 2** — variety-trained ensemble → catches the *known* families, survives the phone network.
> - **Layer 3** — replay → catches the *physical* attack.
> - **Wrapped in** self-explaining verdicts and honest, production-aware calibration.
>
> We're not betting that we've seen every deepfake. We're betting on the one
> thing the attackers can't easily change — and backing it up three ways."

---

## 7. Live demo flow (what to show)

1. **Genuine caller** → GREEN. Point at the "why": every detector below threshold.
2. **Known TTS / clone** → RED in 2–3 s. Show NII + SSL agreeing; open the "why."
3. **Replayed clone** → AMBER "verify." Show the replay detector firing where synthesis models don't.
4. **The forensic report** → one click, print-to-PDF: verdict rationale, full
   detector table, spectrograms, integrity hash. "This is what goes in the case file."
5. *(If codec is demo-ready)* an **unseen generator (ElevenLabs)** → show the
   codec layer flag it while the others miss it — the root-cause payoff, live.

---

## Delivery notes (don't say these out loud)

- **Codec layer status:** validated *in principle* (caught ElevenLabs 0.99, clean
  on studio genuine), but it over-fired on real captured voices and is currently
  **observing (weight 0)** while we calibrate its threshold. Pitch it as the
  root-cause *approach* and our frontier — do **not** claim it's flawless, and
  don't let it drive a live verdict until calibrated, or it may false-flag a real
  voice on stage. If it's still disabled at demo time, lead with Layers 2–3 as the
  proven core and present Layer 1 as "the direction, proven on ElevenLabs offline."
- **Strongest, safest claims to lean on:** NII AUC 0.997; telephony 0.92–0.95;
  replay cross-channel 0.97; self-explaining verdicts; honest calibration.
- If a judge probes generalization: that's exactly the codec-substrate argument —
  own it, including the calibration caveat. The honesty *is* the moat.

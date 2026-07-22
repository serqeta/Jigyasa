"""
Global configuration for VoiceShield.
"""

# Audio pipeline configuration
SAMPLE_RATE = 16000
CHUNK_MS = 500
CHUNK_SAMPLES = int(SAMPLE_RATE * (CHUNK_MS / 1000.0))  # 8000
BUFFER_SECONDS = 10
BUFFER_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS  # 160000

# SNR Quality Gate thresholds (dB). Softened 2026-07-22 (10→NORMAL, 6→GREY)
# so live-mic speech in a normal room isn't over-gated to GREY mid-sentence.
SNR_NORMAL_DB = 10.0
SNR_GREY_DB = 6.0

# Voice-activity gate: forensic features are only meaningful on speech.
# A window with too few voiced frames is scored 0.0 (nothing to analyze)
# rather than letting silence masquerade as a synthetic artifact.
VAD_THRESHOLD_DB = -40.0  # frame energy above this counts as speech (fallback)
MIN_VOICED_RATIO = 0.10  # softened 2026-07-22 (was 0.20): brief pauses/breaths
# mid-speech no longer drop the newest chunk to GREY and reset the hysteresis

# Neural VAD (Silero, MIT ~2 MB). Robust in noise where the energy
# threshold fails — measured 2026-07-20: energy VAD reads pure noise as
# 100% voiced (false speech), Silero reads 0%. Falls back to energy VAD
# if the model can't load.
USE_SILERO_VAD = True

# Risk state mapping
SCORE_AMBER = 0.30
SCORE_RED = 0.70

# Stage 1 specific
STAGE1_WINDOW_SECONDS = 4
USE_FALLBACK_CLASSIFIER = False


def get_device() -> str:
    """Inference device: CUDA when available, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# Stage 2 ensemble
ENABLE_PHASE_PITCH_SCORER = True
ENABLE_REPLAY_DETECTION = True

# Primary detector (2026-07-20 evaluation, 496-clip multi-source benchmark:
# WaveFake GAN vocoders + ASVspoof2021-DF + In-the-Wild + generated clones):
# NII MMS-300M anti-deepfake — AUC 0.997, FPR@0.5 0.05, FNR@0.5 0.02.
# Weights converted from fairseq by scripts/convert_nii.py and validated
# against the reference implementation. License: CC BY-NC-SA 4.0 (research/
# hackathon use — see NOTICE.md before any commercial deployment).
ENABLE_NII_SCORER = True

# Learned replay / physical-access scorer: LoRA-fine-tuned wav2vec2 (NII
# backbone) trained on EchoFake + RawBoost augmentation
# (scripts/finetune_replay.py, models/replay_lora/). Validated CROSS-channel
# — trained only on EchoFake, held-out AUC 0.97 on our own unseen mic
# recordings (deduped: 0 false alarms, 3/4 replays; far-field is the weak
# spot). Covers consumer-device WIDEBAND replay only, NOT telephony.
# Loads only if models/replay_lora/ exists; see docs/REPLAY_FINDINGS.md.
ENABLE_LEARNED_REPLAY = True

# Codecfake countermeasure: XLS-R-300m + W2VAASIST co-trained CSAM checkpoint
# (arXiv:2405.04880). Targets neural-codec/vocoder synthesis artifacts — the
# family modern commercial TTS (ElevenLabs etc.) is built on — to cover
# generators the ASVspoof-trained ensemble misses. Loads only if
# models/codecfake/cotrain_w2v2aasist_CSAM/ exists. Runs at fusion weight 0.0
# initially (observed, not fused) pending validation on real ElevenLabs
# samples — see voiceshield/classifier/codec_scorer.py. LICENSE: CC BY-NC-ND
# (NonCommercial) — research/demo only.
# TEMPORARILY DISABLED (2026-07-22): loading the codec scorer (2nd XLS-R
# front-end + pickled W2VAASIST) destabilizes the process — native heap
# crash "free(): unaligned chunk" on startup/first inference. It was at
# weight 0 (observing) so verdicts are unaffected. Re-enable once the crash
# is isolated (candidate causes: 2nd wav2vec2 build conflict, the pickled-
# module load, or 5-model VRAM pressure on the 6 GB card).
ENABLE_CODEC_DETECTOR = False

# Pretrained anti-spoofing ensemble members (Hugging Face Hub, all 16 kHz
# mono, AutoModelForAudioClassification). spoof_label is the logit index
# whose softmax probability means "synthetic/spoofed" — polarity differs
# between hub models, so it is pinned per model here.
# Downloaded via scripts/download_model.py into HF_CACHE_DIR.
HF_SCORERS: dict[str, dict] = {
    # wav2vec2 XLS-R 300M fine-tuned for deepfake audio (ASVspoof2019 EER 0.04)
    "ssl": {
        "model_id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        "spoof_label": 1,
        # DISABLED 2026-07-22: reduced the ensemble to NII (synthesis) + replay
        # (physical). NII (AUC 0.997, telephony-robust) covers the synthesis
        # axis on its own — incl. ElevenLabs at 0.997 — so ssl is redundant.
        "enabled": False,
    },
    # Audio Spectrogram Transformer fine-tuned on ASVspoof 5.
    # DISABLED after fixture validation: outputs p(Spoof)≈1.0 for every
    # input (genuine speech, white noise, pure tones — logits ~±7.7 in
    # fp32), i.e. it does not generalize outside its training domain.
    # Keep the slot: swap in a better spectrogram checkpoint when found.
    "spec": {
        "model_id": "MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection",
        "spoof_label": 1,
        "enabled": False,
    },
    # WavLM-base fine-tuned on In-the-Wild deepfakes (domain diversity)
    "wavlm": {
        "model_id": "abhishtagatya/wavlm-base-960h-itw-deepfake",
        "spoof_label": 1,
        # DISABLED 2026-07-22: false-fired at 1.0 on genuine voices (a real
        # false-alarm liability), and NII already covers synthesis. Retired.
        "enabled": False,
    },
}
HF_CACHE_DIR = "models/hf"

# Risk fusion weights. Renormalized at runtime over the components that are
# actually available, so a missing model never silently drags the score down.
# Calibrated 2026-07-08 on the tests/fixtures/ set (LibriSpeech genuine,
# SpeechT5 TTS + voice clone): ssl and wavlm are the validated
# discriminators; AASIST-L false-positives on clean mic audio across every
# genuine domain tried, so it keeps only a small vote.
# Calibrated from the 2026-07-20 model evaluation (per-source AUCs in
# tasks/progress.md). Zero-weighted components are computed and displayed
# but do not influence the verdict:
#  - stage1 (AASIST-L): retired — AUC 0.33 on the benchmark, i.e. worse
#    than random on real audio; kept loadable for /v1 compatibility only.
#  - phase_pitch: AUC 0.49 — explainability display, not evidence.
#  - spec (AST): saturates at p(spoof)=1.0 on any input.
#  - replay: the LEARNED EchoFake replay scorer (2026-07-21). Detects the
#    loudspeaker-replay channel (orthogonal to synthesis). Modest weight +
#    AMBER-capped peak floor (see below) — a confident replay raises to
#    AMBER on its own; RED needs corroboration (e.g. replayed clone →
#    replay + synthesis both fire). Wideband only.
# NOTE (2026-07-22): codec (Codecfake W2VAASIST) was briefly promoted to a 2nd
# weight (0.25) + AMBER-cap after it caught an ElevenLabs clone at 0.99 — but it
# then FALSE-fired on real captured voices (0.97 → RED), a classic uncalibrated
# threshold-transfer failure (clean-fixture EER hid it). Reverted to OBSERVING
# (weight 0.0, no peak floor) pending calibration on a real-voice set — see
# docs / codec_scorer.py. codec still runs and shows in the panel; it just does
# not move the verdict.
# 2026-07-22: reduced to a two-detector ensemble on two orthogonal axes —
#   nii = synthesis (AUC 0.997, telephony-robust; catches ElevenLabs at 0.997)
#   replay = physical loudspeaker-replay (cross-channel AUC 0.97, AMBER-capped)
# ssl/wavlm retired (redundant / false-alarming), codec disabled (crash +
# uncalibrated). phase_pitch stays at weight 0 for the explainability cue only.
FUSION_WEIGHTS = {
    "nii": 0.70,
    "replay": 0.30,
    "phase_pitch": 0.0,  # explainability display, not evidence
}

# Peak-evidence rule: a weighted mean dilutes a single confident detector.
# The fused score is floored at factor × that component's score. Factors
# set from measured genuine-score quantiles (2026-07-20 benchmark):
#  - nii 0.8:   genuine p99 0.81 → floor 0.65 (< RED); fake median 0.997
#               → floor 0.80 (RED via hysteresis). Never instant-RED solo.
#  - replay 0.6: cross-channel AUC 0.97 on our recordings but small
#               validation → solo ceiling AMBER ("escalate/verify"); RED
#               needs corroboration (e.g. a replayed clone fires replay + nii).
PEAK_COMPONENTS = {"nii": 0.8, "replay": 0.6}

# Cascade screener component (Stage 1 slot). NII replaced AASIST-L after
# the benchmark; runner falls back to "stage1" for custom ensembles.
CASCADE_SCREENER = "nii"

# Temporal smoothing: the fused score fed to the state engine is the
# median of the last N speech chunks (history clears on silence). Kills
# single-chunk transient blips (26% of genuine clips showed one) at the
# cost of ≤ one chunk of detection delay.
SCORE_SMOOTHING_CHUNKS = 3

# Confidence gate (low side only): when the screener is confidently clean
# (< this) on a non-probe chunk, the diversity models are skipped — the
# dominant genuine-traffic case, where they cannot change the verdict.
# High screener scores always run the full ensemble so that instant-RED
# (fused ≥ 0.85) continues to require multi-model consensus.
CONFIDENCE_GATE_LOW = 0.15

# Evidence scaling: a scoring window that is mostly silence carries weak
# evidence — a fresh speech onset fills only a fraction of the 4 s window
# and the SSL models spike on such truncated speech. Scale the fused
# score linearly until the window is at least this voiced.
EVIDENCE_FULL_VOICED_RATIO = 0.5

# Cascade mode (live streaming): Stage 1 screens every chunk; the full
# Stage 2 ensemble engages when Stage 1 flags suspicion and disengages
# after sustained clean audio. Because AASIST-L can under-score voice
# clones (0.165 mean on the cloned fixture), a periodic deep probe runs
# the full ensemble even while screening, bounding the worst-case
# detection delay to PROBE_EVERY chunks.
CASCADE_TRIGGER = SCORE_AMBER  # stage1 score that engages Stage 2
CASCADE_COOLDOWN_CHUNKS = 4  # clean fused chunks before disengaging
CASCADE_PROBE_EVERY = 4  # deep-probe cadence (chunks) while screening

# AI-watermark detection (Meta AudioSeal, MIT) is NOT in the real-time
# pipeline — at ~150 ms/chunk it is too heavy for a signal that reads ~0 on
# every non-AudioSeal-watermarked source (all of today's commercial TTS).
# It is exposed as a standalone on-demand check (POST /v2/watermark/check)
# for a separate forensic UI. See classifier/watermark_scorer.py.

# Speaker-consistency tracking (ECAPA-TDNN, SpeechBrain, Apache-2.0).
# Detects the voice on the call changing mid-conversation (fraudster
# takeover / splice) — orthogonal to synthetic detection. Advisory signal
# surfaced in the timeline, not fused into the spoof score.
ENABLE_SPEAKER_DRIFT = True
SPEAKER_REF_WINDOWS = 4  # windows to fix the reference voice (stable mean)
SPEAKER_DRIFT_THRESHOLD = 0.65  # cosine distance: within-speaker p95 0.61, cross-speaker p5 0.70
SPEAKER_DRIFT_CONSEC = 2  # consecutive high-drift windows to latch "changed"

# Evidence export (privacy: written ONLY on explicit API request, never
# automatically — G10).
EVIDENCE_DIR = "evidence"

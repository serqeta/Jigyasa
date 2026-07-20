"""
Global configuration for VoiceShield.
"""

# Audio pipeline configuration
SAMPLE_RATE = 16000
CHUNK_MS = 500
CHUNK_SAMPLES = int(SAMPLE_RATE * (CHUNK_MS / 1000.0))  # 8000
BUFFER_SECONDS = 10
BUFFER_SAMPLES = SAMPLE_RATE * BUFFER_SECONDS  # 160000

# SNR Quality Gate thresholds (dB)
SNR_NORMAL_DB = 12.0
SNR_GREY_DB = 8.0

# Voice-activity gate: forensic features are only meaningful on speech.
# A window with too few voiced frames is scored 0.0 (nothing to analyze)
# rather than letting silence masquerade as a synthetic artifact.
VAD_THRESHOLD_DB = -40.0  # frame energy above this counts as speech (fallback)
MIN_VOICED_RATIO = 0.20  # require >=20% voiced frames in the window to score

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
        "enabled": True,
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
        "enabled": True,
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
#  - replay: awaiting real loudspeaker/call recordings for calibration.
FUSION_WEIGHTS = {
    "nii": 0.50,
    "ssl": 0.15,
    "wavlm": 0.15,
    "stage1": 0.0,
    "spec": 0.0,
    "phase_pitch": 0.0,
    "replay": 0.0,
}

# Peak-evidence rule: a weighted mean dilutes a single confident detector.
# The fused score is floored at factor × that component's score. Factors
# set from measured genuine-score quantiles (2026-07-20 benchmark):
#  - nii 0.8:   genuine p99 0.81 → floor 0.65 (< RED); fake median 0.997
#               → floor 0.80 (RED via hysteresis). Never instant-RED solo.
#  - ssl 0.8:   genuine p99 0.85 → floor 0.68 (< RED).
#  - wavlm 0.6: genuine p99 hits 0.999 (≈1% of real speakers) → solo
#               ceiling stays AMBER.
PEAK_COMPONENTS = {"nii": 0.8, "ssl": 0.8, "wavlm": 0.6}

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

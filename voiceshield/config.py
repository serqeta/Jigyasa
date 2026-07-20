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
VAD_THRESHOLD_DB = -40.0  # frame energy above this counts as speech
MIN_VOICED_RATIO = 0.20   # require >=20% voiced frames in the window to score

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
FUSION_WEIGHTS = {
    "stage1": 0.10,
    "ssl": 0.40,
    "spec": 0.15,
    "wavlm": 0.25,
    "phase_pitch": 0.10,
    # Quarantined 2026-07-08: validation on real audio showed the replay
    # detectors produce no discriminative signal yet (background detector
    # saturates on genuine speech; band-limit/codec thresholds never fire
    # on realistic channels). Still computed and displayed as experimental;
    # weight stays 0 until calibrated on real loudspeaker/call recordings.
    "replay": 0.0,
}

# Peak-evidence rule: a weighted mean dilutes a single confident detector
# (a clone scoring 0.91 on ssl must not average down to AMBER). The fused
# score is floored at factor × that component's score, per component.
# Factors encode measured trust (2026-07-08 validation):
#  - ssl 0.8: clean on genuine across domains; its confident hit can
#    carry a chunk to RED-range on its own.
#  - wavlm 0.75: the only member that catches ElevenLabs-class TTS
#    (0.999 digital) — 0.75 lets a sustained confident hit reach RED via
#    hysteresis (2 consecutive ≥ 0.70). CAUTION: wavlm false-fires on
#    ~10% of genuine LibriSpeech speakers; this demo calibration accepts
#    that risk because the demo speakers are validated wavlm-clean. For
#    production, drop to 0.6 (solo ceiling AMBER) or gate RED on
#    two-model consensus.
PEAK_COMPONENTS = {"ssl": 0.8, "wavlm": 0.75}

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
CASCADE_TRIGGER = SCORE_AMBER       # stage1 score that engages Stage 2
CASCADE_COOLDOWN_CHUNKS = 4         # clean fused chunks before disengaging
CASCADE_PROBE_EVERY = 4             # deep-probe cadence (chunks) while screening

# Evidence export (privacy: written ONLY on explicit API request, never
# automatically — G10).
EVIDENCE_DIR = "evidence"

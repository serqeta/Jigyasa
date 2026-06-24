"""
Global configuration for VoiceShield Stage 1.
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

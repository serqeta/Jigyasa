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

# Risk state mapping
SCORE_AMBER = 0.30
SCORE_RED = 0.70

# Stage 1 specific
STAGE1_WINDOW_SECONDS = 4
USE_FALLBACK_CLASSIFIER = False

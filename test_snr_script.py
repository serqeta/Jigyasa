import numpy as np

eps = np.finfo(np.float32).eps

def estimate_snr_perc(audio_1s):
    frame_length = 400
    hop_length = 160
    if len(audio_1s) < frame_length: return 0.0
    num_frames = 1 + (len(audio_1s) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio_1s,
        shape=(num_frames, frame_length),
        strides=(audio_1s.strides[0]*hop_length, audio_1s.strides[0])
    )
    energy = np.sum(frames ** 2, axis=1) / frame_length
    energy_db = 10 * np.log10(energy + eps)

    max_db = np.max(energy_db)
    if max_db < 10 * np.log10(eps * 100):
        return 0.0

    speech_db = np.percentile(energy_db, 95)
    noise_db = np.percentile(energy_db, 10)
    return max(0.0, float(speech_db - noise_db))

silence = np.zeros(16000, dtype=np.float32)
np.random.seed(42)
noise = (np.random.randn(16000) * 0.1).astype(np.float32)
t = np.linspace(0, 0.5, 8000, endpoint=False)
tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
sil = np.zeros(8000, dtype=np.float32)
clean_speech = np.concatenate((tone, sil))

print("Silence:", estimate_snr_perc(silence))
print("Noise:", estimate_snr_perc(noise))
print("Clean:", estimate_snr_perc(clean_speech))

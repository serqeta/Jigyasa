import librosa
from voiceshield.features.snr import estimate_snr
from voiceshield import config

audio, sr = librosa.load("/home/beast/Downloads/sample.mp3", sr=config.SAMPLE_RATE)
print(f"Audio length: {len(audio)/sr:.2f}s")

for i in range(int(len(audio)/sr)):
    chunk = audio[i*sr:(i+1)*sr]
    snr = estimate_snr(chunk)
    print(f"Second {i}-{i+1}: SNR = {snr:.1f} dB")

import librosa

from voiceshield import config
from voiceshield.classifier.fallback import FallbackScorer

scorer = FallbackScorer()
audio, sr = librosa.load("/home/beast/Downloads/sample.mp3", sr=config.SAMPLE_RATE)

for i in range(1, 6):
    chunk = audio[i*sr:(i+1)*sr]
    score = scorer.score(chunk)
    from voiceshield.features.flux import compute_spectral_flux
    from voiceshield.features.phase import compute_phase_discontinuity
    from voiceshield.features.pitch import compute_f0, compute_pitch_smoothness
    from voiceshield.features.subband import compute_subband_energy

    phase = compute_phase_discontinuity(chunk)
    f0 = compute_f0(chunk)
    pitch_var = compute_pitch_smoothness(f0)
    subband = compute_subband_energy(chunk)
    flux = compute_spectral_flux(chunk)

    print(f"Sec {i}: Score={score:.3f} | Phase={phase:.3f} | PitchVar={pitch_var:.3f} | Formant={subband.get('formants', 0):.1f} Artifact={subband.get('artifact', 0):.1f} | Flux={flux:.3f}")


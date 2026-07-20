import abc
from typing import Optional

import numpy as np
import soundfile as sf

from voiceshield import config
from voiceshield.audio.resample import resample_to_16k_mono

# sounddevice is only needed for live mic capture (MicSource) and pulls in
# the PortAudio system library. Import it lazily so FileSource — used by
# tests, CI, and file/stream serving — never depends on it. OSError =
# PortAudio missing; ImportError/ModuleNotFoundError = package not installed.
try:
    import sounddevice as sd
except (OSError, ImportError):
    sd = None


class AudioSource(abc.ABC):
    @abc.abstractmethod
    def read_chunk(self) -> np.ndarray:
        """
        Return float32, mono array of length CHUNK_SAMPLES.
        Values in [-1.0, 1.0].
        If EOF or no data, return zero-padded to exact length,
        or raise EOFError if completely done.
        """
        pass


class MicSource(AudioSource):
    def __init__(self, device: Optional[int] = None):
        if sd is None:
            raise RuntimeError("sounddevice is not available (PortAudio missing)")
        self.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=config.CHUNK_SAMPLES,
        )
        self.stream.start()

    def read_chunk(self) -> np.ndarray:
        data, overflowed = self.stream.read(config.CHUNK_SAMPLES)
        chunk = data[:, 0]
        if len(chunk) < config.CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, config.CHUNK_SAMPLES - len(chunk)))
        return chunk

    def close(self):
        self.stream.stop()
        self.stream.close()


class FileSource(AudioSource):
    def __init__(self, filepath: str):
        self.filepath = filepath
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
        # Resample to 16k mono
        self.audio = resample_to_16k_mono(data.T, sr)
        self.cursor = 0
        self.total_samples = len(self.audio)

    def read_chunk(self) -> np.ndarray:
        if self.cursor >= self.total_samples:
            raise EOFError("End of file")

        end = self.cursor + config.CHUNK_SAMPLES
        if end > self.total_samples:
            chunk = self.audio[self.cursor :]
            chunk = np.pad(chunk, (0, config.CHUNK_SAMPLES - len(chunk)))
        else:
            chunk = self.audio[self.cursor : end]

        self.cursor = end
        return chunk

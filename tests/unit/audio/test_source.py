import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from voiceshield import config
from voiceshield.audio.source import AudioSource, FileSource


def test_a1_1_audiosource_abstract():
    """TEST-A1.1: AudioSource is abstract."""
    with pytest.raises(TypeError):
        AudioSource()


def test_a1_3_filesource_yields_padded_chunks():
    """TEST-A1.3: FileSource chunks and zero padding."""
    # Create a 1.7 second file at 16kHz mono (1.7 * 16000 = 27200 samples)
    # Expected chunks: 27200 / 8000 = 3 full chunks + 1 partial chunk (3200 samples) -> padded to 8000
    n_samples = 27200
    data = np.random.randn(n_samples).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filepath = f.name

    try:
        sf.write(filepath, data, 16000)

        source = FileSource(filepath)
        chunks = []
        while True:
            try:
                chunks.append(source.read_chunk())
            except EOFError:
                break

        assert len(chunks) == 4, "Should yield exactly 4 chunks"
        for chunk in chunks:
            assert chunk.shape == (config.CHUNK_SAMPLES,)
            assert chunk.dtype == np.float32

        # Check padding on the last chunk
        assert np.all(chunks[3][3200:] == 0.0), "Last chunk should be zero padded"
    finally:
        os.remove(filepath)

import numpy as np

from voiceshield import config
from voiceshield.audio.buffer import RollingBuffer


def test_b2_1_buffer_capacity():
    """TEST-B2.1: Buffer correctly wraps around and maintains max size."""
    buf = RollingBuffer()
    chunk = np.ones(config.CHUNK_SAMPLES, dtype=np.float32)

    # Push 20 chunks (160k samples max / 8k per chunk = 20)
    for _ in range(20):
        buf.push(chunk)

    assert buf.size() == 160000
    assert buf.full() is True

    # 21st push
    chunk2 = np.ones(config.CHUNK_SAMPLES, dtype=np.float32) * 2
    buf.push(chunk2)
    assert buf.size() == 160000

def test_b2_2_timestamps():
    """TEST-B2.2: Timestamps are monotonic and contiguous."""
    buf = RollingBuffer()
    chunk = np.ones(config.CHUNK_SAMPLES, dtype=np.float32)

    prev_end = 0.0
    for i in range(5):
        idx, t_start, t_end = buf.push(chunk)
        assert idx == i
        assert np.isclose(t_start, prev_end)
        assert np.isclose(t_end, t_start + (config.CHUNK_MS / 1000.0))
        prev_end = t_end

def test_b2_3_latest_seconds():
    """TEST-B2.3: latest_seconds returns correct sizes."""
    buf = RollingBuffer()
    chunk = np.ones(config.CHUNK_SAMPLES, dtype=np.float32)

    # Push 2 seconds (4 chunks)
    for _ in range(4):
        buf.push(chunk)

    # Request 4 seconds -> should return 2 seconds (32000 samples)
    out = buf.latest_seconds(4.0)
    assert len(out) == 32000

    # Push another 10 seconds to wrap
    for i in range(20):
        c = np.ones(config.CHUNK_SAMPLES, dtype=np.float32) * i
        buf.push(c)

    out2 = buf.latest_seconds(2.0)
    assert len(out2) == 32000

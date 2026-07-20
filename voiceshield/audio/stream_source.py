"""
StreamSource: an AudioSource fed by an external producer (browser
microphone over WebSocket). The producer pushes float32 mono 16 kHz
samples via feed(); read_chunk() assembles 500 ms chunks.

When the producer stalls (mic muted, tab closed, network hiccup) the
source emits silence chunks so the pipeline keeps running and the
dashboard correctly shows GREY instead of freezing.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from voiceshield import config
from voiceshield.audio.source import AudioSource


class StreamSource(AudioSource):
    STARVATION_TIMEOUT_S = 0.6  # a bit over one chunk period

    def __init__(self) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._pending = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._closed = False

    def feed(self, samples: np.ndarray) -> None:
        """Producer thread: push float32 mono samples at SAMPLE_RATE."""
        if samples.size:
            self._q.put(samples.astype(np.float32, copy=False))

    def close(self) -> None:
        self._closed = True
        self._q.put(np.array([], dtype=np.float32))  # unblock reader

    def read_chunk(self) -> np.ndarray:
        with self._lock:
            while len(self._pending) < config.CHUNK_SAMPLES:
                if self._closed:
                    raise EOFError("stream closed")
                try:
                    samples = self._q.get(timeout=self.STARVATION_TIMEOUT_S)
                except queue.Empty:
                    # producer stalled → pad the shortfall with silence
                    shortfall = config.CHUNK_SAMPLES - len(self._pending)
                    self._pending = np.concatenate(
                        [self._pending, np.zeros(shortfall, dtype=np.float32)]
                    )
                    break
                self._pending = np.concatenate([self._pending, samples])

            chunk = self._pending[: config.CHUNK_SAMPLES]
            self._pending = self._pending[config.CHUNK_SAMPLES :]
            return chunk

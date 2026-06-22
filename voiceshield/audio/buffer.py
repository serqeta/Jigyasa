import numpy as np

from voiceshield import config


class RollingBuffer:
    def __init__(self, capacity: int = config.BUFFER_SAMPLES):
        self.capacity = capacity
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.current_size = 0
        self.write_pos = 0

    def push(self, chunk: np.ndarray) -> tuple[int, float, float]:
        """
        Push a chunk and return (chunk_index, t_start, t_end).
        Assumption: chunk is exactly CHUNK_SAMPLES.
        """
        assert len(chunk) == config.CHUNK_SAMPLES, f"Chunk must be {config.CHUNK_SAMPLES}"

        chunk_index = self.current_size // config.CHUNK_SAMPLES
        t_start = chunk_index * (config.CHUNK_MS / 1000.0)
        t_end = t_start + (config.CHUNK_MS / 1000.0)

        n = len(chunk)
        if self.write_pos + n <= self.capacity:
            self.buffer[self.write_pos : self.write_pos + n] = chunk
            self.write_pos = (self.write_pos + n) % self.capacity
        else:
            # Handle wrap-around
            part1 = self.capacity - self.write_pos
            part2 = n - part1
            self.buffer[self.write_pos :] = chunk[:part1]
            self.buffer[:part2] = chunk[part1:]
            self.write_pos = part2

        self.current_size += n

        return chunk_index, t_start, t_end

    def size(self) -> int:
        return min(self.current_size, self.capacity)

    def full(self) -> bool:
        return self.current_size >= self.capacity

    def latest(self, n_samples: int) -> np.ndarray:
        """Return the latest n_samples."""
        available = self.size()
        to_return = min(n_samples, available)
        if to_return == 0:
            return np.array([], dtype=np.float32)

        start_pos = (self.write_pos - to_return) % self.capacity
        if start_pos < self.write_pos:
            return self.buffer[start_pos : self.write_pos].copy()
        else:
            part1 = self.buffer[start_pos:]
            part2 = self.buffer[: self.write_pos]
            return np.concatenate((part1, part2))

    def latest_seconds(self, n_seconds: float) -> np.ndarray:
        return self.latest(int(n_seconds * config.SAMPLE_RATE))

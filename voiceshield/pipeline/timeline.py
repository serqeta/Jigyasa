from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TimelineEntry:
    time: float
    score: float
    state: str  # "green" | "amber" | "red" | "grey"
    snr_db: float
    top_artifact: str | None
    first_amber_t: float | None
    first_red_t: float | None
    speech_active: bool = True
    voiced_ratio: float = 0.0
    component_scores: dict[str, float] | None = None
    replay: dict[str, Any] | None = None
    stage2_active: bool = True
    speaker_drift: float = 0.0
    speaker_changed: bool = False
    spec_linear: list[list[float]] | None = None
    spec_mel: list[list[float]] | None = None
    spec_cqt: list[list[float]] | None = None
    pitch_contour: list[float | None] | None = None
    phase_contour: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Timeline:
    """Rolling ring of the last CAPACITY TimelineEntry objects (10 s @ 500 ms)."""

    CAPACITY = 20

    def __init__(self) -> None:
        self._ring: deque[TimelineEntry] = deque(maxlen=self.CAPACITY)

    def append(self, entry: TimelineEntry) -> None:
        self._ring.append(entry)

    def entries(self) -> list[TimelineEntry]:
        """Return all entries oldest-first."""
        return list(self._ring)

    def latest(self) -> TimelineEntry | None:
        return self._ring[-1] if self._ring else None

    def to_json(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._ring]

    def __len__(self) -> int:
        return len(self._ring)

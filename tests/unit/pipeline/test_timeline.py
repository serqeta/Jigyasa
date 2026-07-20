"""TEST-T6.1, TEST-T6.2, TEST-T6.3: Timeline dataclass, ring, and JSON."""

from voiceshield.pipeline.timeline import Timeline, TimelineEntry


def _entry(t: float) -> TimelineEntry:
    return TimelineEntry(
        time=t,
        score=0.1,
        state="green",
        snr_db=15.0,
        top_artifact=None,
        first_amber_t=None,
        first_red_t=None,
    )


def test_entry_fields():
    """TEST-T6.1: TimelineEntry has all documented fields."""
    e = _entry(0.5)
    assert hasattr(e, "time")
    assert hasattr(e, "score")
    assert hasattr(e, "state")
    assert hasattr(e, "snr_db")
    assert hasattr(e, "top_artifact")
    assert hasattr(e, "first_amber_t")
    assert hasattr(e, "first_red_t")


def test_ring_drops_oldest():
    """TEST-T6.2: appending the 21st entry drops entry 0."""
    tl = Timeline()
    for i in range(21):
        tl.append(_entry(float(i) * 0.5))
    assert len(tl) == Timeline.CAPACITY
    entries = tl.entries()
    assert entries[0].time == 0.5  # index 1 is now the oldest


def test_ring_capacity_after_many():
    tl = Timeline()
    for i in range(50):
        tl.append(_entry(float(i)))
    assert len(tl) == Timeline.CAPACITY


def test_latest_returns_last():
    tl = Timeline()
    tl.append(_entry(1.0))
    tl.append(_entry(2.0))
    assert tl.latest().time == 2.0


def test_latest_empty_returns_none():
    assert Timeline().latest() is None


def test_to_json():
    """TEST-T6.3: JSON output has expected structure."""
    tl = Timeline()
    tl.append(_entry(0.5))
    j = tl.to_json()
    assert isinstance(j, list)
    assert j[0]["time"] == 0.5
    assert "score" in j[0]
    assert "state" in j[0]
    assert "snr_db" in j[0]
    assert "top_artifact" in j[0]


def test_visualization_fields():
    """Verify that TimelineEntry has all visualization fields."""
    e = _entry(0.5)
    assert hasattr(e, "spec_linear")
    assert hasattr(e, "spec_mel")
    assert hasattr(e, "spec_cqt")
    assert hasattr(e, "pitch_contour")
    assert hasattr(e, "phase_contour")

    # Verify we can set them to lists
    e.spec_linear = [[-10.0, -20.0]]
    e.spec_mel = [[-5.0, -15.0]]
    e.spec_cqt = [[-2.0, -12.0]]
    e.pitch_contour = [220.0, None, 222.0]
    e.phase_contour = [0.1, 0.2]

    d = e.to_dict()
    assert d["spec_linear"] == [[-10.0, -20.0]]
    assert d["spec_mel"] == [[-5.0, -15.0]]
    assert d["spec_cqt"] == [[-2.0, -12.0]]
    assert d["pitch_contour"] == [220.0, None, 222.0]
    assert d["phase_contour"] == [0.1, 0.2]

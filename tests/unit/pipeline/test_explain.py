"""
Verdict-explanation tests. The rationale must faithfully track the same
mechanism state_engine.fuse_scores uses — consensus vs. the peak-evidence
floor — and name the correct driver.
"""

from voiceshield.pipeline.explain import explain_verdict
from voiceshield.pipeline.state_engine import fuse_scores


def test_none_when_no_components():
    assert explain_verdict({}, 0.0, "grey") is None


def test_green_is_benign():
    ex = explain_verdict({"nii": 0.05, "ssl": 0.04}, 0.05, "green")
    assert ex["driver"] in ("nii", "ssl")
    assert "GREEN" in ex["headline"]
    assert ex["reasons"] == ["every detector scored below the suspicion threshold"]


def test_peak_driven_single_confident_detector():
    # NII alone is highly confident; the quieter models would average it down,
    # but the peak-evidence rule floors the verdict — the explanation must say so.
    scores = {"nii": 0.95, "ssl": 0.05, "wavlm": 0.05, "replay": 0.05}
    ex = explain_verdict(scores, fuse_scores(scores), "red")
    assert ex["mechanism"] == "peak"
    assert ex["driver"] == "nii"
    assert any("peak-evidence rule" in r for r in ex["reasons"])
    assert "RED" in ex["headline"]


def test_consensus_driven_when_models_agree():
    # Several models moderately high, none flooring via peak → consensus.
    scores = {"nii": 0.35, "ssl": 0.4, "wavlm": 0.38}
    ex = explain_verdict(scores, fuse_scores(scores), "amber")
    assert ex["mechanism"] == "consensus"
    assert any("consensus" in r for r in ex["reasons"])


def test_replay_driver_phrased_as_replay():
    # A replayed genuine clip: replay fires, synthesis models quiet.
    scores = {"nii": 0.02, "ssl": 0.03, "wavlm": 0.02, "replay": 0.76}
    ex = explain_verdict(scores, fuse_scores(scores), "amber")
    assert ex["driver"] == "replay"
    assert "replay" in ex["headline"].lower()
    assert any("loudspeaker" in r for r in ex["reasons"])


def test_contributions_mirror_fuse_scores():
    # The additive contributions must sum to the renormalized weighted mean
    # (which equals fuse_scores whenever no peak floor is higher).
    scores = {"nii": 0.35, "ssl": 0.4, "wavlm": 0.38}
    ex = explain_verdict(scores, fuse_scores(scores), "amber")
    weighted_mean = sum(c["contribution"] for c in ex["contributions"])
    assert abs(weighted_mean - fuse_scores(scores)) < 0.01
    # sorted by contribution descending
    contribs = [c["contribution"] for c in ex["contributions"]]
    assert contribs == sorted(contribs, reverse=True)


def test_cue_surfaced_from_top_artifact():
    scores = {"nii": 0.9}
    ex = explain_verdict(
        scores, fuse_scores(scores), "red", top_artifact="over_smooth_pitch_contour"
    )
    assert ex["cue"] is not None and "pitch" in ex["cue"]
    assert any("pitch" in r for r in ex["reasons"])


def test_speaker_change_advisory_appended():
    scores = {"nii": 0.9}
    ex = explain_verdict(scores, fuse_scores(scores), "red", speaker_changed=True)
    assert any("speaker appears to have changed" in r for r in ex["reasons"])

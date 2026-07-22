"""
Ensemble composition guard (2026-07-22): reduced to a two-detector ensemble —
NII (synthesis) + replay (physical). ssl/wavlm retired, codec disabled. These
tests lock that composition so it can't silently drift back.
"""

from voiceshield import config


def test_only_nii_and_replay_are_peak_floored():
    assert config.FUSION_WEIGHTS["nii"] == 0.70
    assert config.FUSION_WEIGHTS["replay"] == 0.30
    assert set(config.PEAK_COMPONENTS) == {"nii", "replay"}


def test_codec_disabled_and_not_fused():
    assert config.ENABLE_CODEC_DETECTOR is False
    assert "codec" not in config.FUSION_WEIGHTS
    assert "codec" not in config.PEAK_COMPONENTS


def test_ssl_and_wavlm_retired():
    assert config.HF_SCORERS["ssl"]["enabled"] is False
    assert config.HF_SCORERS["wavlm"]["enabled"] is False


def test_display_labels_still_present():
    # Labels are kept for any component that may appear in a payload.
    from voiceshield.pipeline.explain import _COMPONENT_LABEL

    for name in ("nii", "replay"):
        assert name in _COMPONENT_LABEL

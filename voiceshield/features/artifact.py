"""
Determine the top (most anomalous) forensic artifact name for a chunk.
Used to populate TimelineEntry.top_artifact.
"""

# Expected "normal" baselines and scale for each feature scalar.
# A value far above the baseline is suspicious.
_BASELINES: dict[str, tuple[float, float]] = {
    # (expected_normal_value, scale)
    "phase_discontinuity": (0.05, 0.15),
    "pitch_smoothness_inv": (0.2, 0.3),
    "artifact_ratio": (0.2, 0.5),
    "flux_variance_inv": (0.3, 0.3),
}

# Human-readable names surfaced in the dashboard
_DISPLAY_NAMES: dict[str, str] = {
    "phase_discontinuity": "phase_discontinuity",
    "pitch_smoothness_inv": "over_smooth_pitch_contour",
    "artifact_ratio": "vocoder_artifact_band",
    "flux_variance_inv": "over_smooth_spectral_flux",
}


def top_artifact_name(features: dict[str, float]) -> str | None:
    """
    Return the display name of the feature with the highest z-score
    relative to the expected normal baseline, or None if all are normal.
    """
    best_key: str | None = None
    best_z = 1.0  # only flag features more than 1 sigma above baseline

    for key, (baseline, scale) in _BASELINES.items():
        value = features.get(key)
        if value is None:
            continue
        z = (value - baseline) / max(scale, 1e-6)
        if z > best_z:
            best_z = z
            best_key = key

    if best_key is None:
        return None
    return _DISPLAY_NAMES[best_key]

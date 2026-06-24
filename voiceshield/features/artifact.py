"""
Determine the top (most anomalous) forensic artifact name for a chunk.
Used to populate TimelineEntry.top_artifact.
"""

# Expected "normal" baselines and scale for each suspicion-feature scalar.
# A value far above the baseline is suspicious.
_BASELINES: dict[str, tuple[float, float]] = {
    # (expected_normal_value, scale)
    "phase_discontinuity": (0.25, 0.20),
    "pitch_smoothness_inv": (0.05, 0.30),
    "artifact_ratio": (0.05, 0.40),
    "flux_smoothness": (0.05, 0.40),
    "dynamics_inv": (0.85, 0.10),
}

# Human-readable names surfaced in the dashboard
_DISPLAY_NAMES: dict[str, str] = {
    "phase_discontinuity": "phase_discontinuity",
    "pitch_smoothness_inv": "over_smooth_pitch_contour",
    "artifact_ratio": "vocoder_artifact_band",
    "flux_smoothness": "over_smooth_spectral_flux",
    "dynamics_inv": "static_spectral_dynamics",
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

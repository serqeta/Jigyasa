"""
Verdict explanation — a faithful, deterministic decomposition of *why* a
chunk was flagged.

This is not a post-hoc guess. It recomputes the exact weighting that
`state_engine.fuse_scores` uses (renormalized weighted mean, floored by the
per-component peak-evidence rule) and reports:

  - which mechanism decided the verdict: broad *consensus* across models, or
    a single high-confidence validated detector *flooring* the score via the
    peak-evidence rule;
  - which component was the primary driver, and its additive share of the
    fused score;
  - the human-readable reasons (per-model, plus the top waveform cue).

Component scores answer "what did each model say?"; this answers "why is the
verdict what it is?" — the piece a non-expert operator actually needs.
"""

from __future__ import annotations

from typing import Any

from voiceshield import config

# One plain sentence per component, used when that component reads suspicious.
_COMPONENT_WHY: dict[str, str] = {
    "nii": "the primary anti-deepfake model (MMS-300M) matched synthetic-speech signatures",
    "ssl": "the XLS-R self-supervised deepfake classifier flagged synthetic artifacts",
    "wavlm": "the WavLM in-the-wild deepfake model flagged synthetic artifacts",
    "spec": "the spectrogram transformer flagged synthetic artifacts",
    "stage1": "the AASIST screener flagged anti-spoofing artifacts",
    "replay": "the replay detector found the audio consistent with loudspeaker "
    "playback (a recording being replayed, not a live voice)",
    "phase_pitch": "the rule-based check found vocoder phase/pitch artifacts",
    "codec": "the codec-artifact detector (Codecfake) matched neural-codec/vocoder "
    "synthesis signatures typical of modern TTS voice cloning",
}

_COMPONENT_LABEL: dict[str, str] = {
    "nii": "MMS-300M (NII)",
    "ssl": "XLS-R 300M",
    "wavlm": "WavLM",
    "spec": "AST",
    "stage1": "AASIST-L",
    "replay": "Replay (EchoFake LoRA)",
    "phase_pitch": "Phase / Pitch",
    "codec": "Codec (Codecfake W2VAASIST)",
}

# Plain-language phrasing for the top_artifact display names.
_CUE_WHY: dict[str, str] = {
    "phase_discontinuity": "phase discontinuities characteristic of waveform synthesis",
    "over_smooth_pitch_contour": "an unnaturally smooth pitch contour (a common vocoder tell)",
    "vocoder_artifact_band": "excess energy in the vocoder artifact band",
    "over_smooth_spectral_flux": "over-smooth spectral flux — missing the natural micro-variation of a live voice",
    "static_spectral_dynamics": "static spectral dynamics — too little of the natural movement of live speech",
}


def _label(name: str) -> str:
    return _COMPONENT_LABEL.get(name, name)


def explain_verdict(
    component_scores: dict[str, float],
    score: float,
    state: str,
    top_artifact: str | None = None,
    speaker_changed: bool = False,
) -> dict[str, Any] | None:
    """
    Return a structured rationale for a chunk's verdict, or ``None`` when
    there is nothing to explain (no components scored — e.g. silence).

    The ``contributions`` list mirrors ``fuse_scores`` exactly: each entry's
    ``contribution`` is that component's additive share of the renormalized
    weighted mean, and ``peak_floor`` is the value it would floor the verdict
    to under the peak-evidence rule.
    """
    if not component_scores:
        return None

    # --- reproduce fuse_scores' weighting exactly ---
    configured = config.FUSION_WEIGHTS
    default_w = sum(configured.values()) / max(len(configured), 1)
    weights = {k: configured.get(k, default_w) for k in component_scores}
    total = sum(weights.values())

    contributions: list[dict[str, Any]] = []
    for name, s in component_scores.items():
        share = (weights[name] * s / total) if total > 0 else s / len(component_scores)
        peak_floor = config.PEAK_COMPONENTS.get(name, 0.0) * s
        contributions.append(
            {
                "name": name,
                "label": _label(name),
                "score": round(s, 3),
                "weight": round(weights[name], 3),
                "contribution": round(share, 3),
                "peak_floor": round(peak_floor, 3),
            }
        )

    # Which mechanism produced the number the operator sees?
    weighted_mean = sum(c["contribution"] for c in contributions)
    peak_c = max(contributions, key=lambda c: c["peak_floor"])
    mechanism = "peak" if peak_c["peak_floor"] > weighted_mean + 1e-9 else "consensus"
    driver = peak_c if mechanism == "peak" else max(contributions, key=lambda c: c["contribution"])

    is_replay = driver["name"] == "replay"
    kind = "loudspeaker-replay signature" if is_replay else "synthetic-speech signatures"

    if state == "red":
        headline = f"Flagged RED — {kind} ({driver['label']} {driver['score']:.2f})"
    elif state == "amber":
        headline = f"Flagged AMBER — possible {kind} ({driver['label']} {driver['score']:.2f})"
    elif state == "grey":
        headline = "Audio quality insufficient to judge."
    else:
        headline = "Verdict GREEN — no synthetic or replay signatures detected."

    reasons: list[str] = []
    if state in ("amber", "red"):
        reasons.append(_COMPONENT_WHY.get(driver["name"], f"{driver['label']} scored high"))
        # Corroborating detectors (any *other* component also above suspicion).
        for c in sorted(contributions, key=lambda c: c["score"], reverse=True):
            if c["name"] != driver["name"] and c["score"] >= config.SCORE_AMBER:
                reasons.append(
                    "corroborated because "
                    + _COMPONENT_WHY.get(c["name"], f"{c['label']} also scored high")
                )
        if top_artifact and top_artifact in _CUE_WHY:
            reasons.append("the waveform shows " + _CUE_WHY[top_artifact])
        if mechanism == "peak":
            reasons.append(
                f"the decision rests on one high-confidence validated detector "
                f"({driver['label']}); the peak-evidence rule keeps a single strong, "
                f"trusted signal from being averaged away by quieter models"
            )
        else:
            n_susp = sum(1 for c in contributions if c["score"] >= config.SCORE_AMBER)
            if n_susp >= 2:
                reasons.append(f"{n_susp} independent models agree, so the verdict is a consensus")
    else:
        reasons.append("every detector scored below the suspicion threshold")

    if speaker_changed:
        reasons.append("note: the speaker appears to have changed mid-call (advisory, not scored)")

    return {
        "headline": headline,
        "state": state,
        "score": round(score, 3),
        "mechanism": mechanism,  # "peak" | "consensus"
        "driver": driver["name"],
        "reasons": reasons,
        "cue": _CUE_WHY.get(top_artifact) if top_artifact else None,
        "contributions": sorted(contributions, key=lambda c: c["contribution"], reverse=True),
    }

"""
Forensic report renderer.

Fills the UCO-Bank forensic report template (report_template.html) with the
REAL values from an evidence package (evidence.json + the two PNGs produced
by export.py). Images are embedded as base64 data URIs so the output is a
single self-contained HTML file, ready to open and Print → Save as PDF.

The "determination" (the seal, finding, reasons and contribution table) is
taken from the PEAK-severity chunk of the analysed window — the worst moment
of the interaction — not merely the last chunk.
"""

from __future__ import annotations

import base64
import html
import os
import subprocess
from typing import Any

from voiceshield import config

_TEMPLATE = os.path.join(os.path.dirname(__file__), "report_template.html")

_STATE_TEXT = {"red": "RED", "amber": "AMBER", "green": "GREEN", "grey": "GREY"}
_STATE_RANK = {"grey": 0, "green": 1, "amber": 2, "red": 3}
_ACTION = {
    "red": "Step-up verification",
    "amber": "Verify caller identity",
    "green": "No action — proceed",
    "grey": "Request better audio",
}
_MECHANISM_TEXT = {"peak": "Peak evidence", "consensus": "Consensus"}

# Detector short name + model description for the ensemble table.
_SHORT = {
    "nii": "NII",
    "ssl": "SSL",
    "wavlm": "WavLM",
    "replay": "Replay",
    "phase_pitch": "Phase / Pitch",
    "spec": "AST",
    "stage1": "AASIST",
}
_MODEL_DESC = {
    "nii": "MMS-300M anti-deepfake",
    "ssl": "XLS-R 300M deepfake",
    "wavlm": "WavLM in-the-wild",
    "replay": "EchoFake LoRA (wav2vec2)",
    "phase_pitch": "Rule-based (explainability)",
    "spec": "AST spectrogram transformer",
    "stage1": "AASIST-L",
}
_MODEL_VERSIONS = (
    "nii=mms-300m-anti-deepfake · ssl=Gustking/wav2vec2-xlsr-deepfake · "
    "wavlm=abhishtagatya/wavlm-base-960h-itw · replay=echofake-lora"
)


def _git_commit() -> str:
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(__file__),
        )
        return f"git {h.stdout.strip()}" if h.returncode == 0 else "git (unknown)"
    except Exception:
        return "git (unknown)"


def _img_tag(path: str, alt: str) -> str:
    """A base64-embedded <img> (self-contained), or a placeholder box if absent."""
    if not path or not os.path.exists(path):
        return (
            f'<div class="plate" style="display:flex;align-items:center;'
            f'justify-content:center;background:#0b0b1a;color:#8a93a3;'
            f'font-size:9px">{html.escape(alt)} unavailable</div>'
        )
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<img class="plate" src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" />'


def _snr_summary(timeline: list[dict[str, Any]]) -> str:
    vals = [e["snr_db"] for e in timeline if e.get("snr_db", 0) > 0]
    if not vals:
        return "—"
    mean = sum(vals) / len(vals)
    label = "Normal" if mean >= 12 else "Reduced" if mean >= 8 else "Low"
    return f"{mean:.1f} dB ({label})"


def _fmt_t(v: Any) -> str:
    return f"{float(v):04.1f} s" if v is not None else "—"


def _determination(timeline: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The peak-severity scored chunk (the worst moment of the window).

    Severity is primary, but among equally-severe chunks we prefer the one
    that ran the FULL ensemble (most components). In cascade/live mode a
    clean chunk runs only the NII screener, so a naive peak pick would show
    a single-detector breakdown; a periodic deep-probe chunk carries every
    detector and is the representative view of the verdict.
    """
    scored = [e for e in timeline if e.get("state") != "grey" and e.get("explanation")]
    if not scored:
        return None
    return max(
        scored,
        key=lambda e: (
            _STATE_RANK.get(e["state"], 0),
            len(e.get("component_scores") or {}),
            e.get("score", 0.0),
        ),
    )


def _reasons_li(reasons: list[str]) -> str:
    out = []
    for i, r in enumerate(reasons):
        cls = ' class="primary"' if i == 0 else ""
        out.append(f"      <li{cls}>{html.escape(r)}</li>")
    return "\n".join(out)


def _contribution_rows(contributions: list[dict[str, Any]], driver: str) -> str:
    rows = []
    for c in contributions:
        name = c["name"]
        muted = c.get("weight", 0) == 0
        cls = ' class="driver"' if name == driver else ""
        mw = " muted-w" if muted else ""
        peak = "—" if muted else f"{c.get('peak_floor', 0):.3f}"
        short = _SHORT.get(name, name)
        desc = _MODEL_DESC.get(name, name)
        rows.append(
            f'        <tr{cls}>'
            f'<td class="{mw.strip()}">{html.escape(short)}</td>'
            f'<td class="{mw.strip()}">{html.escape(desc)}</td>'
            f'<td class="num{mw}">{c.get("score", 0):.3f}</td>'
            f'<td class="num{mw}">{c.get("weight", 0):.3f}</td>'
            f'<td class="num{mw}">{c.get("contribution", 0):.3f}</td>'
            f'<td class="num{mw}">{peak}</td></tr>'
        )
    return "\n".join(rows)


def _timeline_rows(timeline: list[dict[str, Any]]) -> str:
    rows = []
    for e in timeline:
        st = e.get("state", "grey")
        cs = e.get("component_scores") or {}
        nii = f'{cs["nii"]:.2f}' if "nii" in cs else "—"
        rep = f'{cs["replay"]:.2f}' if "replay" in cs else "—"
        cue = html.escape(str(e["top_artifact"]).replace("_", " ")) if e.get("top_artifact") else "—"
        rows.append(
            f'        <tr><td class="num">{e.get("time", 0):.1f}</td>'
            f'<td><span class="pill {st}">{st.capitalize()}</span></td>'
            f'<td class="num">{e.get("score", 0):.2f}</td>'
            f'<td class="num">{nii}</td><td class="num">{rep}</td>'
            f'<td class="num">{e.get("snr_db", 0):.1f}</td>'
            f'<td class="num">{e.get("voiced_ratio", 0):.2f}</td>'
            f'<td>{cue}</td>'
            f'<td class="num">{e.get("speaker_drift", 0):.2f}</td></tr>'
        )
    return "\n".join(rows)


def render_report(
    manifest: dict[str, Any],
    package_dir: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a self-contained HTML forensic report filled from `manifest`."""
    extra = extra or {}
    timeline = manifest.get("timeline", []) or []
    det = _determination(timeline)
    last = timeline[-1] if timeline else {}

    if det is not None:
        ex = det["explanation"]
        state = ex.get("state", det.get("state", "grey"))
        contributions = ex.get("contributions", [])
        driver = ex.get("driver", "")
        driver_label = next(
            (c["label"] for c in contributions if c["name"] == driver), driver or "—"
        )
        headline = ex.get("headline", "")
        mechanism = _MECHANISM_TEXT.get(ex.get("mechanism", ""), ex.get("mechanism", "—"))
        reasons = ex.get("reasons", [])
        score = ex.get("score", det.get("score", 0.0))
        top_artifact = (
            str(det.get("top_artifact")).replace("_", " ") if det.get("top_artifact") else "None"
        )
        speaker = "Detected" if det.get("speaker_changed") else "Not detected"
    else:
        # No scored chunk — genuine/silent throughout.
        state, contributions, driver, driver_label = "green", [], "", "—"
        headline = "No synthetic or replay signatures detected across the analysed window."
        mechanism, reasons, score = "—", ["Every detector scored below the suspicion threshold."], 0.0
        top_artifact, speaker = "None", "Not detected"

    fw = config.FUSION_WEIGHTS
    fw_caption = " · ".join(
        f"{k} {fw[k]:.2f}" for k in ("nii", "replay", "ssl", "wavlm", "phase_pitch") if k in fw
    )
    config_snapshot = (
        f"fusion_weights={{nii:{fw.get('nii', 0):.2f},ssl:{fw.get('ssl', 0):.2f},"
        f"wavlm:{fw.get('wavlm', 0):.2f},replay:{fw.get('replay', 0):.2f}}} · "
        f"thresholds={{amber:{config.SCORE_AMBER:.2f},red:{config.SCORE_RED:.2f},instant_red:0.85}} · "
        f"smoothing={config.SCORE_SMOOTHING_CHUNKS} chunks · window={config.STAGE1_WINDOW_SECONDS} s"
    )

    sha = manifest.get("audio_sha256", "")
    values = {
        "REPORT_ID": extra.get("report_id", "VS-" + sha[:8]),
        "CASE_ID": extra.get("case_id", "CASE-" + sha[:6].upper()),
        "CREATED_UTC": manifest.get("created_utc", ""),
        "ANALYST": extra.get("analyst", "VoiceShield operator"),
        "SOURCE": extra.get("source", "Uploaded audio · wideband"),
        "AUDIO_SECONDS": f"{manifest.get('audio_seconds', 0)}",
        "SAMPLE_RATE": f"{manifest.get('sample_rate', config.SAMPLE_RATE)}",
        "MODE": extra.get("mode", "Ensemble (cascade)"),
        "STATE": _STATE_TEXT.get(state, state.upper()),
        "STATE_CLASS": state,
        "SCORE": f"{score:.2f}",
        "MECHANISM": mechanism,
        "HEADLINE": html.escape(headline),
        "DRIVER_LABEL": html.escape(driver_label),
        "FIRST_AMBER": _fmt_t(last.get("first_amber_t")),
        "FIRST_RED": _fmt_t(last.get("first_red_t")),
        "TOP_ARTIFACT": html.escape(top_artifact),
        "SPEAKER_CHANGED": speaker,
        "SNR": _snr_summary(timeline),
        "ACTION": _ACTION.get(state, "Review"),
        "REASONS_LI": _reasons_li(reasons),
        "CONTRIBUTIONS_ROWS": _contribution_rows(contributions, driver),
        "TIMELINE_ROWS": _timeline_rows(timeline),
        "FUSION_WEIGHTS_CAPTION": fw_caption,
        "AMBER_TH": f"{config.SCORE_AMBER:.2f}",
        "RED_TH": f"{config.SCORE_RED:.2f}",
        "SPECTROGRAM_IMG": _img_tag(
            os.path.join(package_dir, manifest.get("files", {}).get("spectrogram", "spectrogram.png")),
            "Mel spectrogram",
        ),
        "PHASE_IMG": _img_tag(
            os.path.join(package_dir, manifest.get("files", {}).get("phase_heatmap", "phase_heatmap.png")),
            "Phase discontinuity map",
        ),
        "AUDIO_SHA256": sha,
        "VERDICT_ID": "v:" + sha[:6] + "… (audio+models+config)",
        "GIT_COMMIT": _git_commit(),
        "MODEL_VERSIONS": _MODEL_VERSIONS,
        "CONFIG_SNAPSHOT": config_snapshot,
        "RETENTION": extra.get("retention", "90 days unless placed under legal hold"),
        "CONSENT_BASIS": extra.get(
            "consent_basis",
            "Recorded-line notice + fraud-prevention legitimate interest; explicit consent "
            "where a voiceprint is created (speaker-consistency module).",
        ),
        "ANALYST_NAME": html.escape(extra.get("analyst_name", "VoiceShield operator")),
        "REVIEWER_NAME": html.escape(extra.get("reviewer_name", "Reviewer")),
    }

    with open(_TEMPLATE, encoding="utf-8") as f:
        tpl = f.read()
    for key, val in values.items():
        tpl = tpl.replace("{{" + key + "}}", str(val))
    return tpl

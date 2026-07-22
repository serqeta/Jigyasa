import React from 'react'

const MODEL_META = {
  nii: { label: 'MMS-300M (NII)', detail: 'multilingual anti-deepfake · primary · AUC 0.997' },
  stage1: { label: 'AASIST-L', detail: 'legacy — retired by evaluation, not fused' },
  ssl: { label: 'XLS-R 300M', detail: 'SSL deepfake classifier' },
  spec: { label: 'AST', detail: 'spectrogram transformer · ASVspoof 5' },
  wavlm: { label: 'WavLM', detail: 'In-the-Wild deepfakes' },
  phase_pitch: { label: 'Phase / Pitch', detail: 'rule-based vocoder artifacts' },
  replay: { label: 'Replay', detail: 'loudspeaker-replay detector · EchoFake LoRA (wideband)' },
  codec: { label: 'Codec', detail: 'neural-codec/vocoder artifacts · Codecfake W2VAASIST (observing)' },
}

const card = {
  background: 'var(--color-paper-white)',
  border: '1px solid var(--color-mist-divider)',
  borderRadius: 'var(--radius-md)',
  overflow: 'hidden',
  marginBottom: 14,
}

const cardHead = {
  padding: '8px 14px',
  borderBottom: '1px solid var(--color-mist-divider)',
  fontSize: 10, fontWeight: 700,
  letterSpacing: '0.06em', textTransform: 'uppercase',
  color: 'var(--color-fog-text)',
  background: 'var(--color-cream-surface)',
}

function scoreColor(v) {
  if (v >= 0.70) return '#dc2626'
  if (v >= 0.30) return '#d97706'
  return '#16a34a'
}

function ScoreBar({ name, value }) {
  const meta = MODEL_META[name] || { label: name, detail: '' }
  return (
    <div style={{ padding: '7px 0', borderBottom: '1px solid var(--color-mist-divider)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--color-slate-text)' }}>
          {meta.label}
          {meta.detail && (
            <span style={{ fontSize: 10, fontWeight: 400, color: 'var(--color-ash-text)', marginLeft: 6 }}>
              {meta.detail}
            </span>
          )}
        </span>
        <span style={{ fontSize: 12, fontWeight: 700, fontFamily: 'var(--font-mono)', color: scoreColor(value) }}>
          {value.toFixed(3)}
        </span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: 'var(--color-mist-divider)', marginTop: 5, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${Math.round(value * 100)}%`,
          background: scoreColor(value), borderRadius: 2,
          transition: 'width 0.3s ease, background 0.3s ease',
        }} />
      </div>
    </div>
  )
}

// The replay component is now a single learned scorer (EchoFake LoRA fine-tune
// of wav2vec2), not the retired DSP sub-cue heuristics. Show its score and a
// plain-language reading of what it means.
function ReplayBreakdown({ replay }) {
  const s = replay.score ?? 0
  const reading =
    s >= 0.70 ? 'Strong loudspeaker-playback signature — likely a replayed recording, not a live voice.'
    : s >= 0.30 ? 'Some playback-channel cues present — verify the caller is live.'
    : 'No meaningful replay signature — consistent with a live microphone.'
  return (
    <div style={{
      marginTop: 8, padding: '8px 10px', borderRadius: 6,
      background: 'var(--color-cream-surface)',
      border: '1px solid var(--color-mist-divider)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: 'var(--color-fog-text)' }}>
          Replay likelihood · {replay.model || 'echofake-lora'}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: scoreColor(s) }}>
          {s.toFixed(2)}
        </span>
      </div>
      <div style={{ fontSize: 11, color: 'var(--color-slate-text)', lineHeight: 1.5, marginTop: 4 }}>
        {reading}
      </div>
      <div style={{ fontSize: 9, color: 'var(--color-ash-text)', marginTop: 4 }}>
        Wideband only — not calibrated for narrowband telephony.
      </div>
    </div>
  )
}

export default function EnsemblePanel({ entry }) {
  const scores = entry?.component_scores
  if (!scores || Object.keys(scores).length === 0) return null

  const order = ['nii', 'ssl', 'wavlm', 'codec', 'stage1', 'spec', 'phase_pitch', 'replay']
  const keys = [
    ...order.filter(k => k in scores),
    ...Object.keys(scores).filter(k => !order.includes(k)),
  ]

  return (
    <div style={card}>
      <div style={cardHead}>Ensemble Breakdown · {keys.length} components</div>
      <div style={{ padding: '2px 14px 12px' }}>
        {keys.map(k => <ScoreBar key={k} name={k} value={scores[k]} />)}
        {entry.replay && <ReplayBreakdown replay={entry.replay} />}
      </div>
    </div>
  )
}

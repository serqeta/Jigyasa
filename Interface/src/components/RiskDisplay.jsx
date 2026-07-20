import React, { useState, useEffect, useRef } from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'
import Sparkline from './Sparkline.jsx'
import EnsemblePanel from './EnsemblePanel.jsx'

function SpectrogramHeatmap({ matrix, endTime }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !matrix || matrix.length === 0) return
    const ctx = canvas.getContext('2d')
    const numRows = matrix.length      // Frequency bins
    const numCols = matrix[0].length   // Time frames

    canvas.width = 800
    canvas.height = 200

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const cellWidth = canvas.width / numCols
    const cellHeight = canvas.height / numRows

    // Fixed dB floor: librosa returns values in [-80, 0] dB (ref=np.max).
    // Anchoring to -60 dB means silence is genuinely black; only real
    // speech energy shows as bright. Local min/max normalization stretches
    // noise to fill the full color range, making silence look like signal.
    const DB_FLOOR = -60
    const DB_RANGE = 60  // [-60, 0] dB

    function getMagmaColor(v) {
      const r = Math.floor(255 * Math.pow(v, 1.5))
      const g = Math.floor(255 * Math.pow(v, 3.0) * 0.9 + 20 * (1 - v))
      const b = Math.floor(255 * Math.pow(v, 6.0) * 0.8 + 40 * v * (1 - v))
      return `rgb(${r},${g},${b})`
    }

    for (let r = 0; r < numRows; r++) {
      const y = canvas.height - (r + 1) * cellHeight
      for (let c = 0; c < numCols; c++) {
        const val = matrix[r][c]
        const norm = Math.max(0, Math.min(1, (val - DB_FLOOR) / DB_RANGE))
        ctx.fillStyle = getMagmaColor(norm)
        ctx.fillRect(c * cellWidth, y, cellWidth + 0.5, cellHeight + 0.5)
      }
    }
  }, [matrix])

  if (!matrix || matrix.length === 0) {
    return (
      <div style={{ height: 150, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-ash-text)', fontSize: 12 }}>
        No spectral data available for this chunk
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: 200, borderRadius: 4, border: '1px solid var(--color-mist-divider)' }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: 9, color: 'var(--color-ash-text)', marginTop: 4 }}>
        <span>{endTime != null ? `${Math.max(0, endTime - 4).toFixed(1)}s` : '0.0s'}</span>
        <span>{endTime != null ? `${endTime.toFixed(1)}s` : '4.0s'}</span>
      </div>
    </div>
  )
}

function PitchPhasePlots({ pitch, phase }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    canvas.width = 800
    canvas.height = 200
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const W = canvas.width
    const H = canvas.height

    ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)'
    ctx.lineWidth = 1
    for (let i = 1; i < 4; i++) {
      const y = (H / 4) * i
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(W, y)
      ctx.stroke()
    }

    if (pitch && pitch.length > 0) {
      const valid = pitch.filter(x => x !== null)
      const minP = valid.length > 0 ? Math.min(...valid) : 50
      const maxP = valid.length > 0 ? Math.max(...valid) : 400
      const pRange = (maxP - minP) || 10.0

      ctx.strokeStyle = '#0284c7'
      ctx.lineWidth = 2.5
      ctx.beginPath()
      let first = true
      const step = W / (pitch.length - 1)
      
      for (let i = 0; i < pitch.length; i++) {
        const val = pitch[i]
        if (val === null) {
          first = true
          continue
        }
        const x = i * step
        const y = H * 0.1 + (H * 0.4) * (1 - (val - minP) / pRange)
        if (first) {
          ctx.moveTo(x, y)
          first = false
        } else {
          ctx.lineTo(x, y)
        }
      }
      ctx.stroke()

      ctx.fillStyle = '#0284c7'
      ctx.font = 'bold 9px sans-serif'
      ctx.fillText(`Pitch (F0): ${minP.toFixed(0)} - ${maxP.toFixed(0)} Hz`, 8, 14)
    } else {
      ctx.fillStyle = 'var(--color-ash-text)'
      ctx.font = '9px sans-serif'
      ctx.fillText('Pitch: No data (unvoiced or silent)', 8, 14)
    }

    if (phase && phase.length > 0) {
      ctx.strokeStyle = '#ea580c'
      ctx.lineWidth = 2
      ctx.beginPath()
      const step = W / (phase.length - 1)
      for (let i = 0; i < phase.length; i++) {
        const val = phase[i]
        const x = i * step
        const y = H * 0.9 - (H * 0.3) * val
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
      }
      ctx.stroke()

      ctx.fillStyle = '#ea580c'
      ctx.font = 'bold 9px sans-serif'
      ctx.fillText(`Phase Discontinuity Index`, 8, H - 8)
    } else {
      ctx.fillStyle = 'var(--color-ash-text)'
      ctx.font = '9px sans-serif'
      ctx.fillText('Phase Discontinuity: No data', 8, H - 8)
    }
  }, [pitch, phase])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: 200, borderRadius: 4, border: '1px solid var(--color-mist-divider)', background: 'var(--color-cream-surface)' }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: 9, color: 'var(--color-ash-text)', marginTop: 4 }}>
        <span>0.0s (Start)</span>
        <span>4.0s (End)</span>
      </div>
    </div>
  )
}

function ForensicPanel({ entry }) {
  const [tab, setTab] = useState('stft')

  const tabBtn = (t, label) => {
    const active = tab === t
    return (
      <button
        onClick={() => setTab(t)}
        style={{
          padding: '4px 10px',
          background: active ? 'var(--color-ink-black)' : 'transparent',
          color: active ? '#fff' : 'var(--color-slate-text)',
          border: `1px solid ${active ? 'var(--color-ink-black)' : 'var(--color-mist-divider)'}`,
          borderRadius: 4,
          fontSize: 11,
          fontWeight: active ? 600 : 400,
          cursor: 'pointer',
          outline: 'none',
        }}
      >
        {label}
      </button>
    )
  }

  return (
    <div style={card}>
      <div style={cardHead}>
        <span>Forensic Visualizations</span>
        {entry.time != null && (
          <span style={{ float: 'right', color: 'var(--color-ash-text)', textTransform: 'none', fontFamily: 'var(--font-mono)' }}>
            window ending at {entry.time.toFixed(2)}s
          </span>
        )}
      </div>
      <div style={{ padding: '8px 12px 12px' }}>
        <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
          {tabBtn('stft', 'STFT (Linear)')}
          {tabBtn('mel', 'Mel Spec')}
          {tabBtn('cqt', 'CQT Spec')}
          {tabBtn('pitch_phase', 'Pitch & Phase')}
        </div>
        <div>
          {tab === 'stft' && <SpectrogramHeatmap matrix={entry.spec_linear} endTime={entry.time} />}
          {tab === 'mel' && <SpectrogramHeatmap matrix={entry.spec_mel} endTime={entry.time} />}
          {tab === 'cqt' && <SpectrogramHeatmap matrix={entry.spec_cqt} endTime={entry.time} />}
          {tab === 'pitch_phase' && <PitchPhasePlots pitch={entry.pitch_contour} phase={entry.phase_contour} />}
        </div>
      </div>
    </div>
  )
}

const SC = {
  green: { bg: '#dcfce7', border: '#bbf7d0', text: '#15803d', glow: '#16a34a' },
  amber: { bg: '#fef9c3', border: '#fde68a', text: '#a16207', glow: '#d97706' },
  red: { bg: '#fee2e2', border: '#fecaca', text: '#b91c1c', glow: '#dc2626' },
  grey: { bg: '#f3f4f6', border: '#e5e7eb', text: '#4b5563', glow: '#6b7280' },
}

const ACTIONS = {
  green: 'Voice patterns appear genuine. Continue call normally.',
  amber: 'Suspicious voice characteristics detected. Consider escalating to supervisor for verification.',
  red: 'High-confidence synthetic voice. Terminate call immediately and file a fraud report.',
  grey: 'Audio quality insufficient for analysis. Request caller to move to a quieter environment.',
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
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
}

function MetaRow({ label, value, mono }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '6px 0', borderBottom: '1px solid var(--color-mist-divider)',
    }}>
      <span style={{ fontSize: 12, color: 'var(--color-fog-text)', fontWeight: 500 }}>{label}</span>
      <span style={{ fontSize: 13, color: 'var(--color-ink-black)', fontWeight: 600, fontFamily: mono ? 'var(--font-mono)' : undefined }}>
        {value}
      </span>
    </div>
  )
}

function Bar({ value, max, color }) {
  return (
    <div style={{ height: 3, borderRadius: 2, background: 'var(--color-mist-divider)', margin: '4px 0 10px', overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${Math.min(100, (value / max) * 100)}%`, background: color, borderRadius: 2, transition: 'width 0.3s ease' }} />
    </div>
  )
}

function SummaryGrid({ summary }) {
  if (!summary) return null
  const cells = [
    { label: 'Chunks', value: summary.total_chunks },
    { label: 'Duration', value: `${summary.duration_s}s` },
    { label: 'Mean score', value: summary.mean_score.toFixed(3) },
    { label: 'Final state', value: summary.final_state?.toUpperCase(), color: SC[summary.final_state]?.glow },
    { label: 'First AMBER', value: summary.first_amber_t != null ? `${summary.first_amber_t.toFixed(2)}s` : '—' },
    { label: 'First RED', value: summary.first_red_t != null ? `${summary.first_red_t.toFixed(2)}s` : '—' },
  ]
  return (
    <div style={card}>
      <div style={cardHead}>Analysis Summary</div>
      <div style={{ padding: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {cells.map(({ label, value, color }) => (
          <div key={label} style={{
            padding: '10px 12px',
            background: 'var(--color-cream-surface)',
            border: '1px solid var(--color-mist-divider)',
            borderRadius: 8,
          }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase', color: 'var(--color-fog-text)' }}>{label}</div>
            <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-mono)', marginTop: 2, color: color || 'var(--color-ink-black)' }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function RiskDisplay() {
  const { state } = useVoiceShield()
  const entry = state.current
  const sk = entry?.state || 'grey'
  const c = SC[sk] || SC.grey

  if (!entry) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 12, color: 'var(--color-ash-text)', textAlign: 'center', padding: 32 }}>
        <div style={{ fontSize: 40 }}>◎</div>
        <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--color-slate-text)' }}>No data yet</div>
        <div style={{ fontSize: 13, maxWidth: 280, lineHeight: 1.6 }}>
          Upload a WAV file or connect to a live WebSocket stream to begin analysis.
        </div>
      </div>
    )
  }

  const snrGlow = entry.snr_db >= 12 ? '#16a34a' : entry.snr_db >= 8 ? '#d97706' : '#6b7280'

  // speech_active may be absent on older payloads; default to true so latched
  // RED/AMBER states don't get incorrectly dimmed.
  const speechActive = entry.speech_active !== false
  const voicedPct = entry.voiced_ratio != null ? Math.round(entry.voiced_ratio * 100) : null

  // When no speech, the badge dims to grey regardless of the risk label —
  // the backend already outputs GREY in this case, but this guards older payloads.
  const badgeC = speechActive ? c : SC.grey

  return (
    <div style={{ padding: '16px 16px 0' }}>

      {/* Risk badge */}
      <div style={{
        padding: '12px 18px', borderRadius: 'var(--radius-sm)',
        border: `1px solid ${badgeC.border}`, background: badgeC.bg,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 8,
        transition: 'background 0.3s, border-color 0.3s',
      }}>
        <div>
          <div style={{ fontSize: 30, fontWeight: 700, letterSpacing: '-0.8px', color: badgeC.glow }}>
            {speechActive ? sk.toUpperCase() : 'MONITORING'}
          </div>
          <div style={{ fontSize: 11, color: badgeC.text, marginTop: 1 }}>t = {entry.time.toFixed(2)} s</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 42, fontWeight: 700, fontFamily: 'var(--font-mono)', color: badgeC.glow, lineHeight: 1 }}>
            {speechActive ? Math.round(entry.score * 100) : '—'}
          </div>
          <div style={{ fontSize: 10, color: badgeC.text, fontWeight: 600, letterSpacing: '0.04em' }}>/ 100</div>
        </div>
      </div>

      {/* Speech-activity indicator */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '6px 14px', borderRadius: 'var(--radius-sm)', marginBottom: 10,
        border: `1px solid var(--color-mist-divider)`,
        background: speechActive ? '#ecfdf5' : 'var(--color-cream-surface)',
        transition: 'background 0.3s',
      }}>
        <div style={{
          width: 8, height: 8, borderRadius: '50%',
          background: speechActive ? '#16a34a' : '#94a3b8',
          boxShadow: speechActive ? '0 0 0 3px rgba(22,163,74,0.18)' : 'none',
          flexShrink: 0,
        }} />
        <span style={{ fontSize: 11, fontWeight: 600, color: speechActive ? '#15803d' : 'var(--color-fog-text)' }}>
          {speechActive ? 'Speech detected — analyzing' : 'Listening — no speech detected'}
        </span>
        {voicedPct != null && (
          <span style={{ marginLeft: 'auto', fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--color-ash-text)' }}>
            {voicedPct}% voiced
          </span>
        )}
      </div>

      {/* Advisory action */}
      <div style={{
        padding: '9px 14px', borderRadius: 'var(--radius-sm)',
        border: `1px solid ${speechActive ? c.border : 'var(--color-mist-divider)'}`,
        background: speechActive ? c.bg : 'var(--color-cream-surface)',
        fontSize: 12, color: speechActive ? c.text : 'var(--color-fog-text)', lineHeight: 1.6, marginBottom: 14,
      }}>
        {speechActive ? ACTIONS[sk] : 'Waiting for voice. Risk scoring pauses until speech fills the analysis window.'}
      </div>

      {/* Metrics */}
      <div style={card}>
        <div style={cardHead}>Chunk Metrics</div>
        <div style={{ padding: '4px 14px 10px' }}>
          <MetaRow label="Spoof score" value={entry.score.toFixed(4)} mono />
          <Bar value={entry.score} max={1} color={c.glow} />
          <MetaRow label="SNR" value={`${entry.snr_db.toFixed(1)} dB`} mono />
          <Bar value={Math.max(0, entry.snr_db)} max={40} color={snrGlow} />
          <MetaRow label="Top artifact" value={entry.top_artifact ? entry.top_artifact.replace(/_/g, ' ') : '—'} />
          <MetaRow label="First AMBER" value={entry.first_amber_t != null ? `${entry.first_amber_t.toFixed(2)} s` : '—'} mono />
          <MetaRow label="First RED" value={entry.first_red_t != null ? `${entry.first_red_t.toFixed(2)} s` : '—'} mono />
        </div>
      </div>

      {/* Stage 2 ensemble breakdown */}
      <EnsemblePanel entry={entry} />

      {/* Sparkline */}
      <div style={card}>
        <div style={{ ...cardHead, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Score History</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--color-ash-text)', textTransform: 'none', letterSpacing: 0 }}>
            last {Math.min(state.entries.length, 40)} chunks
          </span>
        </div>
        <div style={{ padding: 10 }}>
          <Sparkline entries={state.entries} amberThreshold={state.amberThreshold} redThreshold={state.redThreshold} />
        </div>
        <div style={{ display: 'flex', gap: 14, padding: '8px 12px', borderTop: '1px solid var(--color-mist-divider)' }}>
          {[['green', '#16a34a', 'Genuine'], ['amber', '#d97706', 'Suspicious'], ['red', '#dc2626', 'Synthetic'], ['grey', '#6b7280', 'Insufficient SNR']].map(([key, color, label]) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--color-fog-text)' }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
              {label}
            </div>
          ))}
        </div>
      </div>

      {/* Forensic Visualizations */}
      <ForensicPanel entry={state.selectedEntry || entry} />

      {/* Summary (batch mode) */}
      <SummaryGrid summary={state.summary} />

    </div>
  )
}

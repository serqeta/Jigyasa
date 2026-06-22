import React from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'
import Sparkline from './Sparkline.jsx'

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

  return (
    <div style={{ padding: '16px 16px 0' }}>

      {/* Risk badge */}
      <div style={{
        padding: '12px 18px', borderRadius: 'var(--radius-sm)',
        border: `1px solid ${c.border}`, background: c.bg,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 12,
      }}>
        <div>
          <div style={{ fontSize: 30, fontWeight: 700, letterSpacing: '-0.8px', color: c.glow }}>
            {sk.toUpperCase()}
          </div>
          <div style={{ fontSize: 11, color: c.text, marginTop: 1 }}>t = {entry.time.toFixed(2)} s</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 42, fontWeight: 700, fontFamily: 'var(--font-mono)', color: c.glow, lineHeight: 1 }}>
            {Math.round(entry.score * 100)}
          </div>
          <div style={{ fontSize: 10, color: c.text, fontWeight: 600, letterSpacing: '0.04em' }}>/ 100</div>
        </div>
      </div>

      {/* Advisory action */}
      <div style={{
        padding: '9px 14px', borderRadius: 'var(--radius-sm)',
        border: `1px solid ${c.border}`, background: c.bg,
        fontSize: 12, color: c.text, lineHeight: 1.6, marginBottom: 14,
      }}>
        {ACTIONS[sk]}
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

      {/* Summary (batch mode) */}
      <SummaryGrid summary={state.summary} />

    </div>
  )
}

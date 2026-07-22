import React from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'

const SC = {
  green: { bg: '#dcfce7', border: '#bbf7d0', text: '#15803d' },
  amber: { bg: '#fef9c3', border: '#fde68a', text: '#a16207' },
  red: { bg: '#fee2e2', border: '#fecaca', text: '#b91c1c' },
  grey: { bg: '#f3f4f6', border: '#e5e7eb', text: '#4b5563' },
}

const card = {
  background: 'var(--color-paper-white)',
  border: '1px solid var(--color-mist-divider)',
  borderRadius: 'var(--radius-md)',
  padding: 16,
  marginBottom: 12,
}

function StatePill({ s }) {
  const c = SC[s] || SC.grey
  return (
    <span style={{
      fontSize: 11, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
      padding: '2px 8px', borderRadius: 4, color: c.text, background: c.bg, border: `1px solid ${c.border}`,
    }}>{s}</span>
  )
}

export default function ReportsView() {
  const { state, generateReport, dispatch } = useVoiceShield()
  const hasCase = state.entries.length > 0
  const generating = state.reportStatus === 'generating'

  return (
    <div style={{ maxWidth: 860, margin: '0 auto', padding: '24px 24px 48px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 6 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.4px', margin: 0 }}>Forensic Reports</h1>
        <span style={{ fontSize: 12, color: 'var(--color-ash-text)' }}>{state.reports.length} generated this session</span>
      </div>
      <p style={{ fontSize: 13, color: 'var(--color-fog-text)', lineHeight: 1.6, marginTop: 4, marginBottom: 20 }}>
        Each report is a self-contained, print-to-PDF forensic document — the verdict rationale, the full detector
        ensemble breakdown, the per-chunk timeline, the integrity hash, and the actual spectrogram &amp; phase-map
        images — all wired from the real analysis. Open one and use your browser's <b>Print → Save as PDF</b>.
      </p>

      {/* Generate for the current case */}
      <div style={{ ...card, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-text)' }}>Current analysis</div>
          <div style={{ fontSize: 12, color: 'var(--color-ash-text)', marginTop: 2 }}>
            {hasCase
              ? `${state.mode === 'file' ? (state.selectedFile?.name || 'Uploaded file') : 'Live capture'} · ${state.chunkCount} chunks analysed`
              : 'No analysis yet — upload a file or run the microphone first.'}
          </div>
        </div>
        <button
          onClick={generateReport}
          disabled={!hasCase || generating}
          style={{
            padding: '9px 16px', borderRadius: 8, border: 'none',
            background: !hasCase || generating ? 'var(--color-mist-divider)' : 'var(--color-ink-black)',
            color: !hasCase || generating ? 'var(--color-ash-text)' : '#fff',
            fontSize: 13, fontWeight: 600, cursor: !hasCase || generating ? 'not-allowed' : 'pointer',
            flexShrink: 0, whiteSpace: 'nowrap',
          }}
        >
          {generating ? '⌛ Generating…' : '📄 Generate forensic report'}
        </button>
      </div>
      {state.reportStatus === 'error' && (
        <div style={{ fontSize: 12, color: '#b91c1c', marginBottom: 12 }}>
          Report generation failed. Check the server is running in the matching mode.
        </div>
      )}

      {/* Generated reports */}
      {state.reports.length === 0 ? (
        <div style={{ ...card, textAlign: 'center', color: 'var(--color-ash-text)', padding: '32px 16px' }}>
          <div style={{ fontSize: 32, marginBottom: 8 }}>📄</div>
          <div style={{ fontSize: 13 }}>No reports generated yet.</div>
        </div>
      ) : (
        state.reports.map(r => (
          <div key={r.id} style={{ ...card, display: 'flex', alignItems: 'center', gap: 14 }}>
            <StatePill s={r.state} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--color-slate-text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {r.label}
              </div>
              <div style={{ fontSize: 11, color: 'var(--color-ash-text)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                {new Date(r.ts).toLocaleString()} · peak score {Math.round(r.score * 100)}/100
              </div>
            </div>
            <button
              onClick={() => dispatch({ type: 'OPEN_REPORT', payload: r })}
              style={{
                padding: '7px 14px', borderRadius: 8, border: '1px solid var(--color-mist-divider)',
                background: 'var(--color-cream-surface)', fontSize: 12, fontWeight: 600,
                color: 'var(--color-slate-text)', cursor: 'pointer', flexShrink: 0,
              }}
            >Open report</button>
          </div>
        ))
      )}
    </div>
  )
}

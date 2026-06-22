import React, { useEffect, useRef, useState } from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'

const SC = {
  green: { bg: '#f0fdf4', border: '#bbf7d0', text: '#15803d' },
  amber: { bg: '#fffbeb', border: '#fde68a', text: '#a16207' },
  red: { bg: '#fff1f2', border: '#fecdd3', text: '#b91c1c' },
  grey: { bg: '#f9fafb', border: '#e5e7eb', text: '#6b7280' },
}

const filterBtn = (active) => ({
  padding: '3px 8px', borderRadius: 4, fontSize: 11, fontWeight: active ? 700 : 400,
  border: `1px solid ${active ? 'var(--color-ink-black)' : 'var(--color-mist-divider)'}`,
  background: active ? 'var(--color-ink-black)' : 'transparent',
  color: active ? '#fff' : 'var(--color-fog-text)',
  cursor: 'pointer',
})

const actionBtn = {
  padding: '4px 8px', borderRadius: 'var(--radius-xs)',
  border: '1px solid var(--color-mist-divider)',
  background: 'transparent', fontSize: 12,
  color: 'var(--color-slate-text)', cursor: 'pointer', fontWeight: 500,
}

function count(entries, s) { return entries.filter(e => e.state === s).length }

export default function TimelineTable() {
  const { state, dispatch, exportJson, exportCsv } = useVoiceShield()
  const bottomRef = useRef(null)
  const [autoScroll, setAutoScroll] = useState(true)

  const filtered = state.entries.filter(e =>
    state.filterState === 'all' || e.state === state.filterState
  )

  useEffect(() => {
    if (autoScroll) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [state.entries.length, autoScroll])

  const jumpTo = (targetState) => {
    const e = filtered.find(e => e.state === targetState)
    if (e) dispatch({ type: 'SET_SELECTED_ENTRY', payload: e })
  }

  const copyRow = () => {
    const e = state.selectedEntry || state.current
    if (e) navigator.clipboard.writeText(JSON.stringify(e, null, 2)).catch(() => {})
  }

  const sectionHead = (title, extra) => (
    <div style={{
      padding: '7px 12px', borderBottom: '1px solid var(--color-mist-divider)',
      fontSize: 10, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase',
      color: 'var(--color-fog-text)', background: 'var(--color-cream-surface)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0,
    }}>
      <span>{title}</span>{extra}
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      {sectionHead(
        'Timeline Log',
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 11,
          background: 'var(--color-mist-divider)',
          padding: '1px 6px', borderRadius: 9, color: 'var(--color-fog-text)',
          letterSpacing: 0, textTransform: 'none', fontWeight: 400,
        }}>{state.entries.length}</span>
      )}

      {/* Filter bar */}
      <div style={{
        padding: '7px 10px', borderBottom: '1px solid var(--color-mist-divider)',
        display: 'flex', gap: 5, flexWrap: 'wrap', alignItems: 'center',
        background: 'var(--color-paper-white)', flexShrink: 0,
      }}>
        {['all', 'green', 'amber', 'red', 'grey'].map(f => (
          <button key={f} style={filterBtn(state.filterState === f)}
            onClick={() => dispatch({ type: 'SET_FILTER_STATE', payload: f })}>
            {f === 'all' ? `All (${state.entries.length})` : `${f.toUpperCase()} (${count(state.entries, f)})`}
          </button>
        ))}
      </div>

      {/* Jump + scroll toolbar */}
      <div style={{
        padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)',
        display: 'flex', gap: 5, flexWrap: 'wrap',
        background: 'var(--color-paper-white)', flexShrink: 0,
      }}>
        <button style={actionBtn} onClick={() => jumpTo('amber')}>→ First AMBER</button>
        <button style={actionBtn} onClick={() => jumpTo('red')}>→ First RED</button>
        <div style={{ flex: 1 }} />
        <button
          style={{ ...actionBtn, color: autoScroll ? 'var(--color-ink-black)' : 'var(--color-ash-text)' }}
          onClick={() => setAutoScroll(a => !a)}
        >{autoScroll ? '⏸' : '▶'} scroll</button>
      </div>

      {/* Export toolbar */}
      <div style={{
        padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)',
        display: 'flex', gap: 5,
        background: 'var(--color-paper-white)', flexShrink: 0,
      }}>
        <button style={actionBtn} onClick={exportJson}>↓ JSON</button>
        <button style={actionBtn} onClick={exportCsv}>↓ CSV</button>
        <button style={actionBtn} onClick={copyRow}>⎘ Copy row</button>
        <div style={{ flex: 1 }} />
        <button
          style={{ ...actionBtn, color: '#dc2626', borderColor: '#fecaca' }}
          onClick={() => dispatch({ type: 'CLEAR_TIMELINE' })}
        >✕ Clear</button>
      </div>

      {/* Table */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {filtered.length === 0 ? (
          <div style={{ padding: '32px 16px', textAlign: 'center', color: 'var(--color-ash-text)', fontSize: 13 }}>
            {state.entries.length === 0 ? 'No data yet' : `No ${state.filterState.toUpperCase()} entries`}
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
            <thead>
              <tr>
                {['Time', 'Score', 'State', 'SNR', 'Artifact'].map(h => (
                  <th key={h} style={{
                    padding: '6px 10px', textAlign: 'left', fontWeight: 600,
                    fontSize: 10, color: 'var(--color-fog-text)',
                    borderBottom: '1px solid var(--color-mist-divider)',
                    background: 'var(--color-cream-surface)',
                    position: 'sticky', top: 0, zIndex: 1,
                    letterSpacing: '0.04em', textTransform: 'uppercase',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((e, i) => {
                const c = SC[e.state] || SC.grey
                const selected = state.selectedEntry?.time === e.time
                return (
                  <tr
                    key={i}
                    onClick={() => dispatch({ type: 'SET_SELECTED_ENTRY', payload: e })}
                    style={{
                      background: selected ? c.bg : 'transparent',
                      cursor: 'pointer',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={ev => { if (!selected) ev.currentTarget.style.background = 'var(--color-cream-surface)' }}
                    onMouseLeave={ev => { if (!selected) ev.currentTarget.style.background = 'transparent' }}
                  >
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)', color: 'var(--color-slate-text)' }}>{e.time.toFixed(2)}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)', color: 'var(--color-slate-text)' }}>{e.score.toFixed(3)}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)' }}>
                      <span style={{
                        padding: '1px 5px', borderRadius: 3,
                        background: c.bg, border: `1px solid ${c.border}`,
                        color: c.text, fontSize: 10, fontWeight: 700, letterSpacing: '0.04em',
                      }}>{e.state.toUpperCase()}</span>
                    </td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)', color: 'var(--color-slate-text)' }}>{e.snr_db.toFixed(1)}</td>
                    <td style={{ padding: '6px 10px', borderBottom: '1px solid var(--color-mist-divider)', color: 'var(--color-ash-text)', maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {e.top_artifact?.replace(/_/g, ' ') || '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Selected entry detail */}
      {state.selectedEntry && (
        <div style={{
          borderTop: '1px solid var(--color-mist-divider)',
          background: 'var(--color-charcoal-surface)',
          flexShrink: 0, maxHeight: 180, overflow: 'auto',
        }}>
          <div style={{
            padding: '5px 12px', borderBottom: '1px solid var(--color-graphite-border)',
            fontSize: 10, fontWeight: 600, letterSpacing: '0.05em',
            color: 'var(--color-ash-text)', textTransform: 'uppercase',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span>Selected entry</span>
            <button
              onClick={() => dispatch({ type: 'SET_SELECTED_ENTRY', payload: null })}
              style={{ background: 'none', border: 'none', color: 'var(--color-ash-text)', cursor: 'pointer', fontSize: 14 }}
            >×</button>
          </div>
          <pre style={{
            padding: 12, margin: 0,
            fontFamily: 'var(--font-mono)', fontSize: 11, lineHeight: 1.5,
            color: '#d7d7d7', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
          }}>
            {JSON.stringify(state.selectedEntry, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

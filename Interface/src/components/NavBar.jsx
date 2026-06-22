import React, { useState } from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'

const btn = {
  padding: '5px 12px',
  borderRadius: 'var(--radius-xs)',
  border: '1px solid var(--color-mist-divider)',
  background: 'transparent',
  fontSize: 13,
  fontWeight: 500,
  color: 'var(--color-slate-text)',
  cursor: 'pointer',
}

export default function NavBar() {
  const { state, dispatch, checkHealth } = useVoiceShield()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(state.serverUrl)

  const applyUrl = () => {
    dispatch({ type: 'SET_CONFIG', payload: { serverUrl: draft.replace(/\/$/, '') } })
    setEditing(false)
    setTimeout(checkHealth, 80)
  }

  const healthColor = state.health === 'ok' ? 'var(--state-green-glow)'
    : state.health === 'error' ? 'var(--state-red-glow)'
    : 'var(--color-ash-text)'

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '0 20px',
      height: '100%',
      background: 'var(--color-paper-white)',
      borderBottom: '1px solid var(--color-mist-divider)',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <div style={{
          width: 22, height: 22,
          background: 'var(--color-ink-black)',
          borderRadius: 5,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 10, fontWeight: 700, color: '#fff', letterSpacing: '-0.5px',
        }}>VS</div>
        <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: '-0.3px' }}>VoiceShield</span>
        <span style={{
          fontSize: 10, fontWeight: 600, color: 'var(--color-fog-text)',
          background: 'var(--color-cream-surface)',
          border: '1px solid var(--color-mist-divider)',
          borderRadius: 3, padding: '1px 5px', letterSpacing: '0.04em',
        }}>STAGE 1</span>
      </div>

      <div style={{ width: 1, height: 20, background: 'var(--color-mist-divider)' }} />

      {/* Server URL */}
      <span style={{ fontSize: 12, color: 'var(--color-fog-text)', fontWeight: 500 }}>Server</span>
      {editing ? (
        <input
          value={draft}
          onChange={e => setDraft(e.target.value)}
          onBlur={applyUrl}
          onKeyDown={e => { if (e.key === 'Enter') applyUrl(); if (e.key === 'Escape') setEditing(false) }}
          autoFocus
          style={{
            width: 220, padding: '4px 9px',
            border: '1px solid var(--color-signal-blue)',
            borderRadius: 'var(--radius-xs)',
            fontFamily: 'var(--font-mono)', fontSize: 12,
            background: 'var(--color-cream-surface)',
            outline: 'none',
          }}
        />
      ) : (
        <code
          onClick={() => { setDraft(state.serverUrl); setEditing(true) }}
          style={{
            padding: '4px 9px',
            border: '1px solid var(--color-mist-divider)',
            borderRadius: 'var(--radius-xs)',
            fontFamily: 'var(--font-mono)', fontSize: 12,
            color: 'var(--color-slate-text)',
            cursor: 'pointer',
            background: 'var(--color-cream-surface)',
          }}
          title="Click to edit server URL"
        >{state.serverUrl}</code>
      )}

      <div style={{ flex: 1 }} />

      {/* Mode indicator */}
      <div style={{
        fontSize: 11, fontWeight: 600,
        color: state.mode === 'stream' ? 'var(--state-green-glow)' : 'var(--color-fog-text)',
        letterSpacing: '0.05em', textTransform: 'uppercase',
      }}>
        {state.mode === 'stream' ? '● LIVE' : '⬤ FILE'}
      </div>

      <div style={{ width: 1, height: 20, background: 'var(--color-mist-divider)' }} />

      {/* Health */}
      <button
        onClick={checkHealth}
        title="Re-check API health"
        style={{ ...btn, display: 'flex', alignItems: 'center', gap: 5 }}
      >
        <div style={{ width: 6, height: 6, borderRadius: '50%', background: healthColor }} />
        <span style={{ color: healthColor }}>
          {state.health === 'ok'
            ? `API OK${state.healthLatencyMs ? ` · ${state.healthLatencyMs}ms` : ''}`
            : state.health === 'error' ? 'API Error'
            : 'Checking…'}
        </span>
      </button>
    </div>
  )
}

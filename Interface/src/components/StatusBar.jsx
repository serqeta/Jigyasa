import React, { useEffect, useState } from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'

function Dot({ status }) {
  const color = status === 'ok' || status === 'connected' ? '#7bd88f'
    : status === 'error' ? '#fc618d'
    : '#a9a9ac'
  return <div style={{ width: 5, height: 5, borderRadius: '50%', background: color, flexShrink: 0 }} />
}

function Sep() {
  return <span style={{ color: '#38383a' }}>│</span>
}

function Item({ label, value, dot }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
      {dot !== undefined && <Dot status={dot} />}
      {label && <span style={{ color: '#6d6d70' }}>{label}</span>}
      <span style={{ color: '#d7d7d7' }}>{value}</span>
    </div>
  )
}

export default function StatusBar() {
  const { state } = useVoiceShield()
  const [now, setNow] = useState(Date.now())

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const lastAgo = state.lastUpdateTime
    ? `${((now - state.lastUpdateTime) / 1000).toFixed(1)}s ago`
    : '—'

  const finalState = state.current?.state?.toUpperCase() || '—'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12,
      padding: '0 14px', height: '100%',
      background: 'var(--color-charcoal-surface)',
      borderTop: '1px solid var(--color-graphite-border)',
      fontSize: 11, fontFamily: 'var(--font-mono)',
      color: 'var(--color-ash-text)', overflow: 'hidden',
    }}>
      <Item label="WS" value={state.wsStatus} dot={state.wsStatus} />
      <Sep />
      <Item label="API" value={state.health || '—'} dot={state.health} />
      <Sep />
      <Item label="mode" value={state.mode} />
      <Sep />
      <Item label="chunks" value={state.chunkCount} />
      <Sep />
      <Item label="entries" value={state.entries.length} />
      <Sep />
      <Item label="state" value={finalState} />
      {state.current && (
        <>
          <Sep />
          <Item label="score" value={state.current.score.toFixed(3)} />
          <Sep />
          <Item label="snr" value={`${state.current.snr_db.toFixed(1)}dB`} />
        </>
      )}
      {state.mode === 'stream' && (
        <>
          <Sep />
          <Item label="last" value={lastAgo} />
        </>
      )}
      {state.paused && (
        <>
          <Sep />
          <Item value="⏸ PAUSED" />
        </>
      )}
      <div style={{ flex: 1 }} />
      <Item label="" value="VoiceShield Stage 1" />
      <Sep />
      <Item value={new Date(now).toLocaleTimeString()} />
    </div>
  )
}

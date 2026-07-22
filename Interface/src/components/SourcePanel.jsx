import React, { useRef, useState } from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'

const section = {
  background: 'var(--color-paper-white)',
  border: '1px solid var(--color-mist-divider)',
  borderRadius: 'var(--radius-md)',
  overflow: 'hidden',
  marginBottom: 12,
}

const sectionHead = {
  padding: '8px 14px',
  borderBottom: '1px solid var(--color-mist-divider)',
  fontSize: 10, fontWeight: 700,
  letterSpacing: '0.06em', textTransform: 'uppercase',
  color: 'var(--color-fog-text)',
  background: 'var(--color-cream-surface)',
}

const sectionBody = { padding: '12px 14px' }

const primaryBtn = (disabled) => ({
  width: '100%', padding: '7px 0', borderRadius: 'var(--radius-sm)', border: 'none',
  background: disabled ? 'var(--color-mist-divider)' : 'var(--color-ink-black)',
  color: disabled ? 'var(--color-ash-text)' : '#fff',
  fontSize: 13, fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer', marginTop: 8,
})

const ghostBtn = (danger) => ({
  width: '100%', padding: '6px 0', borderRadius: 'var(--radius-sm)',
  border: `1px solid ${danger ? 'var(--state-red-glow)' : 'var(--color-mist-divider)'}`,
  background: 'transparent',
  color: danger ? 'var(--state-red-glow)' : 'var(--color-slate-text)',
  fontSize: 13, fontWeight: 500, cursor: 'pointer', marginTop: 6,
})

const sliderLabel = {
  display: 'flex', justifyContent: 'space-between',
  fontSize: 12, color: 'var(--color-slate-text)', marginBottom: 4,
}

const pill = (variant) => {
  const map = {
    done: { bg: '#dcfce7', border: '#bbf7d0', color: '#15803d' },
    error: { bg: '#fee2e2', border: '#fecaca', color: '#b91c1c' },
    analyzing: { bg: '#fef9c3', border: '#fde68a', color: '#a16207' },
    cancelled: { bg: '#f3f4f6', border: '#e5e7eb', color: '#4b5563' },
    idle: { bg: '#f3f4f6', border: '#e5e7eb', color: '#6b7280' },
  }
  const c = map[variant] || map.idle
  return {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    padding: '3px 8px', borderRadius: 4, marginTop: 8, fontSize: 12, fontWeight: 500,
    background: c.bg, border: `1px solid ${c.border}`, color: c.color,
  }
}

function fmt(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 ** 2).toFixed(2)} MB`
}

function Section({ title, children }) {
  return <div style={section}><div style={sectionHead}>{title}</div><div style={sectionBody}>{children}</div></div>
}

export default function SourcePanel() {
  const { state, dispatch, analyzeFile, cancelAnalysis, connectWs, disconnectWs, resetState, exportEvidence, startMic, stopMic } = useVoiceShield()
  const fileRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const setFile = (f) => dispatch({
    type: 'SET_FILE',
    payload: { file: f, info: { name: f.name, size: f.size, type: f.type } },
  })

  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]; if (f) setFile(f)
  }

  const isAnalyzing = state.analysisStatus === 'analyzing'
  const canAnalyze = !!state.selectedFile && !isAnalyzing && state.mode === 'file'

  const wsConnected = state.wsStatus === 'connected'

  return (
    <div style={{ padding: '12px 12px 0' }}>

      {/* Mode toggle */}
      <Section title="Source Mode">
        <div style={{ display: 'flex', gap: 0, border: '1px solid var(--color-mist-divider)', borderRadius: 8, overflow: 'hidden' }}>
          {['file', 'stream', 'mic'].map((m, i) => (
            <button key={m}
              onClick={() => dispatch({ type: 'SET_MODE', payload: m })}
              style={{
                flex: 1, padding: '6px 0', border: 'none',
                borderRight: i < 2 ? '1px solid var(--color-mist-divider)' : 'none',
                background: state.mode === m ? 'var(--color-ink-black)' : 'var(--color-paper-white)',
                color: state.mode === m ? '#fff' : 'var(--color-fog-text)',
                fontSize: 12, fontWeight: state.mode === m ? 600 : 400, cursor: 'pointer',
              }}
            >{m === 'file' ? 'File' : m === 'stream' ? 'Stream' : 'Microphone'}</button>
          ))}
        </div>
      </Section>

      {/* File mode */}
      {state.mode === 'file' && (
        <Section title="Upload Audio">
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${dragging ? 'var(--color-signal-blue)' : 'var(--color-mist-divider)'}`,
              borderRadius: 8, padding: '18px 12px', textAlign: 'center',
              cursor: 'pointer', background: dragging ? '#f0f9ff' : 'transparent',
              transition: 'all 0.12s',
            }}
          >
            <div style={{ fontSize: 20, marginBottom: 5 }}>↑</div>
            <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-slate-text)' }}>
              {state.fileInfo?.name || 'Drop WAV here or click'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--color-ash-text)', marginTop: 2 }}>
              {state.fileInfo ? fmt(state.fileInfo.size) : 'WAV · 16 kHz mono recommended'}
            </div>
          </div>
          <input ref={fileRef} type="file" accept="audio/*,.wav" style={{ display: 'none' }} onChange={e => { const f = e.target.files[0]; if (f) setFile(f) }} />

          {state.analysisStatus !== 'idle' && (
            <div style={pill(state.analysisStatus)}>
              {state.analysisStatus === 'analyzing' && '⏳ Analyzing…'}
              {state.analysisStatus === 'done' && `✓ ${state.entries.length} chunks · ${state.summary?.duration_s}s`}
              {state.analysisStatus === 'error' && `✗ ${state.analysisError?.slice(0, 60) || 'Error'}`}
              {state.analysisStatus === 'cancelled' && '— Cancelled'}
            </div>
          )}

          <button style={primaryBtn(!canAnalyze)} disabled={!canAnalyze} onClick={() => analyzeFile(state.selectedFile)}>
            {isAnalyzing ? '⏳ Analyzing…' : '▶  Analyze'}
          </button>
          {isAnalyzing && <button style={ghostBtn(true)} onClick={cancelAnalysis}>■  Cancel</button>}
          {state.entries.length > 0 && !isAnalyzing && (
            <button style={ghostBtn(false)} onClick={() => dispatch({ type: 'RESET' })}>↺  Clear results</button>
          )}
        </Section>
      )}

      {/* Stream mode */}
      {state.mode === 'stream' && (
        <Section title="Live WebSocket">
          <div style={{ fontSize: 12, color: 'var(--color-fog-text)', marginBottom: 10, lineHeight: 1.6 }}>
            Connects to the pipeline WebSocket. Start the server first with{' '}
            <code style={{ fontFamily: 'var(--font-mono)', fontSize: 11, background: 'var(--color-cream-surface)', padding: '1px 4px', borderRadius: 3 }}>run_live.py</code>.
          </div>

          <div style={pill(wsConnected ? 'done' : state.wsStatus === 'error' ? 'error' : state.wsStatus === 'connecting' ? 'analyzing' : 'idle')}>
            {wsConnected ? '● Connected'
              : state.wsStatus === 'connecting' ? '⌛ Connecting…'
              : state.wsStatus === 'error' ? '✗ Error'
              : state.wsStatus === 'disconnected' ? '○ Disconnected' : '○ Idle'}
          </div>

          <button style={primaryBtn(false)} onClick={wsConnected ? disconnectWs : connectWs}>
            {wsConnected ? '■  Disconnect' : '▶  Connect'}
          </button>
          <button style={ghostBtn(false)} onClick={() => dispatch({ type: 'SET_PAUSED', payload: !state.paused })}>
            {state.paused ? '▶  Resume stream' : '⏸  Pause stream'}
          </button>
          {state.entries.length > 0 && (
            <button style={ghostBtn(false)} onClick={resetState}>↺  Reset engine</button>
          )}
          {wsConnected && state.entries.length > 0 && (
            <>
              <button
                style={ghostBtn(state.evidenceStatus === 'exporting')}
                disabled={state.evidenceStatus === 'exporting'}
                onClick={exportEvidence}
              >
                {state.evidenceStatus === 'exporting' ? '⌛ Packaging…' : '⬇  Save evidence (audit)'}
              </button>
              {state.evidenceStatus === 'done' && state.evidenceManifest && (
                <div style={{ fontSize: 10, color: 'var(--color-fog-text)', marginTop: 4, fontFamily: 'var(--font-mono)', lineHeight: 1.5, wordBreak: 'break-all' }}>
                  saved → {state.evidenceManifest.package_dir}
                </div>
              )}
              {state.evidenceStatus === 'error' && (
                <div style={{ fontSize: 10, color: '#b91c1c', marginTop: 4 }}>
                  Evidence export failed — is the live pipeline running?
                </div>
              )}
            </>
          )}
        </Section>
      )}

      {/* Microphone mode */}
      {state.mode === 'mic' && (
        <Section title="Browser Microphone">
          <div style={{ fontSize: 12, color: 'var(--color-fog-text)', marginBottom: 10, lineHeight: 1.6 }}>
            Captures your voice in the browser and streams it to the cascade
            pipeline. Start the server with{' '}
            <code style={{ fontFamily: 'var(--font-mono)', fontSize: 11, background: 'var(--color-cream-surface)', padding: '1px 4px', borderRadius: 3 }}>run_browser.py</code>.
          </div>

          <div style={pill(state.micStatus === 'live' ? 'done' : state.micStatus === 'error' ? 'error' : state.micStatus === 'starting' ? 'analyzing' : 'idle')}>
            {state.micStatus === 'live' ? '● Mic live — speak now'
              : state.micStatus === 'starting' ? '⌛ Requesting mic…'
              : state.micStatus === 'error' ? '✗ Mic unavailable (permissions? server mode?)'
              : '○ Mic off'}
          </div>

          <button style={primaryBtn(false)} onClick={state.micStatus === 'live' ? stopMic : startMic}>
            {state.micStatus === 'live' ? '■  Stop microphone' : '🎙  Start microphone'}
          </button>

          {state.entries.length > 0 && (
            <button style={ghostBtn(false)} onClick={resetState}>↺  Reset engine</button>
          )}
          {wsConnected && state.entries.length > 0 && (
            <button
              style={ghostBtn(state.evidenceStatus === 'exporting')}
              disabled={state.evidenceStatus === 'exporting'}
              onClick={exportEvidence}
            >
              {state.evidenceStatus === 'exporting' ? '⌛ Packaging…' : '⬇  Save evidence (audit)'}
            </button>
          )}
        </Section>
      )}

      {/* Config */}
      <Section title="Configuration">
        {[
          { key: 'amberThreshold', label: 'AMBER threshold', min: 0, max: 1, step: 0.01 },
          { key: 'redThreshold', label: 'RED threshold', min: 0, max: 1, step: 0.01 },
          { key: 'snrFloor', label: 'SNR floor (dB)', min: 0, max: 20, step: 0.5 },
        ].map(({ key, label, min, max, step }) => (
          <div key={key} style={{ marginBottom: 12 }}>
            <div style={sliderLabel}>
              <span style={{ fontWeight: 500 }}>{label}</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                {state[key].toFixed(key === 'snrFloor' ? 1 : 2)}
              </span>
            </div>
            <input type="range" min={min} max={max} step={step} value={state[key]}
              style={{ width: '100%', display: 'block' }}
              onChange={e => dispatch({ type: 'SET_CONFIG', payload: { [key]: parseFloat(e.target.value) } })}
            />
          </div>
        ))}

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-slate-text)' }}>Model</span>
          <select value={state.modelMode}
            onChange={e => dispatch({ type: 'SET_CONFIG', payload: { modelMode: e.target.value } })}
            style={{
              padding: '4px 8px', border: '1px solid var(--color-mist-divider)',
              borderRadius: 4, fontSize: 12, background: 'var(--color-cream-surface)',
              color: 'var(--color-slate-text)', cursor: 'pointer',
            }}>
            <option value="auto">Auto detect</option>
            <option value="aasist">AASIST-L</option>
            <option value="fallback">Rule-based</option>
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-slate-text)' }}>Window (s)</span>
          <input type="number" min={1} max={10} value={state.windowSeconds}
            onChange={e => dispatch({ type: 'SET_CONFIG', payload: { windowSeconds: parseInt(e.target.value) || 4 } })}
            style={{
              width: 60, padding: '4px 8px', textAlign: 'right',
              border: '1px solid var(--color-mist-divider)',
              borderRadius: 4, fontSize: 12, fontFamily: 'var(--font-mono)',
              background: 'var(--color-cream-surface)',
            }}
          />
        </div>

        <button
          style={{ ...ghostBtn(false), marginTop: 0 }}
          onClick={resetState}
        >↺  Reset state engine</button>
      </Section>
    </div>
  )
}

import React, { useRef } from 'react'
import { useVoiceShield } from '../context/VoiceShieldContext.jsx'

// Right-side report pane (Claude-artifact style): shows the generated forensic
// report inline (no new tab), with a Download-PDF action in the top-right.
export default function ReportPane() {
  const { state, dispatch } = useVoiceShield()
  const report = state.activeReport
  const iframeRef = useRef(null)

  if (!report) return null

  const close = () => dispatch({ type: 'CLOSE_REPORT' })

  // The report HTML carries print-to-PDF CSS; printing the iframe lets the
  // browser's "Save as PDF" produce the downloadable document.
  const downloadPdf = () => {
    const win = iframeRef.current?.contentWindow
    if (win) {
      win.focus()
      win.print()
    }
  }

  return (
    <>
      {/* click-away backdrop over the rest of the page */}
      <div
        onClick={close}
        style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.18)', zIndex: 999 }}
      />
      <div
        style={{
          position: 'fixed', top: 0, right: 0, bottom: 0,
          width: 'min(52vw, 760px)', minWidth: 360, zIndex: 1000,
          background: '#fff', boxShadow: '-10px 0 34px rgba(0,0,0,0.20)',
          display: 'flex', flexDirection: 'column',
          borderLeft: '1px solid var(--color-mist-divider)',
        }}
      >
        {/* header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '10px 14px', flexShrink: 0,
          borderBottom: '1px solid var(--color-mist-divider)',
          background: 'var(--color-cream-surface)',
        }}>
          <div style={{ minWidth: 0, marginRight: 12 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--color-ink-black)' }}>
              Forensic report
            </div>
            <div style={{
              fontSize: 11, color: 'var(--color-ash-text)',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {report.label} · {report.state?.toUpperCase()}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
            <button
              onClick={downloadPdf}
              style={{
                padding: '7px 13px', borderRadius: 8, border: 'none',
                background: 'var(--color-ink-black)', color: '#fff',
                fontSize: 12, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap',
              }}
            >
              ⬇ Download PDF
            </button>
            <button
              onClick={close}
              aria-label="Close report"
              style={{
                width: 30, height: 30, borderRadius: 8, lineHeight: 1,
                border: '1px solid var(--color-mist-divider)', background: '#fff',
                fontSize: 15, color: 'var(--color-fog-text)', cursor: 'pointer',
              }}
            >
              ✕
            </button>
          </div>
        </div>

        {/* report body */}
        <iframe
          ref={iframeRef}
          src={report.url}
          title="Forensic report"
          style={{ flex: 1, width: '100%', border: 'none', background: '#fff' }}
        />
      </div>
    </>
  )
}

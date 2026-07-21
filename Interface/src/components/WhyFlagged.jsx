import React from 'react'

// Plain-language verdict rationale. Renders entry.explanation, the honest
// decomposition of the fusion the backend computes (pipeline/explain.py):
// what drove the score, consensus vs. peak-evidence, and the reasons.

const ACCENT = { green: '#16a34a', amber: '#d97706', red: '#dc2626', grey: '#6b7280' }
const TINT = { green: '#f0fdf4', amber: '#fffbeb', red: '#fef2f2', grey: '#f9fafb' }

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

function barColor(v) {
  if (v >= 0.70) return '#dc2626'
  if (v >= 0.30) return '#d97706'
  return '#16a34a'
}

function ContributionRow({ c, isDriver, maxContribution }) {
  // Bar width is the component's share of the verdict, scaled to the largest
  // share so the dominant driver reads as full-width.
  const pct = maxContribution > 0 ? Math.round((c.contribution / maxContribution) * 100) : 0
  return (
    <div style={{ padding: '6px 0', borderBottom: '1px solid var(--color-mist-divider)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span style={{ fontSize: 12, fontWeight: isDriver ? 700 : 500, color: 'var(--color-slate-text)' }}>
          {isDriver && <span style={{ color: barColor(c.score), marginRight: 4 }}>▸</span>}
          {c.label}
          <span style={{ fontSize: 10, fontWeight: 400, color: 'var(--color-ash-text)', marginLeft: 6 }}>
            score {c.score.toFixed(2)} · weight {c.weight.toFixed(2)}
          </span>
        </span>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--color-ash-text)' }}>
          {Math.round(c.contribution * 100)}% of verdict
        </span>
      </div>
      <div style={{ height: 4, borderRadius: 2, background: 'var(--color-mist-divider)', marginTop: 5, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: barColor(c.score), borderRadius: 2,
          transition: 'width 0.3s ease',
        }} />
      </div>
    </div>
  )
}

export default function WhyFlagged({ entry }) {
  const ex = entry?.explanation
  if (!ex) return null

  const accent = ACCENT[ex.state] || ACCENT.grey
  const tint = TINT[ex.state] || TINT.grey
  const contribs = ex.contributions || []
  const maxContribution = Math.max(...contribs.map(c => c.contribution), 0.001)

  const mechanismChip = ex.state === 'green' ? null : (
    <span style={{
      fontSize: 9, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
      padding: '2px 7px', borderRadius: 4,
      color: accent, background: tint, border: `1px solid ${accent}33`,
    }}>
      {ex.mechanism === 'peak' ? 'Peak evidence' : 'Consensus'}
    </span>
  )

  return (
    <div style={card}>
      <div style={{ ...cardHead, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Why this verdict</span>
        {mechanismChip}
      </div>
      <div style={{ padding: '12px 14px' }}>
        {/* Headline */}
        <div style={{
          fontSize: 13, fontWeight: 700, color: accent, lineHeight: 1.4, marginBottom: 10,
        }}>
          {ex.headline}
        </div>

        {/* Reasons */}
        {ex.reasons && ex.reasons.length > 0 && (
          <ul style={{ margin: '0 0 12px', paddingLeft: 18, listStyle: 'none' }}>
            {ex.reasons.map((r, i) => (
              <li key={i} style={{
                fontSize: 12, color: 'var(--color-slate-text)', lineHeight: 1.55,
                marginBottom: 5, position: 'relative',
              }}>
                <span style={{ position: 'absolute', left: -14, color: accent }}>•</span>
                {r}
              </li>
            ))}
          </ul>
        )}

        {/* Contribution breakdown */}
        {contribs.length > 0 && (
          <>
            <div style={{
              fontSize: 9, fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
              color: 'var(--color-fog-text)', marginBottom: 2,
            }}>
              How each detector fed the score
            </div>
            {contribs.map(c => (
              <ContributionRow
                key={c.name}
                c={c}
                isDriver={c.name === ex.driver}
                maxContribution={maxContribution}
              />
            ))}
          </>
        )}
      </div>
    </div>
  )
}

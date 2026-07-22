import React from 'react'
import { VoiceShieldProvider, useVoiceShield } from './context/VoiceShieldContext.jsx'
import NavBar from './components/NavBar.jsx'
import SourcePanel from './components/SourcePanel.jsx'
import RiskDisplay from './components/RiskDisplay.jsx'
import TimelineTable from './components/TimelineTable.jsx'
import StatusBar from './components/StatusBar.jsx'
import ReportsView from './components/ReportsView.jsx'
import ReportPane from './components/ReportPane.jsx'

function Body() {
  const { state } = useVoiceShield()
  if (state.view === 'reports') {
    return (
      <div style={{ gridColumn: '1 / -1', gridRow: '2', overflowY: 'auto' }}>
        <ReportsView />
      </div>
    )
  }
  return (
    <>
      <div style={{ gridColumn: '1', gridRow: '2', overflowY: 'auto', borderRight: '1px solid var(--color-mist-divider)' }}>
        <SourcePanel />
      </div>
      <div style={{ gridColumn: '2', gridRow: '2', overflowY: 'auto' }}>
        <RiskDisplay />
      </div>
      <div style={{ gridColumn: '3', gridRow: '2', display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--color-mist-divider)', overflow: 'hidden' }}>
        <TimelineTable />
      </div>
    </>
  )
}

export default function App() {
  return (
    <VoiceShieldProvider>
      <div style={{
        display: 'grid',
        gridTemplateRows: '56px 1fr 32px',
        gridTemplateColumns: '268px 1fr 336px',
        height: '100vh',
        background: 'var(--color-cream-surface)',
        overflow: 'hidden',
      }}>
        <div style={{ gridColumn: '1 / -1', gridRow: '1' }}>
          <NavBar />
        </div>
        <Body />
        <div style={{ gridColumn: '1 / -1', gridRow: '3' }}>
          <StatusBar />
        </div>
        <ReportPane />
      </div>
    </VoiceShieldProvider>
  )
}

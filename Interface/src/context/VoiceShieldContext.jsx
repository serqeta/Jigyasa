import React, { createContext, useCallback, useContext, useEffect, useReducer, useRef } from 'react'

const VoiceShieldContext = createContext(null)

const initialState = {
  serverUrl: 'http://localhost:8000',
  amberThreshold: 0.30,
  redThreshold: 0.70,
  snrFloor: 8.0,
  modelMode: 'auto',
  windowSeconds: 4,

  mode: 'file',
  wsStatus: 'idle',
  health: null,
  healthLatencyMs: null,

  entries: [],
  current: null,
  summary: null,

  analysisStatus: 'idle',
  analysisError: null,

  selectedFile: null,
  fileInfo: null,

  paused: false,
  chunkCount: 0,
  lastUpdateTime: null,
  selectedEntry: null,
  filterState: 'all',
}

function reducer(state, action) {
  switch (action.type) {
    case 'SET_CONFIG':
      return { ...state, ...action.payload }
    case 'SET_MODE':
      return { ...state, mode: action.payload }
    case 'SET_WS_STATUS':
      return { ...state, wsStatus: action.payload }
    case 'SET_HEALTH':
      return { ...state, health: action.payload.status, healthLatencyMs: action.payload.latency }
    case 'SET_FILE':
      return { ...state, selectedFile: action.payload.file, fileInfo: action.payload.info }
    case 'SET_ANALYSIS_STATUS':
      return { ...state, analysisStatus: action.payload }
    case 'SET_ANALYSIS_ERROR':
      return { ...state, analysisError: action.payload, analysisStatus: 'error' }
    case 'BATCH_RESULTS':
      return {
        ...state,
        entries: action.payload.entries,
        current: action.payload.entries[action.payload.entries.length - 1] || null,
        summary: action.payload.summary,
        analysisStatus: 'done',
        chunkCount: action.payload.entries.length,
      }
    case 'STREAM_ENTRY': {
      if (state.paused) return state
      const newEntries = [...state.entries, action.payload].slice(-100)
      return {
        ...state,
        entries: newEntries,
        current: action.payload,
        chunkCount: state.chunkCount + 1,
        lastUpdateTime: Date.now(),
      }
    }
    case 'RESET':
      return {
        ...state,
        entries: [],
        current: null,
        summary: null,
        chunkCount: 0,
        analysisStatus: 'idle',
        analysisError: null,
        selectedEntry: null,
      }
    case 'SET_PAUSED':
      return { ...state, paused: action.payload }
    case 'SET_SELECTED_ENTRY':
      return { ...state, selectedEntry: action.payload }
    case 'SET_FILTER_STATE':
      return { ...state, filterState: action.payload }
    case 'CLEAR_TIMELINE':
      return { ...state, entries: [], current: null, chunkCount: 0, summary: null, selectedEntry: null }
    default:
      return state
  }
}

export function VoiceShieldProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const wsRef = useRef(null)
  const abortRef = useRef(null)
  const stateRef = useRef(state)
  stateRef.current = state

  // Health check
  const checkHealth = useCallback(async () => {
    const url = stateRef.current.serverUrl
    const t0 = Date.now()
    try {
      const res = await fetch(`${url}/healthz`, { signal: AbortSignal.timeout(3000) })
      const latency = Date.now() - t0
      dispatch({ type: 'SET_HEALTH', payload: { status: res.ok ? 'ok' : 'error', latency } })
    } catch {
      dispatch({ type: 'SET_HEALTH', payload: { status: 'error', latency: null } })
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const timer = setInterval(checkHealth, 5000)
    return () => clearInterval(timer)
  }, [checkHealth])

  // WS connect
  const connectWs = useCallback(() => {
    if (wsRef.current) wsRef.current.close()
    const base = stateRef.current.serverUrl
    const wsUrl = base.replace('https://', 'wss://').replace('http://', 'ws://') + '/v1/ws/risk'
    dispatch({ type: 'SET_WS_STATUS', payload: 'connecting' })
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    ws.onopen = () => dispatch({ type: 'SET_WS_STATUS', payload: 'connected' })
    ws.onmessage = (e) => {
      try { dispatch({ type: 'STREAM_ENTRY', payload: JSON.parse(e.data) }) } catch {}
    }
    ws.onerror = () => dispatch({ type: 'SET_WS_STATUS', payload: 'error' })
    ws.onclose = () => dispatch({ type: 'SET_WS_STATUS', payload: 'disconnected' })
  }, [])

  const disconnectWs = useCallback(() => {
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null }
    dispatch({ type: 'SET_WS_STATUS', payload: 'idle' })
  }, [])

  useEffect(() => {
    if (state.mode === 'stream') connectWs()
    else disconnectWs()
    return disconnectWs
  }, [state.mode]) // eslint-disable-line

  // Analyze file
  const analyzeFile = useCallback(async (file) => {
    if (!file) return
    dispatch({ type: 'RESET' })
    dispatch({ type: 'SET_ANALYSIS_STATUS', payload: 'analyzing' })
    const controller = new AbortController()
    abortRef.current = controller
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${stateRef.current.serverUrl}/v1/analyze`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      dispatch({ type: 'BATCH_RESULTS', payload: data })
    } catch (e) {
      if (e.name === 'AbortError') dispatch({ type: 'SET_ANALYSIS_STATUS', payload: 'cancelled' })
      else dispatch({ type: 'SET_ANALYSIS_ERROR', payload: e.message })
    }
  }, [])

  const cancelAnalysis = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  const resetState = useCallback(async () => {
    dispatch({ type: 'RESET' })
    if (stateRef.current.mode === 'stream') {
      try { await fetch(`${stateRef.current.serverUrl}/v1/reset`, { method: 'POST' }) } catch {}
    }
  }, [])

  const exportJson = useCallback(() => {
    const data = { summary: stateRef.current.summary, entries: stateRef.current.entries }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `voiceshield-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  const exportCsv = useCallback(() => {
    const header = 'time,score,state,snr_db,top_artifact,first_amber_t,first_red_t'
    const rows = stateRef.current.entries.map(e =>
      `${e.time},${e.score},${e.state},${e.snr_db},${e.top_artifact || ''},${e.first_amber_t ?? ''},${e.first_red_t ?? ''}`
    )
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `voiceshield-${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [])

  return (
    <VoiceShieldContext.Provider value={{
      state, dispatch,
      connectWs, disconnectWs,
      analyzeFile, cancelAnalysis,
      resetState, exportJson, exportCsv,
      checkHealth,
    }}>
      {children}
    </VoiceShieldContext.Provider>
  )
}

export function useVoiceShield() {
  const ctx = useContext(VoiceShieldContext)
  if (!ctx) throw new Error('useVoiceShield must be within VoiceShieldProvider')
  return ctx
}

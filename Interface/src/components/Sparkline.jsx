import React, { useEffect, useRef } from 'react'

const STATE_COLOR = {
  green: '#16a34a',
  amber: '#d97706',
  red: '#dc2626',
  grey: '#6b7280',
}

export default function Sparkline({ entries, amberThreshold = 0.30, redThreshold = 0.70, height = 90 }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const dpr = window.devicePixelRatio || 1
    const W = canvas.offsetWidth
    const H = height

    canvas.width = W * dpr
    canvas.height = H * dpr
    const ctx = canvas.getContext('2d')
    ctx.scale(dpr, dpr)

    // Background
    ctx.fillStyle = '#fafafa'
    ctx.fillRect(0, 0, W, H)

    const data = entries.slice(-40)

    // Threshold zones
    const yAmber = H - amberThreshold * H
    const yRed = H - redThreshold * H

    ctx.fillStyle = 'rgba(220,38,38,0.06)'
    ctx.fillRect(0, 0, W, yRed)
    ctx.fillStyle = 'rgba(217,119,6,0.06)'
    ctx.fillRect(0, yRed, W, yAmber - yRed)
    ctx.fillStyle = 'rgba(22,163,74,0.04)'
    ctx.fillRect(0, yAmber, W, H - yAmber)

    // Threshold dashed lines
    ctx.setLineDash([4, 4])
    ctx.lineWidth = 1

    ctx.strokeStyle = '#d97706'
    ctx.beginPath(); ctx.moveTo(0, yAmber); ctx.lineTo(W, yAmber); ctx.stroke()

    ctx.strokeStyle = '#dc2626'
    ctx.beginPath(); ctx.moveTo(0, yRed); ctx.lineTo(W, yRed); ctx.stroke()

    ctx.setLineDash([])

    if (data.length === 0) {
      ctx.fillStyle = '#a9a9ac'
      ctx.font = '11px ui-monospace, monospace'
      ctx.textAlign = 'center'
      ctx.fillText('No data', W / 2, H / 2 + 4)
      return
    }

    const step = W / Math.max(data.length - 1, 1)

    // Score polyline, state-colored segments
    for (let i = 1; i < data.length; i++) {
      const x0 = (i - 1) * step
      const x1 = i * step
      const y0 = H - data[i - 1].score * H
      const y1 = H - data[i].score * H

      const grad = ctx.createLinearGradient(x0, 0, x1, 0)
      grad.addColorStop(0, STATE_COLOR[data[i - 1].state] || '#6b7280')
      grad.addColorStop(1, STATE_COLOR[data[i].state] || '#6b7280')

      ctx.beginPath()
      ctx.moveTo(x0, y0)
      ctx.lineTo(x1, y1)
      ctx.strokeStyle = grad
      ctx.lineWidth = 2.5
      ctx.stroke()
    }

    // Dots
    data.forEach((e, i) => {
      const x = i * step
      const y = H - e.score * H
      ctx.beginPath()
      ctx.arc(x, y, 2.5, 0, Math.PI * 2)
      ctx.fillStyle = STATE_COLOR[e.state] || '#6b7280'
      ctx.fill()
    })

    // Threshold labels
    ctx.font = '9px ui-monospace, monospace'
    ctx.textAlign = 'right'
    ctx.fillStyle = '#d97706'
    ctx.fillText(`${(amberThreshold * 100).toFixed(0)}`, W - 3, yAmber - 3)
    ctx.fillStyle = '#dc2626'
    ctx.fillText(`${(redThreshold * 100).toFixed(0)}`, W - 3, yRed - 3)

    // Latest score
    if (data.length > 0) {
      const last = data[data.length - 1]
      ctx.font = '11px ui-monospace, monospace'
      ctx.textAlign = 'left'
      ctx.fillStyle = STATE_COLOR[last.state] || '#6b7280'
      ctx.fillText(last.score.toFixed(3), 4, 14)
    }

  }, [entries, amberThreshold, redThreshold, height])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height, display: 'block', borderRadius: 6 }}
    />
  )
}

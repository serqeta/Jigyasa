'use strict';

const ACTIONS = {
  green:  'Proceed normally — standard verification only.',
  amber:  'Send OTP to registered mobile. Preserve buffer. Escalate if score rises.',
  red:    'Do not process sensitive request. Transfer to supervisor immediately.',
  grey:   'Audio quality insufficient. Continue call; flag for manual post-call review.',
};

const SCORE_RING = new Float32Array(20);  // circular score history
let ringHead = 0;

const card       = document.getElementById('card');
const badge      = document.getElementById('badge');
const scoreEl    = document.getElementById('score');
const triggerEl  = document.getElementById('trigger');
const artifactEl = document.getElementById('artifact');
const actionEl   = document.getElementById('action');
const clockEl    = document.getElementById('clock');
const dotEl      = document.getElementById('dot');
const statusEl   = document.getElementById('status-text');
const canvas     = document.getElementById('sparkline');
const ctx        = canvas.getContext('2d');

function fmtTime(t) {
  if (t == null) return '—';
  return t.toFixed(1) + ' s';
}

function applyState(state) {
  ['green','amber','red','grey'].forEach(s => {
    card.classList.toggle('state-' + s, s === state);
    badge.classList.toggle('state-' + s, s === state);
    actionEl.classList.toggle('state-' + s, s === state);
  });
  badge.textContent = state.toUpperCase();
  actionEl.textContent = ACTIONS[state] || '';
}

function drawSparkline(state) {
  const W = canvas.offsetWidth || 500;
  const H = canvas.height;
  canvas.width = W;

  ctx.clearRect(0, 0, W, H);

  const n = SCORE_RING.length;
  const step = W / (n - 1);

  // Background
  ctx.fillStyle = 'rgba(0,0,0,0.15)';
  ctx.fillRect(0, 0, W, H);

  // Threshold lines
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;

  ctx.strokeStyle = 'rgba(245,158,11,0.4)';
  ctx.beginPath();
  const yAmber = H - 0.30 * H;
  ctx.moveTo(0, yAmber); ctx.lineTo(W, yAmber);
  ctx.stroke();

  ctx.strokeStyle = 'rgba(239,68,68,0.4)';
  ctx.beginPath();
  const yRed = H - 0.70 * H;
  ctx.moveTo(0, yRed); ctx.lineTo(W, yRed);
  ctx.stroke();

  ctx.setLineDash([]);

  // Score line
  const colorMap = { green: '#22c55e', amber: '#f59e0b', red: '#ef4444', grey: '#6b7280' };
  ctx.strokeStyle = colorMap[state] || '#6b7280';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const idx = (ringHead + i) % n;
    const x = i * step;
    const y = H - SCORE_RING[idx] * H;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function onMessage(data) {
  SCORE_RING[ringHead] = Math.min(1, Math.max(0, data.score));
  ringHead = (ringHead + 1) % SCORE_RING.length;

  const state = data.state || 'grey';
  applyState(state);
  scoreEl.textContent = data.score.toFixed(3);
  triggerEl.textContent = fmtTime(data.first_red_t ?? data.first_amber_t);
  artifactEl.textContent = data.top_artifact || '—';
  drawSparkline(state);
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/v1/ws/risk`);

  ws.onopen = () => {
    dotEl.classList.remove('disconnected');
    statusEl.textContent = 'Connected';
  };

  ws.onmessage = (evt) => {
    try { onMessage(JSON.parse(evt.data)); } catch (_) {}
  };

  ws.onclose = () => {
    dotEl.classList.add('disconnected');
    statusEl.textContent = 'Reconnecting…';
    setTimeout(connect, 2000);
  };

  ws.onerror = () => ws.close();
}

// Live clock
setInterval(() => {
  clockEl.textContent = new Date().toLocaleTimeString();
}, 1000);

// Initial sparkline (all zeros)
drawSparkline('grey');

connect();

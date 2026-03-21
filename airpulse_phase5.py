"""
AirPulse - Phase 5 v2: Location Tracking (Improved UI)
========================================================
Project   : AirPulse (Next-gen WiFi vital signs detection)
File      : airpulse_phase5.py

Changes from v1:
  - Bigger room map
  - Animated person icon
  - Fixed breathing graph
  - Larger BPM numbers
  - Alert system for abnormal BPM
  - Better overall theme and layout

How to run:
    python airpulse_phase5.py
    Open: http://localhost:8001
"""

import os
import asyncio
import math
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_DIR       = "airpulse_data"
SAMPLE_RATE     = 100
WINDOW_SIZE     = 100
STREAM_INTERVAL = 0.15

ROOM_W          = 6.0
ROOM_H          = 5.0

ACCESS_POINTS = [
    {"id": "AP1", "x": 0.0, "y": 0.0,   "color": "#60a5fa"},
    {"id": "AP2", "x": 6.0, "y": 0.0,   "color": "#34d399"},
    {"id": "AP3", "x": 6.0, "y": 5.0,   "color": "#f59e0b"},
]

BREATHING_BPM   = 15
HEART_BPM       = 72
NOISE_LEVEL     = 0.3
DEVICE          = torch.device("cpu")

# Alert thresholds
BREATH_LOW_ALERT  = 8
BREATH_HIGH_ALERT = 25
HEART_LOW_ALERT   = 50
HEART_HIGH_ALERT  = 100


# =============================================================
# LSTM MODEL
# =============================================================

class VitalSignLSTM(nn.Module):
    def __init__(self, input_size=1, hidden1=64,
                 hidden2=32, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out[:, -1, :])
        out    = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


def load_models():
    bm = VitalSignLSTM().to(DEVICE)
    hm = VitalSignLSTM().to(DEVICE)
    bp = os.path.join(MODEL_DIR, "lstm_breathing.pt")
    hp = os.path.join(MODEL_DIR, "lstm_heart.pt")
    if os.path.exists(bp) and os.path.exists(hp):
        bm.load_state_dict(
            torch.load(bp, map_location=DEVICE, weights_only=True)
        )
        hm.load_state_dict(
            torch.load(hp, map_location=DEVICE, weights_only=True)
        )
        print("  LSTM models loaded.")
    bm.eval()
    hm.eval()
    return bm, hm


# =============================================================
# PERSON SIMULATOR
# =============================================================

class PersonSimulator:
    def __init__(self):
        self.x           = ROOM_W / 2.0
        self.y           = ROOM_H / 2.0
        self.vx          = 0.03
        self.vy          = 0.02
        self.t           = 0
        self.breath_freq = BREATHING_BPM / 60.0
        self.heart_freq  = HEART_BPM / 60.0
        self.buffer      = np.zeros(WINDOW_SIZE * 4)

    def move(self):
        self.vx += np.random.randn() * 0.008
        self.vy += np.random.randn() * 0.008
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > 0.06:
            self.vx = (self.vx / speed) * 0.06
            self.vy = (self.vy / speed) * 0.06
        self.x += self.vx
        self.y += self.vy
        margin = 0.3
        if self.x < margin:
            self.x  = margin
            self.vx = abs(self.vx)
        if self.x > ROOM_W - margin:
            self.x  = ROOM_W - margin
            self.vx = -abs(self.vx)
        if self.y < margin:
            self.y  = margin
            self.vy = abs(self.vy)
        if self.y > ROOM_H - margin:
            self.y  = ROOM_H - margin
            self.vy = -abs(self.vy)

    def get_rssi(self):
        rssi_values = []
        for ap in ACCESS_POINTS:
            dist = math.sqrt(
                (self.x - ap["x"])**2 + (self.y - ap["y"])**2
            )
            dist = max(dist, 0.1)
            rssi = -30.0 - 10 * 2.8 * math.log10(dist)
            rssi += np.random.randn() * 2.5
            rssi_values.append(round(rssi, 2))
        return rssi_values

    def get_csi_sample(self):
        t = self.t / SAMPLE_RATE
        s = (
            4.0  * np.sin(2 * np.pi * self.breath_freq * t)
            + 0.35 * np.sin(2 * np.pi * self.heart_freq  * t)
            + 0.05 * np.sin(4 * np.pi * self.heart_freq  * t)
            + NOISE_LEVEL * np.random.randn()
        )
        self.buffer      = np.roll(self.buffer, -1)
        self.buffer[-1]  = s
        self.t          += 1
        return float(s)

    def step(self):
        self.move()
        return {
            "x"      : round(self.x, 3),
            "y"      : round(self.y, 3),
            "rssi"   : self.get_rssi(),
            "sample" : self.get_csi_sample(),
            "buffer" : self.buffer.copy(),
        }


# =============================================================
# TRILATERATION + SIGNAL PROCESSING
# =============================================================

def trilaterate(rssi_values):
    distances = []
    for rssi in rssi_values:
        dist = 10 ** ((-30.0 - rssi) / (10 * 2.8))
        distances.append(round(dist, 3))

    def error(pos):
        ex, ey = pos
        return sum(
            (math.sqrt((ex - ap["x"])**2 + (ey - ap["y"])**2) - d)**2
            for ap, d in zip(ACCESS_POINTS, distances)
        )

    result = minimize(
        error, x0=[ROOM_W / 2, ROOM_H / 2],
        method="Nelder-Mead",
        options={"xatol": 0.01, "fatol": 0.01, "maxiter": 200},
    )
    est_x = float(np.clip(result.x[0], 0, ROOM_W))
    est_y = float(np.clip(result.x[1], 0, ROOM_H))
    return round(est_x, 2), round(est_y, 2), distances


def bandpass(signal, low, high, fs=SAMPLE_RATE):
    nyq  = fs / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def predict_bpm(model, window):
    mn, mx = window.min(), window.max()
    norm   = (
        2.0 * (window - mn) / (mx - mn) - 1.0
        if mx > mn else window
    )
    x = torch.tensor(
        norm.astype(np.float32).reshape(1, WINDOW_SIZE, 1)
    ).to(DEVICE)
    with torch.no_grad():
        return round(max(0.0, model(x).item()), 1)


def get_confidence(sig, lo, hi):
    mag  = np.abs(np.fft.rfft(sig[-WINDOW_SIZE:]))
    freq = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / SAMPLE_RATE)
    mask = (freq >= lo) & (freq <= hi)
    if not mask.any() or mag[mask].sum() == 0:
        return 0.0
    return round(min(mag[mask].max() / mag[mask].sum() * 200, 99.0), 1)


# =============================================================
# DASHBOARD HTML
# =============================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AirPulse - Live Dashboard</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }

:root {
  --bg:       #090d13;
  --surface:  #0f1520;
  --border:   #1c2333;
  --muted:    #4a5568;
  --text:     #e2e8f0;
  --green:    #34d399;
  --amber:    #fbbf24;
  --blue:     #60a5fa;
  --purple:   #a78bfa;
  --red:      #f87171;
  --green-bg: rgba(52,211,153,.08);
  --amber-bg: rgba(251,191,36,.08);
  --red-bg:   rgba(248,113,113,.12);
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  padding: 18px 22px;
  min-height: 100vh;
}

/* Header */
.header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--border);
}
.logo { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
.badge {
  font-size: 10px; font-weight: 600;
  padding: 3px 10px; border-radius: 20px;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-live {
  background: rgba(52,211,153,.15);
  color: var(--green);
  border: 1px solid rgba(52,211,153,.25);
}
.badge-phase {
  background: rgba(167,139,250,.12);
  color: var(--purple);
  border: 1px solid rgba(167,139,250,.25);
}
.pulse-dot {
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 0 rgba(52,211,153,.4);
  animation: pulseAnim 2s infinite;
}
@keyframes pulseAnim {
  0%   { box-shadow: 0 0 0 0 rgba(52,211,153,.4); }
  70%  { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}

/* Alert banner */
.alert-banner {
  display: none;
  align-items: center;
  gap: 10px;
  background: var(--red-bg);
  border: 1px solid rgba(248,113,113,.3);
  border-radius: 10px;
  padding: 10px 16px;
  margin-bottom: 16px;
  font-size: 13px;
  color: var(--red);
  font-weight: 500;
}
.alert-banner.show { display: flex; }
.alert-icon {
  width: 18px; height: 18px;
  background: var(--red);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: #000; font-weight: 700; flex-shrink: 0;
}

/* Metric cards */
.metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 18px;
}
.mcard {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 20px;
  position: relative;
  overflow: hidden;
  transition: border-color .3s;
}
.mcard.alert-card {
  border-color: rgba(248,113,113,.5);
  background: var(--red-bg);
}
.mcard-accent {
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  border-radius: 14px 0 0 14px;
}
.mcard-label {
  font-size: 11px; font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 8px;
}
.mcard-value {
  font-size: 36px; font-weight: 700;
  line-height: 1; margin-bottom: 4px;
  letter-spacing: -1px;
}
.mcard-sub {
  font-size: 11px; color: var(--muted);
}
.conf-bar {
  height: 3px; border-radius: 2px;
  background: var(--border);
  margin-top: 8px; overflow: hidden;
}
.conf-fill {
  height: 100%; border-radius: 2px;
  transition: width .5s ease;
}

/* Main layout */
.main {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 16px;
}
@media(max-width:900px){ .main{ grid-template-columns:1fr; } }

/* Panel */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
}
.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.panel-title {
  font-size: 11px; font-weight: 600;
  color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.8px;
}
.panel-badge {
  font-size: 10px; padding: 2px 8px;
  border-radius: 6px; font-weight: 600;
}

/* Map canvas */
#mapCanvas {
  display: block;
  width: 100% !important;
  border-radius: 10px;
  background: #060a10;
}

/* RSSI bars */
.rssi-section { margin-top: 14px; }
.rssi-row {
  display: flex; align-items: center;
  gap: 10px; margin-bottom: 8px; font-size: 12px;
}
.rssi-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink:0; }
.rssi-label { min-width: 32px; color: var(--muted); font-weight: 600; }
.rssi-track {
  flex: 1; height: 6px; background: var(--border);
  border-radius: 3px; overflow: hidden;
}
.rssi-fill { height: 100%; border-radius: 3px; transition: width .4s ease; }
.rssi-val { min-width: 68px; text-align: right; color: var(--muted); font-size: 11px; }

/* Right column charts */
.charts-col { display: flex; flex-direction: column; gap: 12px; }
.chart-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}
.chart-label {
  font-size: 10px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.8px;
  margin-bottom: 8px;
  display: flex; align-items: center; justify-content: space-between;
}
canvas { display: block; width: 100% !important; }

/* Alerts log */
.alerts-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
  max-height: 120px;
  overflow-y: auto;
}
.alert-item {
  font-size: 11px;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
  color: var(--red);
  display: flex; gap: 8px;
}
.alert-time { color: var(--muted); flex-shrink: 0; }
.no-alerts { font-size: 11px; color: var(--muted); text-align: center; padding: 8px 0; }

/* Person animation */
@keyframes personPulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50%      { opacity: 0.7; transform: scale(1.15); }
}
</style>
</head>
<body>

<div class="header">
  <div class="pulse-dot"></div>
  <span class="logo">AirPulse</span>
  <span class="badge badge-live">Live</span>
  <span class="badge badge-phase">Phase 5 - Location Tracking</span>
  <span style="margin-left:auto;font-size:11px;color:var(--muted)" id="uptime">00:00</span>
</div>

<div class="alert-banner" id="alertBanner">
  <div class="alert-icon">!</div>
  <span id="alertText">Alert</span>
</div>

<!-- Metric cards -->
<div class="metrics">
  <div class="mcard" id="bcard">
    <div class="mcard-accent" style="background:var(--green)"></div>
    <div class="mcard-label">Breathing rate</div>
    <div class="mcard-value" id="bbpm" style="color:var(--green)">--</div>
    <div class="mcard-sub" id="bsub">Waiting...</div>
    <div class="conf-bar">
      <div class="conf-fill" id="bconf-fill"
           style="width:0%;background:var(--green)"></div>
    </div>
  </div>
  <div class="mcard" id="hcard">
    <div class="mcard-accent" style="background:var(--amber)"></div>
    <div class="mcard-label">Heart rate</div>
    <div class="mcard-value" id="hbpm" style="color:var(--amber)">--</div>
    <div class="mcard-sub" id="hsub">Waiting...</div>
    <div class="conf-bar">
      <div class="conf-fill" id="hconf-fill"
           style="width:0%;background:var(--amber)"></div>
    </div>
  </div>
  <div class="mcard">
    <div class="mcard-accent" style="background:var(--blue)"></div>
    <div class="mcard-label">Estimated position</div>
    <div class="mcard-value" id="pos"
         style="color:var(--blue);font-size:22px;letter-spacing:-0.5px">--</div>
    <div class="mcard-sub" id="poserr">Trilateration</div>
  </div>
  <div class="mcard">
    <div class="mcard-accent" style="background:var(--purple)"></div>
    <div class="mcard-label">Presence</div>
    <div class="mcard-value" id="pres" style="color:var(--purple)">--</div>
    <div class="mcard-sub" id="presub">--</div>
  </div>
</div>

<!-- Main layout -->
<div class="main">

  <!-- Left: Room map -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Room map</span>
      <span class="panel-badge"
            style="background:rgba(96,165,250,.12);color:var(--blue)">
        6m x 5m
      </span>
    </div>
    <canvas id="mapCanvas" height="380"></canvas>
    <div class="rssi-section" id="rssi-bars"></div>
  </div>

  <!-- Right: Charts + Alerts -->
  <div class="charts-col">

    <div class="chart-panel">
      <div class="chart-label">
        <span>Raw CSI signal</span>
        <span style="color:var(--blue);font-weight:400" id="snr-lbl">SNR: --</span>
      </div>
      <canvas id="raw" height="80"></canvas>
    </div>

    <div class="chart-panel">
      <div class="chart-label">
        <span>Breathing filtered (0.1-0.5 Hz)</span>
      </div>
      <canvas id="breath" height="80"></canvas>
    </div>

    <div class="chart-panel">
      <div class="chart-label">
        <span>Heart rate filtered (0.67-2.0 Hz)</span>
      </div>
      <canvas id="heart" height="80"></canvas>
    </div>

    <div class="chart-panel">
      <div class="chart-label">
        <span>BPM history</span>
        <span style="font-size:10px;font-weight:400">
          <span style="color:var(--green)">Breathing</span>
          &nbsp;
          <span style="color:var(--amber)">Heart</span>
        </span>
      </div>
      <canvas id="hist" height="70"></canvas>
    </div>

    <div>
      <div class="chart-label" style="margin-bottom:8px">
        <span>Alerts log</span>
      </div>
      <div class="alerts-panel" id="alerts-log">
        <div class="no-alerts">No alerts</div>
      </div>
    </div>

  </div>
</div>

<script>
const RW=6, RH=5;
const APS=[
  {id:"AP1",x:0,y:0,color:"#60a5fa"},
  {id:"AP2",x:6,y:0,color:"#34d399"},
  {id:"AP3",x:6,y:5,color:"#fbbf24"},
];
const BREATH_LOW=8, BREATH_HIGH=25, HEART_LOW=50, HEART_HIGH=100;

let trail=[], alerts=[], startTime=Date.now(), frame=0;

// Uptime counter
setInterval(()=>{
  const s=Math.floor((Date.now()-startTime)/1000);
  const m=Math.floor(s/60).toString().padStart(2,"0");
  const sc=(s%60).toString().padStart(2,"0");
  document.getElementById("uptime").textContent=m+":"+sc;
},1000);

// =============================================================
// ROOM MAP
// =============================================================
function drawMap(tx,ty,ex,ey,rssi){
  const c=document.getElementById("mapCanvas");
  const W=c.offsetWidth, H=380;
  c.width=W; c.height=H;
  const ctx=c.getContext("2d");
  const pad=42;
  const sx=(W-pad*2)/RW, sy=(H-pad*2)/RH;
  const tc=(rx,ry)=>[pad+rx*sx, H-pad-ry*sy];

  // Background
  ctx.fillStyle="#060a10"; ctx.fillRect(0,0,W,H);

  // Grid
  ctx.strokeStyle="#0d1520"; ctx.lineWidth=1;
  for(let x=0;x<=RW;x++){
    const[cx]=tc(x,0);
    ctx.beginPath();ctx.moveTo(cx,pad);ctx.lineTo(cx,H-pad);ctx.stroke();
  }
  for(let y=0;y<=RH;y++){
    const[,cy]=tc(0,y);
    ctx.beginPath();ctx.moveTo(pad,cy);ctx.lineTo(W-pad,cy);ctx.stroke();
  }

  // Room border
  ctx.strokeStyle="#1c2333"; ctx.lineWidth=2;
  ctx.strokeRect(pad,pad,W-pad*2,H-pad*2);

  // Scale labels
  ctx.fillStyle="#2d3748"; ctx.font="10px sans-serif"; ctx.textAlign="center";
  for(let x=0;x<=RW;x++){
    const[cx]=tc(x,0);
    ctx.fillText(x+"m",cx,H-pad+16);
  }
  ctx.textAlign="right";
  for(let y=0;y<=RH;y++){
    const[,cy]=tc(0,y);
    ctx.fillText(y+"m",pad-8,cy+4);
  }
  ctx.textAlign="left";

  // AP signal rings
  APS.forEach((ap,i)=>{
    if(!rssi||rssi[i]==null) return;
    const[cx,cy]=tc(ap.x,ap.y);
    const dist=Math.pow(10,(-30-rssi[i])/28);
    const r=dist*sx;
    ctx.strokeStyle=ap.color; ctx.lineWidth=1;
    ctx.globalAlpha=0.08;
    ctx.beginPath(); ctx.arc(cx,cy,Math.min(r,W*2),0,Math.PI*2); ctx.stroke();
    ctx.globalAlpha=0.04;
    ctx.fillStyle=ap.color;
    ctx.beginPath(); ctx.arc(cx,cy,Math.min(r,W*2),0,Math.PI*2); ctx.fill();
    ctx.globalAlpha=1;
  });

  // Trail with gradient fade
  trail.forEach((p,i)=>{
    const[cx,cy]=tc(p.x,p.y);
    const alpha=(i/trail.length)*0.6;
    const size=1+(i/trail.length)*2;
    ctx.fillStyle=`rgba(96,165,250,${alpha})`;
    ctx.beginPath(); ctx.arc(cx,cy,size,0,Math.PI*2); ctx.fill();
  });

  // Connect trail
  if(trail.length>1){
    ctx.strokeStyle="rgba(96,165,250,0.15)"; ctx.lineWidth=1.5;
    ctx.beginPath();
    trail.forEach((p,i)=>{
      const[cx,cy]=tc(p.x,p.y);
      i===0?ctx.moveTo(cx,cy):ctx.lineTo(cx,cy);
    });
    ctx.stroke();
  }

  // Estimated position ring
  const[ex2,ey2]=tc(ex,ey);
  ctx.strokeStyle="rgba(248,113,113,0.5)"; ctx.lineWidth=1.5;
  ctx.setLineDash([5,4]);
  ctx.beginPath(); ctx.arc(ex2,ey2,12,0,Math.PI*2); ctx.stroke();
  ctx.setLineDash([]);

  // Error line
  const[tx2,ty2]=tc(tx,ty);
  ctx.strokeStyle="rgba(248,113,113,0.25)"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(tx2,ty2); ctx.lineTo(ex2,ey2); ctx.stroke();

  // Person glow
  const glow=ctx.createRadialGradient(tx2,ty2,2,tx2,ty2,22);
  glow.addColorStop(0,"rgba(96,165,250,0.35)");
  glow.addColorStop(1,"rgba(96,165,250,0)");
  ctx.fillStyle=glow;
  ctx.beginPath(); ctx.arc(tx2,ty2,22,0,Math.PI*2); ctx.fill();

  // Person icon (animated breathing ring)
  const breathScale=1+0.08*Math.sin(frame*0.15);
  ctx.strokeStyle="rgba(96,165,250,0.4)"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(tx2,ty2,14*breathScale,0,Math.PI*2); ctx.stroke();

  // Person solid dot
  ctx.fillStyle="#60a5fa";
  ctx.beginPath(); ctx.arc(tx2,ty2,7,0,Math.PI*2); ctx.fill();
  ctx.fillStyle="#fff";
  ctx.beginPath(); ctx.arc(tx2,ty2,3,0,Math.PI*2); ctx.fill();

  // AP icons
  APS.forEach(ap=>{
    const[cx,cy]=tc(ap.x,ap.y);
    ctx.fillStyle=ap.color+"22";
    ctx.beginPath(); ctx.arc(cx,cy,12,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=ap.color;
    ctx.beginPath(); ctx.arc(cx,cy,7,0,Math.PI*2); ctx.fill();
    ctx.fillStyle="#000"; ctx.font="bold 7px sans-serif"; ctx.textAlign="center";
    ctx.fillText(ap.id,cx,cy+2.5);
    ctx.textAlign="left";
  });

  frame++;
}

// =============================================================
// RSSI BARS
// =============================================================
function updateRSSI(rssi){
  const el=document.getElementById("rssi-bars");
  el.innerHTML="";
  APS.forEach((ap,i)=>{
    const v=rssi[i]||-90;
    const pct=Math.max(0,Math.min(100,(v+90)/60*100));
    el.innerHTML+=`<div class="rssi-row">
      <div class="rssi-dot" style="background:${ap.color}"></div>
      <span class="rssi-label">${ap.id}</span>
      <div class="rssi-track">
        <div class="rssi-fill" style="width:${pct}%;background:${ap.color}"></div>
      </div>
      <span class="rssi-val">${v.toFixed(1)} dBm</span>
    </div>`;
  });
}

// =============================================================
// WAVEFORM CHARTS
// =============================================================
function makeWave(id,color,yMin,yMax){
  const c=document.getElementById(id);
  const ctx=c.getContext("2d");
  let data=new Array(250).fill(0);
  return {push(v){
    data.push(v); if(data.length>250) data.shift();
    const W=c.offsetWidth||400, H=c.offsetHeight||80;
    c.width=W; c.height=H;
    ctx.fillStyle="#060a10"; ctx.fillRect(0,0,W,H);

    // Zero line
    const zy=H-((0-yMin)/(yMax-yMin))*H;
    ctx.strokeStyle="#1c2333"; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(0,zy); ctx.lineTo(W,zy); ctx.stroke();

    // Signal
    ctx.strokeStyle=color; ctx.lineWidth=1.5;
    ctx.shadowColor=color; ctx.shadowBlur=3;
    ctx.beginPath();
    data.forEach((d,i)=>{
      const x=(i/(data.length-1))*W;
      const y=H-((d-yMin)/(yMax-yMin))*H;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.shadowBlur=0;

    // Fill under curve
    ctx.fillStyle=color.replace(")",",0.06)").replace("rgb","rgba").replace("#","rgba(")+"";
    const gradient=ctx.createLinearGradient(0,0,0,H);
    gradient.addColorStop(0, color+"33");
    gradient.addColorStop(1, color+"00");
    ctx.fillStyle=gradient;
    ctx.beginPath();
    data.forEach((d,i)=>{
      const x=(i/(data.length-1))*W;
      const y=H-((d-yMin)/(yMax-yMin))*H;
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    });
    ctx.lineTo(W,H); ctx.lineTo(0,H); ctx.closePath(); ctx.fill();
  }};
}

function makeHist(){
  const c=document.getElementById("hist");
  const ctx=c.getContext("2d");
  let bd=new Array(100).fill(0), hd=new Array(100).fill(0);
  return {push(b,h){
    bd.push(b); if(bd.length>100) bd.shift();
    hd.push(h); if(hd.length>100) hd.shift();
    const W=c.offsetWidth||400, H=c.offsetHeight||70;
    c.width=W; c.height=H;
    ctx.fillStyle="#060a10"; ctx.fillRect(0,0,W,H);

    // Horizontal guides
    [30,60,90,120].forEach(v=>{
      const y=H-(v/150)*H;
      ctx.strokeStyle="#0d1520"; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
      ctx.fillStyle="#2d3748"; ctx.font="9px sans-serif";
      ctx.fillText(v,2,y-2);
    });

    [[bd,"#34d399"],[hd,"#fbbf24"]].forEach(([d,col])=>{
      ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.beginPath();
      d.forEach((v,i)=>{
        const x=(i/(d.length-1))*W;
        const y=H-((Math.max(0,v)/150))*H;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      });
      ctx.stroke();
    });
  }};
}

const rawC  = makeWave("raw",    "#60a5fa", -8,   8);
const bthC  = makeWave("breath", "#34d399", -5,   5);
const hrtC  = makeWave("heart",  "#fbbf24", -1.5, 1.5);
const hstC  = makeHist();

// =============================================================
// ALERTS
// =============================================================
function checkAlerts(bbpm, hbpm){
  const msgs = [];
  if(bbpm>0 && bbpm < BREATH_LOW)  msgs.push("Low breathing rate: "+bbpm+" BPM");
  if(bbpm > BREATH_HIGH)           msgs.push("High breathing rate: "+bbpm+" BPM");
  if(hbpm>0 && hbpm < HEART_LOW)   msgs.push("Low heart rate: "+hbpm+" BPM");
  if(hbpm > HEART_HIGH)            msgs.push("High heart rate: "+hbpm+" BPM");

  const banner = document.getElementById("alertBanner");
  const bcard  = document.getElementById("bcard");
  const hcard  = document.getElementById("hcard");

  if(msgs.length>0){
    banner.classList.add("show");
    document.getElementById("alertText").textContent = msgs.join("  |  ");
    if(bbpm < BREATH_LOW || bbpm > BREATH_HIGH) bcard.classList.add("alert-card");
    else bcard.classList.remove("alert-card");
    if(hbpm < HEART_LOW  || hbpm > HEART_HIGH)  hcard.classList.add("alert-card");
    else hcard.classList.remove("alert-card");

    const log  = document.getElementById("alerts-log");
    const time = new Date().toLocaleTimeString();
    msgs.forEach(m=>{
      const no = log.querySelector(".no-alerts");
      if(no) no.remove();
      const item = document.createElement("div");
      item.className="alert-item";
      item.innerHTML=`<span class="alert-time">${time}</span><span>${m}</span>`;
      log.prepend(item);
      while(log.children.length>20) log.removeChild(log.lastChild);
    });
  } else {
    banner.classList.remove("show");
    bcard.classList.remove("alert-card");
    hcard.classList.remove("alert-card");
  }
}

// =============================================================
// WEBSOCKET
// =============================================================
const ws = new WebSocket("ws://"+location.host+"/ws");

ws.onmessage = (e) => {
  const d = JSON.parse(e.data);

  // Metrics
  document.getElementById("bbpm").textContent = d.breath_bpm+" BPM";
  document.getElementById("bsub").textContent = "Conf: "+d.breath_conf+"% | 6-30 normal";
  document.getElementById("bconf-fill").style.width = d.breath_conf+"%";

  document.getElementById("hbpm").textContent = d.heart_bpm+" BPM";
  document.getElementById("hsub").textContent = "Conf: "+d.heart_conf+"% | 50-100 normal";
  document.getElementById("hconf-fill").style.width = d.heart_conf+"%";

  document.getElementById("pos").textContent =
    "("+d.est_x+"m, "+d.est_y+"m)";
  document.getElementById("poserr").textContent =
    "Error: "+d.loc_error+"m  |  True: "+d.true_x+", "+d.true_y;

  document.getElementById("pres").textContent =
    d.presence ? "Detected" : "Empty room";
  document.getElementById("presub").textContent =
    "RSSI variance: "+d.rssi_var.toFixed(2);

  document.getElementById("snr-lbl").textContent = "SNR: "+d.snr+" dB";

  // Map
  trail.push({x:d.true_x, y:d.true_y});
  if(trail.length>50) trail.shift();
  drawMap(d.true_x,d.true_y,d.est_x,d.est_y,d.rssi);
  updateRSSI(d.rssi);

  // Charts
  rawC.push(d.raw_sample);
  bthC.push(d.breath_sample);
  hrtC.push(d.heart_sample);
  hstC.push(d.breath_bpm, d.heart_bpm);

  // Alerts
  checkAlerts(d.breath_bpm, d.heart_bpm);
};

ws.onopen  = () => { console.log("AirPulse connected"); };
ws.onclose = () => {
  document.querySelector(".pulse-dot").style.background="#ef4444";
  document.querySelector(".pulse-dot").style.animation="none";
};
</script>
</body>
</html>"""


# =============================================================
# FASTAPI APP
# =============================================================

app      = FastAPI(title="AirPulse Phase 5 v2")
sim      = PersonSimulator()
bm = hm  = None


@app.on_event("startup")
async def startup():
    global bm, hm
    print("\nLoading LSTM models...")
    bm, hm = load_models()
    print("AirPulse Phase 5 v2 ready.")
    print("Open browser: http://localhost:8001\n")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("  Client connected.")
    try:
        while True:
            state    = sim.step()
            buf      = state["buffer"]
            breath_f = bandpass(buf, 0.1,  0.5)
            heart_f  = bandpass(buf, 0.67, 2.0)
            b_bpm    = predict_bpm(bm, breath_f[-WINDOW_SIZE:])
            h_bpm    = predict_bpm(hm, heart_f[-WINDOW_SIZE:])
            est_x, est_y, dists = trilaterate(state["rssi"])

            loc_err = round(math.sqrt(
                (est_x - state["x"])**2 +
                (est_y - state["y"])**2
            ), 2)

            noise_p  = np.var(buf - breath_f - heart_f)
            signal_p = np.var(breath_f) + np.var(heart_f)
            snr      = round(
                10 * math.log10(signal_p / noise_p)
                if noise_p > 0 else 0.0, 1
            )
            rssi_var = float(np.var(state["rssi"]))

            await websocket.send_json({
                "true_x"       : round(state["x"], 2),
                "true_y"       : round(state["y"], 2),
                "est_x"        : est_x,
                "est_y"        : est_y,
                "loc_error"    : loc_err,
                "rssi"         : state["rssi"],
                "breath_bpm"   : b_bpm,
                "heart_bpm"    : h_bpm,
                "breath_conf"  : get_confidence(breath_f, 0.1,  0.5),
                "heart_conf"   : get_confidence(heart_f,  0.67, 2.0),
                "raw_sample"   : round(float(buf[-1]), 4),
                "breath_sample": round(float(breath_f[-1]), 4),
                "heart_sample" : round(float(heart_f[-1]), 4),
                "snr"          : snr,
                "presence"     : rssi_var > 1.0,
                "rssi_var"     : round(rssi_var, 3),
            })
            await asyncio.sleep(STREAM_INTERVAL)

    except WebSocketDisconnect:
        print("  Client disconnected.")


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    print("=" * 58)
    print("  AirPulse - Phase 5 v2: Improved Dashboard")
    print("  Location Tracking + Alerts + Better UI")
    print("=" * 58)
    uvicorn.run(
        "airpulse_phase5:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="warning",
    )

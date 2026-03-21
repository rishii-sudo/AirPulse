"""
AirPulse — Phase 4: Real-Time Web Dashboard
============================================
Project   : AirPulse (Next-gen WiFi vital signs detection)
File      : airpulse_phase4.py
Purpose   : Stream live vital sign predictions to a browser
            dashboard using FastAPI and WebSocket.
            The trained LSTM models from Phase 3 run in
            real-time on simulated (or real) CSI data.

How to run:
    pip install fastapi uvicorn torch numpy scipy
    python airpulse_phase4.py

Then open browser:
    http://localhost:8000

Requires:
    airpulse_data/lstm_breathing.pt   (from Phase 3)
    airpulse_data/lstm_heart.pt       (from Phase 3)
    airpulse_data/metadata.json       (from Phase 1)
"""

import os
import json
import asyncio
import numpy as np
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL_DIR       = "airpulse_data"
SAMPLE_RATE     = 100
WINDOW_SIZE     = 100
STREAM_INTERVAL = 0.1       # Seconds between updates (10 Hz)
BREATHING_BPM   = 15        # Simulated true value
HEART_BPM       = 72
NOISE_LEVEL     = 0.3
DEVICE          = torch.device("cpu")


# ─────────────────────────────────────────────────────────────
# LSTM MODEL DEFINITION (same as Phase 3)
# ─────────────────────────────────────────────────────────────

class VitalSignLSTM(nn.Module):
    def __init__(self, input_size=1, hidden1=64,
                 hidden2=32, dropout=0.2):
        super().__init__()
        self.lstm1    = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1    = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2    = nn.Dropout(dropout)
        self.fc1      = nn.Linear(hidden2, 16)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out[:, -1, :])
        out    = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Load trained LSTM models from Phase 3.

    Returns:
        breath_model : Loaded breathing LSTM
        heart_model  : Loaded heart rate LSTM
    """
    breath_model = VitalSignLSTM().to(DEVICE)
    heart_model  = VitalSignLSTM().to(DEVICE)

    breath_path = os.path.join(MODEL_DIR, "lstm_breathing.pt")
    heart_path  = os.path.join(MODEL_DIR, "lstm_heart.pt")

    if os.path.exists(breath_path) and os.path.exists(heart_path):
        breath_model.load_state_dict(
            torch.load(breath_path, map_location=DEVICE,
                       weights_only=True)
        )
        heart_model.load_state_dict(
            torch.load(heart_path, map_location=DEVICE,
                       weights_only=True)
        )
        print("  LSTM models loaded from Phase 3.")
    else:
        print("  WARNING: Phase 3 models not found.")
        print("  Using untrained models — run Phase 3 first.")

    breath_model.eval()
    heart_model.eval()
    return breath_model, heart_model


# ─────────────────────────────────────────────────────────────
# SIGNAL PIPELINE
# ─────────────────────────────────────────────────────────────

class SignalBuffer:
    """
    Rolling buffer that simulates a live CSI signal stream.

    In production: replace _generate_sample() with actual
    ESP32-S3 CSI data read from serial/UDP socket.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate    = sample_rate
        self.buffer         = np.zeros(WINDOW_SIZE * 4)
        self.t              = 0
        self.breath_freq    = BREATHING_BPM / 60.0
        self.heart_freq     = HEART_BPM / 60.0

    def _generate_sample(self) -> float:
        """Simulate one CSI sample (replace with real hardware read)."""
        t = self.t / self.sample_rate
        sample = (
            4.0  * np.sin(2 * np.pi * self.breath_freq * t)
            + 0.35 * np.sin(2 * np.pi * self.heart_freq  * t)
            + 0.05 * np.sin(4 * np.pi * self.heart_freq  * t)
            + NOISE_LEVEL * np.random.randn()
        )
        self.t += 1
        return float(sample)

    def push(self) -> np.ndarray:
        """Push one new sample and return updated buffer."""
        sample      = self._generate_sample()
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = sample
        return self.buffer.copy()


def bandpass(signal: np.ndarray, low: float,
             high: float, fs: int) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    nyq     = fs / 2.0
    b, a    = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def predict_bpm(
    model   : nn.Module,
    window  : np.ndarray,
) -> float:
    """
    Run one LSTM inference on a signal window.

    Args:
        model  : Trained LSTM model
        window : Signal window of shape (WINDOW_SIZE,)

    Returns:
        Predicted BPM as float
    """
    # Normalize
    mn, mx  = window.min(), window.max()
    if mx - mn > 0:
        norm = 2.0 * (window - mn) / (mx - mn) - 1.0
    else:
        norm = window

    x = torch.tensor(
        norm.astype(np.float32).reshape(1, WINDOW_SIZE, 1)
    ).to(DEVICE)

    with torch.no_grad():
        bpm = model(x).item()

    return round(max(0.0, bpm), 1)


def compute_confidence(filtered: np.ndarray,
                       low: float, high: float,
                       fs: int) -> float:
    """Compute detection confidence from FFT peak ratio."""
    mag     = np.abs(np.fft.rfft(filtered[-WINDOW_SIZE:]))
    freqs   = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / fs)
    mask    = (freqs >= low) & (freqs <= high)
    if mask.sum() == 0 or mag[mask].sum() == 0:
        return 0.0
    conf = float(mag[mask].max() / mag[mask].sum()) * 100
    return round(min(conf * 2, 99.0), 1)


# ─────────────────────────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AirPulse Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: #0d1117;
    color: #f0f6fc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    min-height: 100vh;
    padding: 24px;
  }

  header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
    border-bottom: 0.5px solid #21262d;
    padding-bottom: 16px;
  }

  header h1 {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.3px;
  }

  .badge {
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
  }

  .badge-live {
    background: rgba(34,197,94,0.15);
    color: #22c55e;
    border: 0.5px solid rgba(34,197,94,0.3);
  }

  .badge-sim {
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border: 0.5px solid rgba(99,102,241,0.3);
  }

  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }

  .metric-card {
    background: #161b22;
    border: 0.5px solid #21262d;
    border-radius: 12px;
    padding: 16px 20px;
  }

  .metric-label {
    font-size: 12px;
    color: #6b7280;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-value {
    font-size: 32px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
  }

  .metric-sub {
    font-size: 11px;
    color: #6b7280;
  }

  .breath-val  { color: #34d399; }
  .heart-val   { color: #f59e0b; }
  .snr-val     { color: #60a5fa; }
  .pres-val    { color: #a78bfa; }

  .charts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  @media (max-width: 700px) {
    .charts { grid-template-columns: 1fr; }
  }

  .chart-card {
    background: #161b22;
    border: 0.5px solid #21262d;
    border-radius: 12px;
    padding: 16px;
  }

  .chart-title {
    font-size: 12px;
    color: #8b949e;
    margin-bottom: 10px;
    font-weight: 500;
  }

  canvas {
    width: 100% !important;
    display: block;
  }

  .footer {
    text-align: center;
    font-size: 11px;
    color: #30363d;
    margin-top: 20px;
  }
</style>
</head>
<body>

<header>
  <div class="dot"></div>
  <h1>AirPulse</h1>
  <span class="badge badge-live">Live</span>
  <span class="badge badge-sim">Simulation mode</span>
</header>

<div class="metrics">
  <div class="metric-card">
    <div class="metric-label">Breathing rate</div>
    <div class="metric-value breath-val" id="breath-bpm">--</div>
    <div class="metric-sub" id="breath-conf">Confidence: --%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Heart rate</div>
    <div class="metric-value heart-val" id="heart-bpm">--</div>
    <div class="metric-sub" id="heart-conf">Confidence: --%</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Signal SNR</div>
    <div class="metric-value snr-val" id="snr-val">--</div>
    <div class="metric-sub">dB</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Presence</div>
    <div class="metric-value pres-val" id="presence">--</div>
    <div class="metric-sub" id="pres-sub">Detecting...</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <div class="chart-title">RAW CSI SIGNAL</div>
    <canvas id="rawChart" height="120"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">BREATHING — FILTERED (0.1–0.5 Hz)</div>
    <canvas id="breathChart" height="120"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">HEART RATE — FILTERED (0.67–2.0 Hz)</div>
    <canvas id="heartChart" height="120"></canvas>
  </div>
  <div class="chart-card">
    <div class="chart-title">BPM HISTORY (LSTM predictions)</div>
    <canvas id="histChart" height="120"></canvas>
  </div>
</div>

<div class="footer">AirPulse &mdash; WiFi vital signs detection &mdash; Phase 4</div>

<script>
const MAX_POINTS = 200;
const BPM_HIST   = 60;

function makeChart(id, color, yMin, yMax) {
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext("2d");
  let data     = new Array(MAX_POINTS).fill(0);

  function draw() {
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight || 120;
    canvas.width  = W;
    canvas.height = H;

    ctx.fillStyle = "#161b22";
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = "#21262d";
    ctx.lineWidth   = 0.5;
    for (let i = 1; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(0, H * i / 4);
      ctx.lineTo(W, H * i / 4);
      ctx.stroke();
    }

    // Signal
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (MAX_POINTS - 1)) * W;
      const y = H - ((v - yMin) / (yMax - yMin)) * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  return {
    push(v) {
      data.push(v);
      if (data.length > MAX_POINTS) data.shift();
      draw();
    },
    draw
  };
}

function makeHistChart(id) {
  const canvas = document.getElementById(id);
  const ctx    = canvas.getContext("2d");
  let bData    = new Array(BPM_HIST).fill(0);
  let hData    = new Array(BPM_HIST).fill(0);

  return {
    push(b, h) {
      bData.push(b); if (bData.length > BPM_HIST) bData.shift();
      hData.push(h); if (hData.length > BPM_HIST) hData.shift();

      const W = canvas.offsetWidth;
      const H = canvas.offsetHeight || 120;
      canvas.width  = W;
      canvas.height = H;

      ctx.fillStyle = "#161b22";
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "#21262d";
      ctx.lineWidth   = 0.5;
      for (let i = 1; i < 4; i++) {
        ctx.beginPath();
        ctx.moveTo(0, H * i / 4);
        ctx.lineTo(W, H * i / 4);
        ctx.stroke();
      }

      [[bData, "#34d399"], [hData, "#f59e0b"]].forEach(([d, col]) => {
        const mn = 0, mx = 150;
        ctx.strokeStyle = col;
        ctx.lineWidth   = 1.5;
        ctx.beginPath();
        d.forEach((v, i) => {
          const x = (i / (BPM_HIST - 1)) * W;
          const y = H - ((v - mn) / (mx - mn)) * H;
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
      });

      // Legend
      ctx.font      = "10px sans-serif";
      ctx.fillStyle = "#34d399";
      ctx.fillText("Breathing", 8, 14);
      ctx.fillStyle = "#f59e0b";
      ctx.fillText("Heart rate", 8, 28);
    }
  };
}

const rawC    = makeChart("rawChart",    "#60a5fa", -8,  8);
const breathC = makeChart("breathChart", "#34d399", -6,  6);
const heartC  = makeChart("heartChart",  "#f59e0b", -2,  2);
const histC   = makeHistChart("histChart");

// WebSocket connection
const ws = new WebSocket("ws://" + location.host + "/ws");

ws.onmessage = (event) => {
  const d = JSON.parse(event.data);

  document.getElementById("breath-bpm").textContent =
    d.breath_bpm + " BPM";
  document.getElementById("breath-conf").textContent =
    "Confidence: " + d.breath_conf + "%";
  document.getElementById("heart-bpm").textContent =
    d.heart_bpm + " BPM";
  document.getElementById("heart-conf").textContent =
    "Confidence: " + d.heart_conf + "%";
  document.getElementById("snr-val").textContent =
    d.snr.toFixed(1);
  document.getElementById("presence").textContent =
    d.presence ? "Detected" : "Empty";
  document.getElementById("pres-sub").textContent =
    "Variance: " + d.rssi_var.toFixed(2);

  rawC.push(d.raw_sample);
  breathC.push(d.breath_sample);
  heartC.push(d.heart_sample);
  histC.push(d.breath_bpm, d.heart_bpm);
};

ws.onclose = () => {
  document.querySelector(".dot").style.background = "#ef4444";
};
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────

app             = FastAPI(title="AirPulse Dashboard")
signal_buffer   = SignalBuffer()
breath_model    = None
heart_model     = None


@app.on_event("startup")
async def startup():
    global breath_model, heart_model
    print("\nLoading LSTM models...")
    breath_model, heart_model = load_models()
    print("AirPulse server ready.")
    print("Open browser: http://localhost:8000\n")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML page."""
    return DASHBOARD_HTML


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Stream live vital sign data to the browser via WebSocket.

    Sends a JSON packet every STREAM_INTERVAL seconds containing:
      - Raw CSI sample
      - Filtered samples (breathing + heart)
      - LSTM BPM predictions
      - Confidence scores
      - SNR and presence detection
    """
    await websocket.accept()
    print(f"  Client connected: {websocket.client}")

    try:
        while True:
            # Get new signal sample
            buffer  = signal_buffer.push()

            # Apply bandpass filters on full buffer
            breath_f    = bandpass(buffer, 0.1,  0.5,  SAMPLE_RATE)
            heart_f     = bandpass(buffer, 0.67, 2.0,  SAMPLE_RATE)

            # LSTM predictions on latest window
            breath_win  = breath_f[-WINDOW_SIZE:]
            heart_win   = heart_f[-WINDOW_SIZE:]

            breath_bpm  = predict_bpm(breath_model, breath_win)
            heart_bpm   = predict_bpm(heart_model,  heart_win)

            # Confidence
            breath_conf = compute_confidence(
                breath_f, 0.1, 0.5, SAMPLE_RATE
            )
            heart_conf  = compute_confidence(
                heart_f, 0.67, 2.0, SAMPLE_RATE
            )

            # SNR
            noise_power     = np.var(buffer - breath_f - heart_f)
            signal_power    = np.var(breath_f) + np.var(heart_f)
            snr             = (
                10 * np.log10(signal_power / noise_power)
                if noise_power > 0 else 0.0
            )

            # Presence detection
            rssi_var    = float(np.var(buffer[-SAMPLE_RATE * 2:]))
            presence    = rssi_var > 0.5

            payload = {
                "raw_sample"    : round(float(buffer[-1]), 4),
                "breath_sample" : round(float(breath_f[-1]), 4),
                "heart_sample"  : round(float(heart_f[-1]), 4),
                "breath_bpm"    : breath_bpm,
                "heart_bpm"     : heart_bpm,
                "breath_conf"   : breath_conf,
                "heart_conf"    : heart_conf,
                "snr"           : round(float(snr), 2),
                "presence"      : presence,
                "rssi_var"      : round(rssi_var, 4),
            }

            await websocket.send_json(payload)
            await asyncio.sleep(STREAM_INTERVAL)

    except WebSocketDisconnect:
        print(f"  Client disconnected.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "airpulse_phase4:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",
    )
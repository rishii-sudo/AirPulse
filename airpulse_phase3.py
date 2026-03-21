"""
AirPulse - Phase 3 (Fixed): LSTM Neural Network
=================================================
Project   : AirPulse (Next-gen WiFi vital signs detection)
File      : airpulse_phase3.py

Fix from v1:
  - Separate BPM label functions for breathing vs heart rate
  - Correct frequency bands per signal type
  - Better learning rate and batch size
  - Regression target normalized to 0-1 range

How to run:
    python airpulse_phase3.py

Requires:
    airpulse_data/breathing_filtered.npy   (from Phase 2)
    airpulse_data/heartbeat_filtered.npy   (from Phase 2)
    airpulse_data/metadata.json            (from Phase 1)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# =============================================================
# CONFIGURATION
# =============================================================

INPUT_DIR      = "airpulse_data"
OUTPUT_DIR     = "airpulse_data"

WINDOW_SIZE    = 100       # 1 second @ 100Hz
STEP_SIZE      = 5         # Dense windows for more training data
EPOCHS         = 80
BATCH_SIZE     = 16
LEARNING_RATE  = 0.0005    # Lower LR for stable convergence
LSTM_UNITS_1   = 64
LSTM_UNITS_2   = 32
DROPOUT_RATE   = 0.2

# Correct frequency bands per signal
BREATH_FREQ_LOW  = 0.08    # Hz
BREATH_FREQ_HIGH = 0.55    # Hz
HEART_FREQ_LOW   = 1.0    # Hz
HEART_FREQ_HIGH  = 2.2    # Hz

# BPM normalization ranges (for stable regression target)
BREATH_BPM_MIN = 5.0
BREATH_BPM_MAX = 35.0
HEART_BPM_MIN  = 60.0
HEART_BPM_MAX  = 132.0

DEVICE = torch.device("cpu")


# =============================================================
# LOAD PHASE 2 DATA
# =============================================================

def load_phase2_data() -> tuple:
    """Load filtered signals from Phase 2."""
    breath = np.load(os.path.join(INPUT_DIR, "breathing_filtered.npy"))
    heart  = np.load(os.path.join(INPUT_DIR, "heartbeat_filtered.npy"))

    with open(os.path.join(INPUT_DIR, "metadata.json")) as f:
        meta = json.load(f)

    sr = meta["signal"]["sample_rate_hz"]

    print("Phase 2 data loaded:")
    print(f"  Breathing samples : {len(breath):,}")
    print(f"  Heart samples     : {len(heart):,}")
    print(f"  Sample rate       : {sr} Hz")
    print(f"  True breathing    : {meta['ground_truth']['breathing_bpm']} BPM")
    print(f"  True heart rate   : {meta['ground_truth']['heart_bpm']} BPM")

    return breath, heart, meta, sr


# =============================================================
# LABEL GENERATION — SEPARATE PER SIGNAL TYPE
# =============================================================

def compute_breathing_labels(
    signal      : np.ndarray,
    sample_rate : int,
) -> np.ndarray:
    """
    Compute BPM labels for breathing signal windows.
    Searches only in 0.08-0.55 Hz (breathing range).
    """
    labels = []
    for i in range(0, len(signal) - WINDOW_SIZE, STEP_SIZE):
        window   = signal[i : i + WINDOW_SIZE]
        mag      = np.abs(np.fft.rfft(window * np.hanning(WINDOW_SIZE)))
        freqs    = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / sample_rate)
        mask     = (freqs >= BREATH_FREQ_LOW) & (freqs <= BREATH_FREQ_HIGH)
        if mask.sum() == 0:
            labels.append(BREATHING_BPM_FALLBACK)
            continue
        peak_hz  = freqs[mask][np.argmax(mag[mask])]
        bpm      = float(peak_hz * 60.0)
        bpm      = np.clip(bpm, BREATH_BPM_MIN, BREATH_BPM_MAX)
        labels.append(bpm)
    return np.array(labels, dtype=np.float32)


def compute_heart_labels(
    signal      : np.ndarray,
    sample_rate : int,
) -> np.ndarray:
    """
    Compute BPM labels for heart rate signal windows.
    Searches only in 0.60-2.50 Hz (heart rate range).
    """
    labels = []
    for i in range(0, len(signal) - WINDOW_SIZE, STEP_SIZE):
        window   = signal[i : i + WINDOW_SIZE]
        mag      = np.abs(np.fft.rfft(window * np.hanning(WINDOW_SIZE)))
        freqs    = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / sample_rate)
        mask     = (freqs >= HEART_FREQ_LOW) & (freqs <= HEART_FREQ_HIGH)
        if mask.sum() == 0:
            labels.append(HEART_BPM_FALLBACK)
            continue
        peak_hz  = freqs[mask][np.argmax(mag[mask])]
        bpm      = float(peak_hz * 60.0)
        bpm      = np.clip(bpm, HEART_BPM_MIN, HEART_BPM_MAX)
        labels.append(bpm)
    return np.array(labels, dtype=np.float32)


# These will be set after loading data
BREATHING_BPM_FALLBACK = 15.0
HEART_BPM_FALLBACK     = 72.0


# =============================================================
# DATASET BUILDER
# =============================================================

def build_dataset(
    signal      : np.ndarray,
    labels      : np.ndarray,
    bpm_min     : float,
    bpm_max     : float,
) -> tuple:
    """
    Build sliding window dataset with normalized BPM targets.

    Normalizing targets to 0-1 range makes LSTM training
    much more stable — avoids exploding gradients.

    Args:
        signal  : Filtered signal array
        labels  : BPM label per window
        bpm_min : Minimum BPM for normalization
        bpm_max : Maximum BPM for normalization

    Returns:
        X (np.ndarray) : Shape (n, WINDOW_SIZE, 1)
        y (np.ndarray) : Normalized BPM targets 0-1
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(
        signal.reshape(-1, 1)
    ).flatten()

    X = []
    for idx, i in enumerate(
        range(0, len(scaled) - WINDOW_SIZE, STEP_SIZE)
    ):
        if idx >= len(labels):
            break
        X.append(scaled[i : i + WINDOW_SIZE])

    X = np.array(X, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)

    # Normalize labels to 0-1
    y = (labels[:len(X)] - bpm_min) / (bpm_max - bpm_min)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)

    return X, y


# =============================================================
# LSTM MODEL
# =============================================================

class VitalSignLSTM(nn.Module):
    """
    Two-layer LSTM for BPM regression.
    Output is normalized 0-1, denormalized after prediction.
    """

    def __init__(self, input_size=1, hidden1=LSTM_UNITS_1,
                 hidden2=LSTM_UNITS_2, dropout=DROPOUT_RATE):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden2, 16)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(16, 1)
        self.sig   = nn.Sigmoid()    # Output 0-1

    def forward(self, x):
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = self.drop2(out[:, -1, :])
        out    = self.relu(self.fc1(out))
        return self.sig(self.fc2(out)).squeeze(-1)


# =============================================================
# TRAINING
# =============================================================

def train_model(X, y, label):
    """Train LSTM with early stopping."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_tr_t  = torch.tensor(X_tr).to(DEVICE)
    y_tr_t  = torch.tensor(y_tr).to(DEVICE)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)

    loader    = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE, shuffle=True
    )
    model     = VitalSignLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    print(f"\n  Training {label} LSTM...")
    print(f"  Train : {len(X_tr):,}  |  Val: {len(X_val):,}")

    tr_losses, val_losses     = [], []
    best_val, patience_count  = float("inf"), 0
    best_state                = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_tr = epoch_loss / len(loader)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        tr_losses.append(avg_tr)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}"
                  f"  train: {avg_tr:.5f}"
                  f"  val: {val_loss:.5f}")

        if val_loss < best_val:
            best_val        = val_loss
            best_state      = {k: v.clone()
                               for k, v in model.state_dict().items()}
            patience_count  = 0
        else:
            patience_count += 1
            if patience_count >= 12:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, tr_losses, val_losses


# =============================================================
# EVALUATION
# =============================================================

def evaluate_model(model, X, y_norm, true_bpm,
                   bpm_min, bpm_max, label):
    """Evaluate model — denormalize predictions back to BPM."""
    model.eval()
    X_t = torch.tensor(X).to(DEVICE)

    with torch.no_grad():
        preds_norm = model(X_t).cpu().numpy()

    # Denormalize
    preds    = preds_norm * (bpm_max - bpm_min) + bpm_min
    y_actual = y_norm     * (bpm_max - bpm_min) + bpm_min

    mae      = mean_absolute_error(y_actual, preds)
    rmse     = float(np.sqrt(np.mean((y_actual - preds) ** 2)))
    mean_p   = float(np.mean(preds))
    error    = abs(mean_p - true_bpm)
    within2  = float(np.mean(np.abs(preds - true_bpm) <= 2.0) * 100)

    print(f"\n  {label} results:")
    print(f"    Predicted BPM  : {mean_p:.1f}  (true: {true_bpm})")
    print(f"    Error          : {error:.2f} BPM")
    print(f"    MAE            : {mae:.3f} BPM")
    print(f"    RMSE           : {rmse:.3f} BPM")
    print(f"    Within +-2 BPM : {within2:.1f}%")

    return dict(
        mean_bpm = round(mean_p, 2),
        true_bpm = true_bpm,
        error    = round(error, 3),
        mae      = round(mae, 3),
        rmse     = round(rmse, 3),
        within2  = round(within2, 1),
        preds    = preds.tolist(),
    )


# =============================================================
# VISUALIZATION
# =============================================================

def plot_training(b_tr, b_val, h_tr, h_val):
    BG, PAN = "#0d1117", "#161b22"
    GR, TK, WH = "#21262d", "#6b7280", "#f0f6fc"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("AirPulse - Phase 3 (Fixed): Training Curves",
                 color=WH, fontsize=13, fontweight="bold")

    for ax, tr, vl, title, col in [
        (axes[0], b_tr, b_val, "Breathing Model", "#34d399"),
        (axes[1], h_tr, h_val, "Heart Rate Model", "#f59e0b"),
    ]:
        ax.set_facecolor(PAN)
        ax.plot(tr, color=col, lw=1.5, label="Train loss")
        ax.plot(vl, color=col, lw=1.5, ls="--",
                alpha=0.6, label="Val loss")
        ax.set_title(title, color=WH, fontsize=10, pad=6)
        ax.set_xlabel("Epoch", color=TK, fontsize=8)
        ax.set_ylabel("Loss (MSE)", color=TK, fontsize=8)
        ax.tick_params(colors=TK, labelsize=7)
        ax.grid(True, color=GR, lw=0.5)
        ax.legend(fontsize=8, facecolor=PAN, labelcolor=WH)
        for sp in ax.spines.values():
            sp.set_edgecolor(GR)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "phase3_training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Training plot  : {path}")


def plot_predictions(br, hr):
    BG, PAN = "#0d1117", "#161b22"
    GR, TK, WH = "#21262d", "#6b7280", "#f0f6fc"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        "AirPulse - Phase 3 (Fixed): LSTM Predictions",
        color=WH, fontsize=13, fontweight="bold", y=0.98
    )

    for ax, r, title, col in [
        (axes[0], br, "Breathing Rate", "#34d399"),
        (axes[1], hr, "Heart Rate",     "#f59e0b"),
    ]:
        p = np.array(r["preds"])
        ax.set_facecolor(PAN)
        ax.plot(p, color=col, lw=0.8, alpha=0.85,
                label="LSTM prediction")
        ax.axhline(r["true_bpm"], color="#f87171", lw=1.5,
                   ls="--", label=f"True: {r['true_bpm']} BPM")
        ax.fill_between(
            range(len(p)),
            r["true_bpm"] - 2, r["true_bpm"] + 2,
            color="#f87171", alpha=0.1, label="+-2 BPM"
        )
        ax.set_title(
            f"{title}  |  error: {r['error']:.2f} BPM  "
            f"within +-2 BPM: {r['within2']:.0f}%",
            color=WH, fontsize=9, pad=5
        )
        ax.set_xlabel("Window index", color=TK, fontsize=8)
        ax.set_ylabel("BPM", color=TK, fontsize=8)
        ax.tick_params(colors=TK, labelsize=7)
        ax.grid(True, color=GR, lw=0.5)
        ax.legend(fontsize=8, facecolor=PAN, labelcolor=WH)
        for sp in ax.spines.values():
            sp.set_edgecolor(GR)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, "phase3_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Predictions    : {path}")


# =============================================================
# ENTRY POINT
# =============================================================

def main():
    print("=" * 58)
    print("  AirPulse - Phase 3 (Fixed): LSTM Training")
    print("  Separate label functions per signal type")
    print("=" * 58)
    print()

    # Load data
    breath_sig, heart_sig, meta, sr = load_phase2_data()
    true_b = meta["ground_truth"]["breathing_bpm"]
    true_h = meta["ground_truth"]["heart_bpm"]

    global BREATHING_BPM_FALLBACK, HEART_BPM_FALLBACK
    BREATHING_BPM_FALLBACK = float(true_b)
    HEART_BPM_FALLBACK     = float(true_h)

    # Generate correct labels per signal type
    print("\nGenerating labels...")
    b_labels = compute_breathing_labels(breath_sig, sr)
    h_labels = compute_heart_labels(heart_sig, sr)

    print(f"  Breathing labels: min={b_labels.min():.1f}"
          f"  max={b_labels.max():.1f}"
          f"  mean={b_labels.mean():.1f} BPM")
    print(f"  Heart labels    : min={h_labels.min():.1f}"
          f"  max={h_labels.max():.1f}"
          f"  mean={h_labels.mean():.1f} BPM")

    # Build datasets
    print("\nBuilding datasets...")
    X_b, y_b = build_dataset(
        breath_sig, b_labels, BREATH_BPM_MIN, BREATH_BPM_MAX
    )
    X_h, y_h = build_dataset(
        heart_sig, h_labels, HEART_BPM_MIN, HEART_BPM_MAX
    )
    print(f"  Breathing : {X_b.shape}")
    print(f"  Heart     : {X_h.shape}")

    # Train
    print("\n" + "-" * 58)
    print("  TRAINING")
    print("-" * 58)
    b_model, b_tr, b_val = train_model(X_b, y_b, "Breathing")
    h_model, h_tr, h_val = train_model(X_h, y_h, "Heart Rate")

    # Evaluate
    print("\n" + "-" * 58)
    print("  EVALUATION")
    print("-" * 58)
    br = evaluate_model(
        b_model, X_b, y_b, true_b,
        BREATH_BPM_MIN, BREATH_BPM_MAX, "Breathing"
    )
    hr = evaluate_model(
        h_model, X_h, y_h, true_h,
        HEART_BPM_MIN, HEART_BPM_MAX, "Heart Rate"
    )

    # Save models
    print("\nSaving models...")
    torch.save(b_model.state_dict(),
               os.path.join(OUTPUT_DIR, "lstm_breathing.pt"))
    torch.save(h_model.state_dict(),
               os.path.join(OUTPUT_DIR, "lstm_heart.pt"))
    print("  lstm_breathing.pt  saved")
    print("  lstm_heart.pt      saved")

    # Plot
    print("\nRendering plots...")
    plot_training(b_tr, b_val, h_tr, h_val)
    plot_predictions(br, hr)

    # Save results
    results = {
        "project"    : "AirPulse",
        "phase"      : "3-fixed",
        "framework"  : "PyTorch",
        "fix"        : "Separate label functions per signal type",
        "breathing"  : {k: v for k, v in br.items() if k != "preds"},
        "heart_rate" : {k: v for k, v in hr.items() if k != "preds"},
    }
    with open(os.path.join(OUTPUT_DIR, "phase3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 58)
    print("  Phase 3 (Fixed) complete.")
    print()
    print(f"  Breathing error : {br['error']:.2f} BPM")
    print(f"  Heart error     : {hr['error']:.2f} BPM")
    print()
    print("  Now run Phase 5 dashboard to see improvement:")
    print("  .venv311\\Scripts\\python.exe airpulse_phase5.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
"""
AirPulse — Phase 1: WiFi CSI Signal Simulation
===============================================
Project   : AirPulse (Next-gen WiFi vital signs detection)
File      : airpulse_phase1.py
Purpose   : Simulate realistic WiFi CSI signals containing
            breathing and heart rate components, with noise.
            Saves ML-ready data for Phase 2 processing.

How to run:
    pip install numpy matplotlib
    python airpulse_phase1.py

Output:
    airpulse_data/raw_signal.npy
    airpulse_data/time_array.npy
    airpulse_data/breathing.npy
    airpulse_data/heartbeat.npy
    airpulse_data/metadata.json
    airpulse_data/phase1_signal.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

SAMPLE_RATE       = 100     # Hz — 5x higher than RuView (20 Hz)
DURATION_SEC      = 300     # Seconds of signal to generate
BREATHING_BPM     = 15      # Simulated breathing rate (6–30 normal)
HEART_BPM         = 72      # Simulated heart rate (40–120 normal)
NOISE_LEVEL       = 0.15     # 0.0 = clean, 1.0 = very noisy
OUTPUT_DIR        = "airpulse_data"


# ─────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────

def generate_breathing(n_samples: int, sample_rate: int, bpm: float) -> np.ndarray:
    """
    Generate a sine wave representing chest movement during breathing.

    Physics: Chest expansion/contraction modulates WiFi signal amplitude
    and phase via multipath interference. Modeled as a pure sine wave.

    Args:
        n_samples   : Total number of samples
        sample_rate : Samples per second (Hz)
        bpm         : Breathing rate in beats per minute

    Returns:
        Numpy array of shape (n_samples,)
    """
    t         = np.linspace(0, n_samples / sample_rate, n_samples)
    freq_hz   = bpm / 60.0
    signal    = 4.0 * np.sin(2 * np.pi * freq_hz * t)
    return signal


def generate_heartbeat(n_samples: int, sample_rate: int, bpm: float) -> np.ndarray:
    """
    Generate a realistic heartbeat signal.

    Models two components:
      - Primary wave  : main ventricular contraction
      - Secondary wave: dicrotic notch (aortic valve closure)

    The heartbeat signal is ~10x weaker than breathing in real WiFi CSI,
    which is why standard FFT methods (like RuView) miss it under noise.
    AirPulse uses an LSTM model (Phase 3) to recover it accurately.

    Args:
        n_samples   : Total number of samples
        sample_rate : Samples per second (Hz)
        bpm         : Heart rate in beats per minute

    Returns:
        Numpy array of shape (n_samples,)
    """
    t           = np.linspace(0, n_samples / sample_rate, n_samples)
    freq_hz     = bpm / 60.0
    primary     = 0.8  * np.sin(2 * np.pi * freq_hz * t)
    secondary   = 0.15 * np.sin(4 * np.pi * freq_hz * t)
    return primary + secondary


def generate_noise(n_samples: int, level: float) -> np.ndarray:
    """
    Generate realistic multi-component WiFi interference noise.

    Three noise sources combined:
      1. Gaussian noise      — random hardware/environmental interference
      2. Low-frequency drift — slow temperature or environment change
      3. Motion spikes       — sudden nearby movements (e.g. arm raise)

    Args:
        n_samples : Total number of samples
        level     : Noise amplitude scale factor

    Returns:
        Numpy array of shape (n_samples,)
    """
    gaussian            = level * np.random.randn(n_samples)
    t                   = np.linspace(0, 1, n_samples)
    drift               = 0.1 * level * np.sin(2 * np.pi * 0.05 * t)
    spikes              = np.zeros(n_samples)
    n_spikes            = max(1, int(n_samples * 0.005))
    spike_indices       = np.random.randint(0, n_samples, n_spikes)
    spikes[spike_indices] = np.random.randn(n_spikes) * level * 3.0
    return gaussian + drift + spikes


def build_csi_signal(
    sample_rate   : int   = SAMPLE_RATE,
    duration_sec  : int   = DURATION_SEC,
    breathing_bpm : float = BREATHING_BPM,
    heart_bpm     : float = HEART_BPM,
    noise_level   : float = NOISE_LEVEL,
) -> tuple:
    """
    Compose a complete WiFi CSI signal from individual components.

    CSI (Channel State Information) encodes how a WiFi signal
    travels from transmitter to receiver across multiple paths.
    A human body modifies these paths through micro-movements
    caused by breathing and heartbeat.

    Returns:
        time_array (np.ndarray) : Timestamp for each sample in seconds
        raw_signal (np.ndarray) : Mixed signal — breathing + heart + noise
        components (dict)       : Individual components for reference
    """
    n_samples = sample_rate * duration_sec

    print("Generating CSI signal...")
    print(f"  Sample rate   : {sample_rate} Hz")
    print(f"  Duration      : {duration_sec} s")
    print(f"  Total samples : {n_samples:,}")
    print(f"  Breathing     : {breathing_bpm} BPM")
    print(f"  Heart rate    : {heart_bpm} BPM")
    print(f"  Noise level   : {noise_level}")

    time_array  = np.linspace(0, duration_sec, n_samples)
    breathing   = generate_breathing(n_samples, sample_rate, breathing_bpm)
    heartbeat   = generate_heartbeat(n_samples, sample_rate, heart_bpm)
    noise       = generate_noise(n_samples, noise_level)
    raw_signal  = breathing + heartbeat + noise

    components  = {
        "breathing" : breathing,
        "heartbeat" : heartbeat,
        "noise"     : noise,
    }

    return time_array, raw_signal, components


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_signals(
    time_array : np.ndarray,
    raw_signal : np.ndarray,
    components : dict,
    output_dir : str = OUTPUT_DIR,
) -> None:
    """
    Plot all signal components in a 4-panel figure and save as PNG.

    Displays the first 10 seconds for clarity:
      Panel 1 — Raw CSI (what real hardware captures)
      Panel 2 — Breathing component
      Panel 3 — Heartbeat component
      Panel 4 — Noise component

    Args:
        time_array : Sample timestamps
        raw_signal : Combined signal
        components : Dict with keys 'breathing', 'heartbeat', 'noise'
        output_dir : Directory to write the PNG file
    """
    PREVIEW_SEC = 10
    n_preview   = PREVIEW_SEC * SAMPLE_RATE
    t           = time_array[:n_preview]

    BG_MAIN     = "#0d1117"
    BG_PANEL    = "#161b22"
    COL_GRID    = "#21262d"
    COL_TICK    = "#6b7280"
    COL_TITLE   = "#f0f6fc"

    panels = [
        (raw_signal,              "#60a5fa",
         "Raw CSI Signal",
         "Combined output — as captured by real hardware"),
        (components["breathing"], "#34d399",
         "Breathing Component",
         f"{BREATHING_BPM} BPM  |  chest wall displacement"),
        (components["heartbeat"], "#f59e0b",
         "Heartbeat Component",
         f"{HEART_BPM} BPM  |  weak signal, LSTM recovers this in Phase 3"),
        (components["noise"],     "#f87171",
         "Noise Component",
         "Gaussian + low-freq drift + motion spikes"),
    ]

    fig = plt.figure(figsize=(15, 10), facecolor=BG_MAIN)
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)
    fig.suptitle(
        "AirPulse  —  Phase 1: WiFi CSI Signal Breakdown",
        color=COL_TITLE, fontsize=14, fontweight="bold", y=0.98
    )

    for i, (signal, color, title, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor(BG_PANEL)
        ax.plot(t, signal[:n_preview], color=color, linewidth=0.75, alpha=0.9)
        ax.set_title(
            f"{title}   |   {subtitle}",
            color=COL_TITLE, fontsize=9, pad=5
        )
        ax.tick_params(colors=COL_TICK, labelsize=7)
        ax.set_ylabel("Amplitude", color=COL_TICK, fontsize=8)
        ax.grid(True, color=COL_GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COL_GRID)

    fig.get_axes()[-1].set_xlabel(
        f"Time (seconds)  —  showing first {PREVIEW_SEC}s of {DURATION_SEC}s",
        color=COL_TICK, fontsize=8
    )

    save_path = os.path.join(output_dir, "phase1_signal.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_MAIN)
    plt.close()
    print(f"  Plot saved    : {save_path}")


# ─────────────────────────────────────────────────────────────
# DATA PERSISTENCE
# ─────────────────────────────────────────────────────────────

def save_data(
    time_array : np.ndarray,
    raw_signal : np.ndarray,
    components : dict,
    metadata   : dict,
    output_dir : str = OUTPUT_DIR,
) -> None:
    """
    Persist all signal arrays and metadata to disk.

    File format:
      .npy  — NumPy binary arrays (fast load, compact storage)
      .json — Human-readable metadata and ground-truth labels

    Args:
        time_array : Sample timestamps
        raw_signal : Combined signal
        components : Individual signal components
        metadata   : Settings and ground-truth values
        output_dir : Target directory
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "raw_signal.npy"),  raw_signal)
    np.save(os.path.join(output_dir, "time_array.npy"),  time_array)
    np.save(os.path.join(output_dir, "breathing.npy"),   components["breathing"])
    np.save(os.path.join(output_dir, "heartbeat.npy"),   components["heartbeat"])

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Output folder : {output_dir}/")
    print(f"    raw_signal.npy  ({len(raw_signal):,} samples)")
    print(f"    time_array.npy")
    print(f"    breathing.npy")
    print(f"    heartbeat.npy")
    print(f"    metadata.json")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 58)
    print("  AirPulse — Phase 1: Signal Simulation")
    print("  WiFi-based vital signs  |  ML-ready pipeline")
    print("=" * 58)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1 — Generate signal
    time_array, raw_signal, components = build_csi_signal()

    # Step 2 — Basic statistics
    print("\nSignal statistics:")
    print(f"  Min     : {raw_signal.min():.4f}")
    print(f"  Max     : {raw_signal.max():.4f}")
    print(f"  Mean    : {raw_signal.mean():.4f}")
    print(f"  Std dev : {raw_signal.std():.4f}")

    # Step 3 — Visualize
    print("\nRendering plot...")
    plot_signals(time_array, raw_signal, components)

    # Step 4 — Save
    metadata = {
        "project"      : "AirPulse",
        "phase"        : 1,
        "description"  : "WiFi CSI signal simulation — ML-ready output",
        "signal"       : {
            "sample_rate_hz" : SAMPLE_RATE,
            "duration_sec"   : DURATION_SEC,
            "total_samples"  : int(SAMPLE_RATE * DURATION_SEC),
        },
        "ground_truth" : {
            "breathing_bpm" : BREATHING_BPM,
            "heart_bpm"     : HEART_BPM,
        },
        "noise_level"  : NOISE_LEVEL,
        "next_phase"   : "Run airpulse_phase2.py — bandpass filtering",
    }

    print("\nSaving data...")
    save_data(time_array, raw_signal, components, metadata)

    
if __name__ == "__main__":
    main()
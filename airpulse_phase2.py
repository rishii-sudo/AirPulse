"""
AirPulse — Phase 2: Bandpass Filtering
=======================================
Project   : AirPulse (Next-gen WiFi vital signs detection)
File      : airpulse_phase2.py
Purpose   : Load raw CSI signal from Phase 1, apply bandpass
            filters to isolate breathing and heart rate,
            detect BPM using FFT peak detection, and save
            filtered signals for Phase 3 LSTM training.

How to run:
    pip install numpy matplotlib scipy
    python airpulse_phase2.py

Requires:
    airpulse_data/raw_signal.npy    (from Phase 1)
    airpulse_data/time_array.npy    (from Phase 1)
    airpulse_data/metadata.json     (from Phase 1)

Output:
    airpulse_data/breathing_filtered.npy
    airpulse_data/heartbeat_filtered.npy
    airpulse_data/phase2_results.json
    airpulse_data/phase2_filter.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, welch


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

INPUT_DIR   = "airpulse_data"
OUTPUT_DIR  = "airpulse_data"

# Breathing band: 6–30 BPM = 0.1–0.5 Hz
BREATH_LOW  = 0.1   # Hz
BREATH_HIGH = 0.5   # Hz

# Heart rate band: 40–120 BPM = 0.67–2.0 Hz
HEART_LOW   = 1.1  # Hz
HEART_HIGH  = 2.2  # Hz

# Filter order — higher = sharper cutoff, more accurate
FILTER_ORDER = 4


# ─────────────────────────────────────────────────────────────
# LOAD PHASE 1 DATA
# ─────────────────────────────────────────────────────────────

def load_phase1_data(input_dir: str = INPUT_DIR) -> tuple:
    """
    Load raw signal and metadata saved by Phase 1.

    Args:
        input_dir : Folder containing Phase 1 output files

    Returns:
        raw_signal  (np.ndarray) : Combined CSI signal
        time_array  (np.ndarray) : Timestamps in seconds
        metadata    (dict)       : Phase 1 settings and ground truth
        sample_rate (int)        : Samples per second
    """
    raw_signal  = np.load(os.path.join(input_dir, "raw_signal.npy"))
    time_array  = np.load(os.path.join(input_dir, "time_array.npy"))

    with open(os.path.join(input_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    sample_rate = metadata["signal"]["sample_rate_hz"]

    print("Phase 1 data loaded:")
    print(f"  Samples       : {len(raw_signal):,}")
    print(f"  Sample rate   : {sample_rate} Hz")
    print(f"  Duration      : {metadata['signal']['duration_sec']} s")
    print(f"  True breathing: {metadata['ground_truth']['breathing_bpm']} BPM")
    print(f"  True heart    : {metadata['ground_truth']['heart_bpm']} BPM")

    return raw_signal, time_array, metadata, sample_rate


# ─────────────────────────────────────────────────────────────
# BANDPASS FILTER
# ─────────────────────────────────────────────────────────────

def bandpass_filter(
    signal      : np.ndarray,
    low_hz      : float,
    high_hz     : float,
    sample_rate : int,
    order       : int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Why Butterworth?
      - Maximally flat frequency response in passband
      - No ripple — does not distort the signal shape
      - Standard choice for biomedical signal processing

    Why zero-phase (filtfilt)?
      - Normal filters introduce time delay (phase shift)
      - filtfilt applies filter forward then backward
      - Result: zero time delay, perfect peak alignment
      - Critical for accurate BPM detection

    Args:
        signal      : Input signal array
        low_hz      : Lower cutoff frequency in Hz
        high_hz     : Upper cutoff frequency in Hz
        sample_rate : Sampling rate in Hz
        order       : Filter order (4 = good balance)

    Returns:
        Filtered signal array of same shape as input
    """
    nyquist     = sample_rate / 2.0
    low_norm    = low_hz  / nyquist
    high_norm   = high_hz / nyquist

    b, a        = butter(order, [low_norm, high_norm], btype="band")
    filtered    = filtfilt(b, a, signal)

    return filtered


# ─────────────────────────────────────────────────────────────
# BPM DETECTION VIA FFT
# ─────────────────────────────────────────────────────────────

def detect_bpm(
    filtered_signal : np.ndarray,
    low_hz          : float,
    high_hz         : float,
    sample_rate     : int,
) -> tuple:
    """
    Detect the dominant frequency in a filtered signal using
    Welch's Power Spectral Density method.

    Why Welch instead of simple FFT?
      - Splits signal into overlapping windows
      - Averages the spectra of all windows
      - Reduces random noise in the spectrum
      - Much more reliable peak detection than single FFT
      - This is one key advantage over RuView's approach

    Args:
        filtered_signal : Bandpass filtered signal
        low_hz          : Lower bound of search range
        high_hz         : Upper bound of search range
        sample_rate     : Sampling rate in Hz

    Returns:
        detected_bpm  (float)      : Detected rate in BPM
        peak_freq_hz  (float)      : Dominant frequency in Hz
        freqs         (np.ndarray) : Full frequency axis
        psd           (np.ndarray) : Power spectral density
        confidence    (float)      : Detection confidence 0–100%
    """
    # Window size: 10 seconds of data for good frequency resolution
    nperseg     = min(len(filtered_signal), sample_rate * 10)
    freqs, psd  = welch(filtered_signal, fs=sample_rate, nperseg=nperseg)

    # Restrict search to the band of interest
    band_mask   = (freqs >= low_hz) & (freqs <= high_hz)
    band_freqs  = freqs[band_mask]
    band_psd    = psd[band_mask]

    if len(band_psd) == 0 or band_psd.max() == 0:
        return 0.0, 0.0, freqs, psd, 0.0

    peak_idx      = np.argmax(band_psd)
    peak_freq_hz  = band_freqs[peak_idx]
    detected_bpm  = peak_freq_hz * 60.0

    # Confidence: how dominant is the peak vs rest of band
    confidence    = float(band_psd[peak_idx] / band_psd.sum()) * 100
    confidence    = min(confidence * 1.5, 99.0)

    return detected_bpm, peak_freq_hz, freqs, psd, confidence


# ─────────────────────────────────────────────────────────────
# SIGNAL QUALITY METRICS
# ─────────────────────────────────────────────────────────────

def compute_snr(
    filtered_signal : np.ndarray,
    raw_signal      : np.ndarray,
) -> float:
    """
    Compute Signal-to-Noise Ratio after filtering.

    SNR = 10 * log10(signal_power / noise_power)

    Higher SNR = cleaner signal = better detection accuracy.

    Args:
        filtered_signal : Signal after bandpass filter
        raw_signal      : Original noisy signal

    Returns:
        SNR in decibels (dB)
    """
    signal_power    = np.var(filtered_signal)
    noise_power     = np.var(raw_signal - filtered_signal)

    if noise_power == 0:
        return 99.0

    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_results(
    time_array         : np.ndarray,
    raw_signal         : np.ndarray,
    breath_filtered    : np.ndarray,
    heart_filtered     : np.ndarray,
    breath_freqs       : np.ndarray,
    breath_psd         : np.ndarray,
    heart_freqs        : np.ndarray,
    heart_psd          : np.ndarray,
    breath_bpm         : float,
    heart_bpm          : float,
    breath_conf        : float,
    heart_conf         : float,
    true_breath_bpm    : float,
    true_heart_bpm     : float,
    sample_rate        : int,
    output_dir         : str = OUTPUT_DIR,
) -> None:
    """
    Generate a 6-panel dashboard showing filtering results.

    Layout:
      Row 1: Raw signal  |  Breathing filtered  |  Heart filtered
      Row 2: (empty)     |  Breathing FFT       |  Heart FFT

    Args:
        All signal arrays, detection results, and ground truth values.
        output_dir : Directory to save the PNG
    """
    BG_MAIN     = "#0d1117"
    BG_PANEL    = "#161b22"
    COL_GRID    = "#21262d"
    COL_TICK    = "#6b7280"
    COL_WHITE   = "#f0f6fc"
    COL_MUTED   = "#8b949e"

    PREVIEW_SEC = 15
    n_preview   = PREVIEW_SEC * sample_rate
    t           = time_array[:n_preview]

    fig = plt.figure(figsize=(16, 11), facecolor=BG_MAIN)
    gs  = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.55, wspace=0.35
    )
    fig.suptitle(
        "AirPulse  —  Phase 2: Bandpass Filter Results",
        color=COL_WHITE, fontsize=14, fontweight="bold", y=0.98
    )

    def style_ax(ax, title, subtitle=""):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=COL_TICK, labelsize=7)
        ax.grid(True, color=COL_GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(COL_GRID)
        label = f"{title}" + (f"   |   {subtitle}" if subtitle else "")
        ax.set_title(label, color=COL_WHITE, fontsize=9, pad=5)
        ax.set_ylabel("Amplitude", color=COL_TICK, fontsize=8)

    # ── Panel 1: Raw CSI signal ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, "Raw CSI Signal", "input to filter — breathing + heart + noise mixed")
    ax1.plot(t, raw_signal[:n_preview], color="#60a5fa",
             linewidth=0.6, alpha=0.8)
    ax1.set_xlabel("Time (s)", color=COL_TICK, fontsize=8)

    # ── Panel 2: Breathing filtered signal ──────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    err_b = abs(breath_bpm - true_breath_bpm)
    style_ax(ax2,
             "Breathing — Filtered Signal",
             f"0.1–0.5 Hz bandpass  |  error: {err_b:.1f} BPM")
    ax2.plot(t, breath_filtered[:n_preview], color="#34d399",
             linewidth=0.9)
    ax2.set_xlabel("Time (s)", color=COL_TICK, fontsize=8)

    # ── Panel 3: Heart rate filtered signal ─────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    err_h = abs(heart_bpm - true_heart_bpm)
    style_ax(ax3,
             "Heart Rate — Filtered Signal",
             f"0.67–2.0 Hz bandpass  |  error: {err_h:.1f} BPM")
    ax3.plot(t, heart_filtered[:n_preview], color="#f59e0b",
             linewidth=0.9)
    ax3.set_xlabel("Time (s)", color=COL_TICK, fontsize=8)

    # ── Panel 4: Breathing FFT spectrum ─────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(BG_PANEL)
    ax4.tick_params(colors=COL_TICK, labelsize=7)
    ax4.grid(True, color=COL_GRID, linewidth=0.5)
    for spine in ax4.spines.values():
        spine.set_edgecolor(COL_GRID)

    mask_b      = (breath_freqs >= BREATH_LOW) & (breath_freqs <= BREATH_HIGH)
    ax4.fill_between(breath_freqs[mask_b], breath_psd[mask_b],
                     color="#34d399", alpha=0.3)
    ax4.plot(breath_freqs[mask_b], breath_psd[mask_b],
             color="#34d399", linewidth=1.2)
    ax4.axvline(breath_bpm / 60, color="#f87171", linewidth=1.5,
                linestyle="--", label=f"Detected: {breath_bpm:.1f} BPM")
    ax4.axvline(true_breath_bpm / 60, color="#ffffff", linewidth=1,
                linestyle=":", alpha=0.5,
                label=f"True: {true_breath_bpm:.0f} BPM")
    ax4.set_title(
        f"Breathing FFT   |   confidence: {breath_conf:.0f}%",
        color=COL_WHITE, fontsize=9, pad=5
    )
    ax4.set_xlabel("Frequency (Hz)", color=COL_TICK, fontsize=8)
    ax4.set_ylabel("Power", color=COL_TICK, fontsize=8)
    ax4.legend(fontsize=7, facecolor=BG_PANEL,
               labelcolor=COL_WHITE, framealpha=0.8)

    # ── Panel 5: Heart rate FFT spectrum ────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(BG_PANEL)
    ax5.tick_params(colors=COL_TICK, labelsize=7)
    ax5.grid(True, color=COL_GRID, linewidth=0.5)
    for spine in ax5.spines.values():
        spine.set_edgecolor(COL_GRID)

    mask_h      = (heart_freqs >= HEART_LOW) & (heart_freqs <= HEART_HIGH)
    ax5.fill_between(heart_freqs[mask_h], heart_psd[mask_h],
                     color="#f59e0b", alpha=0.3)
    ax5.plot(heart_freqs[mask_h], heart_psd[mask_h],
             color="#f59e0b", linewidth=1.2)
    ax5.axvline(heart_bpm / 60, color="#f87171", linewidth=1.5,
                linestyle="--", label=f"Detected: {heart_bpm:.1f} BPM")
    ax5.axvline(true_heart_bpm / 60, color="#ffffff", linewidth=1,
                linestyle=":", alpha=0.5,
                label=f"True: {true_heart_bpm:.0f} BPM")
    ax5.set_title(
        f"Heart Rate FFT   |   confidence: {heart_conf:.0f}%",
        color=COL_WHITE, fontsize=9, pad=5
    )
    ax5.set_xlabel("Frequency (Hz)", color=COL_TICK, fontsize=8)
    ax5.set_ylabel("Power", color=COL_TICK, fontsize=8)
    ax5.legend(fontsize=7, facecolor=BG_PANEL,
               labelcolor=COL_WHITE, framealpha=0.8)

    save_path = os.path.join(output_dir, "phase2_filter.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG_MAIN)
    plt.close()
    print(f"  Plot saved    : {save_path}")


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_results(
    breath_filtered : np.ndarray,
    heart_filtered  : np.ndarray,
    results         : dict,
    output_dir      : str = OUTPUT_DIR,
) -> None:
    """
    Save filtered signals and detection results to disk.

    These files are the direct input to Phase 3 LSTM training.

    Args:
        breath_filtered : Breathing bandpass filtered signal
        heart_filtered  : Heart rate bandpass filtered signal
        results         : Detection results and metrics dict
        output_dir      : Target directory
    """
    np.save(
        os.path.join(output_dir, "breathing_filtered.npy"),
        breath_filtered
    )
    np.save(
        os.path.join(output_dir, "heartbeat_filtered.npy"),
        heart_filtered
    )
    with open(os.path.join(output_dir, "phase2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Output folder : {output_dir}/")
    print(f"    breathing_filtered.npy  ({len(breath_filtered):,} samples)")
    print(f"    heartbeat_filtered.npy  ({len(heart_filtered):,} samples)")
    print(f"    phase2_results.json")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 58)
    print("  AirPulse — Phase 2: Bandpass Filtering")
    print("  Isolate breathing + heart rate from raw CSI")
    print("=" * 58)
    print()

    # Step 1 — Load Phase 1 data
    raw_signal, time_array, metadata, sample_rate = load_phase1_data()
    true_breath_bpm = metadata["ground_truth"]["breathing_bpm"]
    true_heart_bpm  = metadata["ground_truth"]["heart_bpm"]

    # Step 2 — Apply bandpass filters
    print("\nApplying bandpass filters...")
    breath_filtered = bandpass_filter(
        raw_signal, BREATH_LOW, BREATH_HIGH, sample_rate
    )
    heart_filtered  = bandpass_filter(
        raw_signal, HEART_LOW, HEART_HIGH, sample_rate
    )
    print(f"  Breathing band : {BREATH_LOW}–{BREATH_HIGH} Hz")
    print(f"  Heart rate band: {HEART_LOW}–{HEART_HIGH} Hz")
    print(f"  Filter order   : {FILTER_ORDER} (Butterworth, zero-phase)")

    # Step 3 — Detect BPM from filtered signals
    print("\nDetecting BPM...")
    breath_bpm, breath_hz, b_freqs, b_psd, breath_conf = detect_bpm(
        breath_filtered, BREATH_LOW, BREATH_HIGH, sample_rate
    )
    heart_bpm, heart_hz, h_freqs, h_psd, heart_conf = detect_bpm(
        heart_filtered, HEART_LOW, HEART_HIGH, sample_rate
    )

    # Step 4 — Compute SNR
    breath_snr  = compute_snr(breath_filtered, raw_signal)
    heart_snr   = compute_snr(heart_filtered,  raw_signal)

    # Step 5 — Print results
    print()
    print("─" * 58)
    print("  RESULTS")
    print("─" * 58)
    print(f"  Breathing detected : {breath_bpm:.1f} BPM"
          f"  (true: {true_breath_bpm} BPM)"
          f"  error: {abs(breath_bpm - true_breath_bpm):.1f} BPM")
    print(f"  Breathing conf     : {breath_conf:.0f}%")
    print(f"  Breathing SNR      : {breath_snr:.1f} dB")
    print()
    print(f"  Heart rate detected: {heart_bpm:.1f} BPM"
          f"  (true: {true_heart_bpm} BPM)"
          f"  error: {abs(heart_bpm - true_heart_bpm):.1f} BPM")
    print(f"  Heart rate conf    : {heart_conf:.0f}%")
    print(f"  Heart rate SNR     : {heart_snr:.1f} dB")
    print("─" * 58)

    # Step 6 — Plot
    print("\nRendering plot...")
    plot_results(
        time_array, raw_signal,
        breath_filtered, heart_filtered,
        b_freqs, b_psd, h_freqs, h_psd,
        breath_bpm, heart_bpm,
        breath_conf, heart_conf,
        true_breath_bpm, true_heart_bpm,
        sample_rate,
    )

    # Step 7 — Save
    results = {
        "project"   : "AirPulse",
        "phase"     : 2,
        "breathing" : {
            "detected_bpm"  : round(breath_bpm, 2),
            "true_bpm"      : true_breath_bpm,
            "error_bpm"     : round(abs(breath_bpm - true_breath_bpm), 2),
            "confidence_pct": round(breath_conf, 1),
            "snr_db"        : round(breath_snr, 2),
            "band_hz"       : [BREATH_LOW, BREATH_HIGH],
        },
        "heart_rate" : {
            "detected_bpm"  : round(heart_bpm, 2),
            "true_bpm"      : true_heart_bpm,
            "error_bpm"     : round(abs(heart_bpm - true_heart_bpm), 2),
            "confidence_pct": round(heart_conf, 1),
            "snr_db"        : round(heart_snr, 2),
            "band_hz"       : [HEART_LOW, HEART_HIGH],
        },
        "filter" : {
            "type"          : "Butterworth bandpass (zero-phase)",
            "order"         : FILTER_ORDER,
            "method"        : "Welch PSD peak detection",
        },
        "next_phase" : "Run airpulse_phase3.py — LSTM model training",
    }

    print("\nSaving results...")
    save_results(breath_filtered, heart_filtered, results)



if __name__ == "__main__":
    main()
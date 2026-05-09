"""
src/utils/signal_processing.py
───────────────────────────────
Physiological signal processing utilities for rPPG / HR estimation.

Key operations
──────────────
* Butterworth bandpass filter (keeps 45–180 BPM band)
* Welch / standard FFT power spectral density
* BPM estimation with parabolic peak interpolation for sub-bin accuracy
* Signal quality metrics: SNR, correlation, lag
* Normalisation helpers used across dataset and model

References
──────────
[1] McDuff, D. et al. (2015) "Improvements in Remote Cardiopulmonary Measurement
    Using a Five Band Frequency Selection Method." IEEE T-AFFC.
[2] de Haan, G. & Jeanne, V. (2013) "Robust Pulse Rate from Chrominance-Based
    rPPG." IEEE T-BME.
[3] Yu, Z. et al. (2022) "PhysFormer: Facial Video-based Physiological
    Measurement with Temporal Difference Transformer." CVPR 2022.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HR_LOW_HZ = 0.75      # 45 BPM
HR_HIGH_HZ = 3.0      # 180 BPM


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def bandpass_filter(
    sig: np.ndarray,
    fps: float,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    sig   : 1-D float array (T,)
    fps   : sampling rate in Hz
    low_hz / high_hz : passband limits
    order : filter order

    Returns
    -------
    Filtered signal of the same shape.
    """
    nyq = fps / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    # Clamp to avoid ill-conditioned filters
    low = max(low, 1e-4)
    high = min(high, 0.9999)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, sig)


def detrend_signal(sig: np.ndarray, lambda_val: float = 10.0) -> np.ndarray:
    """
    Smoothness-priors detrending (Tarvainen et al. 2002).
    Removes slow baseline drift while preserving BVP oscillation.
    """
    T = len(sig)
    I = np.eye(T)
    D2 = np.diff(I, n=2, axis=0)
    return (I - np.linalg.solve((I + lambda_val**2 * D2.T @ D2), I)) @ sig


def normalise_signal(sig: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalise a 1-D signal.  method ∈ {zscore, minmax, none}."""
    if method == "zscore":
        std = sig.std()
        if std < 1e-8:
            return sig - sig.mean()
        return (sig - sig.mean()) / std
    elif method == "minmax":
        lo, hi = sig.min(), sig.max()
        if hi - lo < 1e-8:
            return np.zeros_like(sig)
        return 2.0 * (sig - lo) / (hi - lo) - 1.0
    return sig


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def welch_psd(
    sig: np.ndarray,
    fps: float,
    nperseg_sec: float = 5.0,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch power spectral density restricted to the valid HR band.

    Returns
    -------
    freqs : frequency array (Hz)
    power : PSD array
    """
    nperseg = int(nperseg_sec * fps)
    freqs, power = signal.welch(
        sig, fs=fps, nperseg=min(nperseg, len(sig)), window="hann"
    )
    # Restrict to HR band
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return freqs[mask], power[mask]


def fft_psd(
    sig: np.ndarray,
    fps: float,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard FFT-based PSD in the HR band."""
    T = len(sig)
    window = np.hanning(T)
    fft_vals = np.fft.rfft(sig * window, n=T)
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    power = np.abs(fft_vals) ** 2
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return freqs[mask], power[mask]


def estimate_bpm(
    sig: np.ndarray,
    fps: float,
    method: str = "welch",
    peak_pick: str = "parabolic",
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
) -> float:
    """
    Estimate Heart Rate in BPM from a 1-D rPPG signal.

    Parameters
    ----------
    sig       : 1-D rPPG waveform (T,)
    fps       : camera frame rate (Hz)
    method    : "welch" or "fft"
    peak_pick : "argmax" or "parabolic" (sub-bin accuracy via parabolic interp)

    Returns
    -------
    BPM as a float.
    """
    if method == "welch":
        freqs, power = welch_psd(sig, fps, low_hz=low_hz, high_hz=high_hz)
    else:
        freqs, power = fft_psd(sig, fps, low_hz=low_hz, high_hz=high_hz)

    if len(power) == 0:
        return float("nan")

    peak_idx = int(np.argmax(power))

    if peak_pick == "parabolic" and 1 <= peak_idx < len(power) - 1:
        # Parabolic interpolation for sub-bin peak frequency
        alpha = power[peak_idx - 1]
        beta = power[peak_idx]
        gamma = power[peak_idx + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-10:
            offset = 0.5 * (alpha - gamma) / denom
        else:
            offset = 0.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
        peak_freq = freqs[peak_idx] + offset * df
    else:
        peak_freq = freqs[peak_idx]

    return float(peak_freq * 60.0)


# ---------------------------------------------------------------------------
# Signal quality metrics
# ---------------------------------------------------------------------------


def compute_snr(
    sig: np.ndarray,
    fps: float,
    hr_bpm: float,
    harmonic_range_hz: float = 0.1,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
) -> float:
    """
    Compute Signal-to-Noise Ratio (dB) of an rPPG signal at a known HR.

    Signal power = power in a narrow band around HR frequency + 1st harmonic.
    Noise power = remaining HR-band power.

    Parameters
    ----------
    hr_bpm : reference heart rate in BPM (used to define signal band)
    harmonic_range_hz : half-width of the HR / harmonic peaks (Hz)
    """
    freqs, power = fft_psd(sig, fps, low_hz=low_hz, high_hz=high_hz)
    if len(power) == 0:
        return float("nan")

    hr_hz = hr_bpm / 60.0
    harmonics = [hr_hz, 2.0 * hr_hz]
    signal_mask = np.zeros(len(freqs), dtype=bool)
    for hf in harmonics:
        if low_hz <= hf <= high_hz:
            signal_mask |= np.abs(freqs - hf) <= harmonic_range_hz

    sig_power = power[signal_mask].sum()
    noise_power = power[~signal_mask].sum()
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(sig_power / noise_power))


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Sliding-window BPM trace (for inference on long sequences)
# ---------------------------------------------------------------------------


def sliding_window_bpm(
    rppg: np.ndarray,
    fps: float,
    window_sec: float = 10.0,
    step_sec: float = 2.5,
    method: str = "welch",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply BPM estimation over a sliding window on a long rPPG sequence.

    Returns
    -------
    times_sec : centre time of each window (seconds)
    bpms      : estimated BPM per window
    """
    win = int(window_sec * fps)
    step = int(step_sec * fps)
    T = len(rppg)
    times, bpms = [], []
    start = 0
    while start + win <= T:
        seg = rppg[start : start + win]
        seg = bandpass_filter(seg, fps)
        seg = normalise_signal(seg)
        bpm = estimate_bpm(seg, fps, method=method)
        times.append((start + win / 2) / fps)
        bpms.append(bpm)
        start += step
    return np.array(times), np.array(bpms)


# ---------------------------------------------------------------------------
# Frequency-domain probability (used in loss computation)
# ---------------------------------------------------------------------------


def signal_to_psd_target(
    bvp: np.ndarray,
    fps: float,
    n_bins: int = 200,
    low_hz: float = HR_LOW_HZ,
    high_hz: float = HR_HIGH_HZ,
    bandwidth_hz: float = 0.25,
) -> np.ndarray:
    """
    Convert a BVP ground-truth signal to a soft-label frequency distribution.

    Used by the frequency-domain cross-entropy loss.  The target is a
    Gaussian-smoothed one-hot centred at the true HR frequency.

    Returns
    -------
    soft_label : (n_bins,) float32 probability vector
    """
    freqs = np.linspace(low_hz, high_hz, n_bins)
    hr_hz = estimate_bpm(bvp, fps, method="fft") / 60.0
    soft = np.exp(-0.5 * ((freqs - hr_hz) / bandwidth_hz) ** 2)
    soft = soft / (soft.sum() + 1e-9)
    return soft.astype(np.float32)

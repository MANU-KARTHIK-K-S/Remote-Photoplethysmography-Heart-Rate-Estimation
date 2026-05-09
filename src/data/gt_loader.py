"""
src/data/gt_loader.py
──────────────────────
Ground-truth Heart-Rate derivation for MR-NIRP-D.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW WE DERIVE BPM FROM THE REFERENCE SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The MR-NIRP-D dataset provides two files per session that together
constitute the ground truth:

┌─────────────────────────────────────────────────────────┐
│  File 1:  PulseOX/pulseOx.mat                           │
│  A contact pulse-oximeter (finger-clip) captures a      │
│  photoplethysmography (PPG/BVP) waveform at 125 Hz.     │
│  Fields: time (Unix epoch), data (BVP amplitude),        │
│          rate (Hz), spo2, hr_bpm                         │
│                                                          │
│  File 2:  <session>/cam0_full_log.txt  (or partial)      │
│  A Python list-literal of Unix timestamps — one per NIR  │
│  camera frame, at ~30 fps.                               │
│    e.g.  [1540577618.696552, 1540577618.715550, ...]     │
└─────────────────────────────────────────────────────────┘

STEP-BY-STEP DERIVATION
────────────────────────
Step 1  Load pulseOx.mat
        → bvp_raw   : float64 array  shape (N_pulseox,)   at ~125 Hz
        → t_pulseox : float64 Unix timestamps             shape (N_pulseox,)
        → fs_pulseox: scalar Hz (usually 125.0)

Step 2  Parse cam0_full_log.txt
        → t_cam     : float64 Unix timestamps             shape (N_cam_frames,)
        The file is a Python list literal — we use ast.literal_eval to parse it.
        We choose cam0_full_log.txt (all captured frames) over cam0_partial
        because it has fewer dropped frames and aligns with the PGM sequence.

Step 3  Find temporal overlap
        The pulseOx and camera clocks are both Unix epoch but may have up to
        a few seconds of misalignment at the recording boundaries.  We find
        the intersection of [t_cam[0], t_cam[-1]] with the pulseOx time range
        and clip both arrays to the common interval.

Step 4  Bandpass filter the pulseOx BVP (at 125 Hz)
        Butterworth 4th-order bandpass [0.75 Hz, 3.0 Hz]  (= 45–180 BPM).
        We filter at the ORIGINAL high sample rate so the filter has
        maximum frequency resolution before downsampling.

Step 5  Interpolate to camera frame rate
        scipy.interpolate.interp1d  (kind='linear')
        Maps the filtered 125-Hz BVP onto the t_cam timestamps.
        Result: bvp_cam  shape (N_cam_frames,)  float32

Step 6  Z-score normalise
        bvp_cam = (bvp_cam − mean) / std
        This removes DC offset and makes the loss scale-invariant.

Step 7  BPM estimation (used for evaluation, NOT for training labels)
        We apply Welch PSD on 10-second windows of bvp_cam:
          freqs, power = scipy.signal.welch(window, fs=30, nperseg=5*30)
          hr_hz = freqs[argmax(power)]  within [0.75, 3.0] Hz
          hr_bpm = hr_hz × 60
        Parabolic interpolation gives sub-bin accuracy (±0.2 BPM).

Note: The training label is the BVP waveform (Step 6 output),
      NOT the scalar BPM — the model learns to predict the full
      rPPG signal; BPM is extracted from the predicted signal
      at inference time (same FFT pipeline, Step 7).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

References
──────────
[1] Hu S. et al. "MR-NIRP: A Multi-Resolution NIR rPPG Dataset."
    Rice Computational Imaging Lab, 2019.
    https://computationalimaging.rice.edu/mr-nirp-dataset/

[2] de Haan G. & Jeanne V. (2013) "Robust Pulse Rate from
    Chrominance-Based rPPG." IEEE T-BME 60(10).
    — BVP bandpass filtering methodology.

[3] Poh M-Z et al. (2010) "Non-contact Cardiac Pulse Measurements."
    Optics Express 18(10).
    — FFT-based HR estimation from rPPG.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import scipy.interpolate
import scipy.signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PulseOxRecord:
    """Raw pulseOx.mat contents after loading."""
    bvp_raw: np.ndarray       # (N,) float64  — raw BVP waveform
    timestamps: np.ndarray    # (N,) float64  — Unix epoch seconds
    fs: float                 # sample rate in Hz (typically 125.0)
    spo2: Optional[float]     # SpO2 percentage (if present)
    hr_ref_bpm: Optional[float]  # Reference BPM recorded by the device (if present)


@dataclass
class SyncedGT:
    """Ground truth signal synchronised to camera frame rate."""
    bvp_cam: np.ndarray       # (N_frames,) float32  — BVP at camera fps
    t_cam: np.ndarray         # (N_frames,) float64  — Unix timestamps
    n_valid_frames: int       # frames within the pulseOx time range
    fs_cam: float             # camera effective frame rate (Hz)
    hr_ref_bpm: Optional[float]
    spo2: Optional[float]


# ---------------------------------------------------------------------------
# Step 1: Load pulseOx.mat
# ---------------------------------------------------------------------------


# All field names we've seen across MR-NIRP-D versions
_BVP_KEYS = ["data", "bvp", "BVP", "ppg", "PPG", "signal", "waveform", "pulse","pulseOxRecord"]
_TIME_KEYS = ["time", "Time", "t", "timestamps", "timestamp","pulseOxTime"]
_RATE_KEYS = ["rate", "fs", "Fs", "sample_rate", "sampleRate", "hz", "Hz"]
_SPO2_KEYS = ["spo2", "SpO2", "SPO2", "oxygen"]
_HR_KEYS   = ["hr_bpm", "hr", "HR", "heartrate", "heart_rate", "bpm", "BPM"]


def load_pulseox_mat(mat_path: str | Path) -> Optional[PulseOxRecord]:
    """
    Load pulseOx.mat and return a PulseOxRecord.

    Handles:
    ─────────
    * Field names: data / bvp / ppg (see _BVP_KEYS)
    * Row-vector vs column-vector vs 2-D array storage
    * Rate field stored as scalar, (1,1) matrix, or missing (inferred)
    * Timestamps stored in 'time' field or absent (reconstructed from rate)
    * MATLAB struct-of-arrays (rare — flattened automatically)

    Returns None if the file cannot be loaded or parsed.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        logger.error("pulseOx.mat not found: %s", mat_path)
        return None

    try:
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except Exception as e:
        logger.error("Failed to load %s: %s", mat_path, e)
        return None

    # ── BVP waveform ──────────────────────────────────────────────────────
    bvp_raw = _extract_array(mat, _BVP_KEYS, mat_path)
    if bvp_raw is None:
        logger.error("Cannot find BVP field in %s  (tried %s)", mat_path, _BVP_KEYS)
        return None
    bvp_raw = bvp_raw.ravel().astype(np.float64)

    # ── Timestamps ────────────────────────────────────────────────────────
    timestamps = _extract_array(mat, _TIME_KEYS, mat_path)
    if timestamps is not None:
        timestamps = timestamps.ravel().astype(np.float64)
        if len(timestamps) != len(bvp_raw):
            logger.warning(
                "Timestamp length (%d) ≠ BVP length (%d) in %s — reconstructing",
                len(timestamps), len(bvp_raw), mat_path.name,
            )
            timestamps = None

    if timestamps is None:
        # Reconstruct timestamps: we don't know t0, so use relative (0-based)
        # The synchronisation step handles absolute alignment via cam timestamps.
        logger.debug("Reconstructing relative timestamps at %.1f Hz", fs)
        timestamps = np.arange(len(bvp_raw), dtype=np.float64) / fs
        
    # ── Sample rate ────────────────────────────────────────────────────────
    fs = _extract_scalar(mat, _RATE_KEYS)
    if fs is None or fs < 1.0:
        if timestamps is not None and len(timestamps) > 1:
            fs = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
            logger.info("Inferred rate from timestamps: %.2f Hz", fs)
        else:
            logger.warning("Rate field missing in %s — assuming 125 Hz", mat_path.name)
            fs = 125.0

    

    # ── SpO2 & reference HR ───────────────────────────────────────────────
    spo2 = _extract_scalar(mat, _SPO2_KEYS)
    hr_ref = _extract_scalar(mat, _HR_KEYS)

    logger.info(
        "PulseOx loaded: %s  |  N=%d  fs=%.0f Hz  spo2=%s  hr_ref=%s BPM",
        mat_path.name, len(bvp_raw), fs,
        f"{spo2:.1f}" if spo2 else "—",
        f"{hr_ref:.1f}" if hr_ref else "—",
    )
    return PulseOxRecord(
        bvp_raw=bvp_raw,
        timestamps=timestamps,
        fs=fs,
        spo2=spo2,
        hr_ref_bpm=hr_ref,
    )


# ---------------------------------------------------------------------------
# Step 2: Parse camera timestamp log
# ---------------------------------------------------------------------------


def load_camera_timestamps(log_path: str | Path) -> Optional[np.ndarray]:
    """
    Parse cam0_full_log.txt (or cam0_partial_log.txt).

    Format: a Python list-literal of Unix timestamps written as one line or
    spread across multiple lines, e.g.:
        [1540577618.696552, 1540577618.71555, 1540577618.736305, ...]

    We use ast.literal_eval which is safe (no exec) and handles multi-line.

    Returns
    -------
    np.ndarray of shape (N_frames,) float64, or None on parse failure.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        logger.error("Camera log not found: %s", log_path)
        return None

    raw = log_path.read_text().strip()
    # Handle both single-line and multi-line list literals
    raw = raw.replace("\n", " ").replace("\r", " ")

    try:
        ts_list = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        # Fallback: try to parse each token as a float
        logger.warning("ast.literal_eval failed (%s) — trying token-by-token", e)
        try:
            import re
            ts_list = [float(x) for x in re.findall(r"[\d.]+", raw)]
        except Exception as e2:
            logger.error("Cannot parse camera timestamps from %s: %s", log_path, e2)
            return None

    if not ts_list:
        logger.error("Empty timestamp list in %s", log_path)
        return None

    ts = np.array(ts_list, dtype=np.float64)
    logger.debug(
        "Camera timestamps: %d frames  t0=%.3f  fps≈%.1f",
        len(ts), ts[0], 1.0 / (ts[1] - ts[0]) if len(ts) > 1 else 0,
    )
    return ts


def find_camera_log(session_dir: Path, prefer_full: bool = True) -> Optional[Path]:
    """
    Locate cam0_full_log.txt or cam0_partial_log.txt in a session directory.

    We prefer 'full' because it contains ALL captured frames and aligns
    directly with the PGM frame sequence (no dropped frames).
    """
    candidates = (
        ["cam0_full_log.txt", "cam0_partial_log.txt"]
        if prefer_full
        else ["cam0_partial_log.txt", "cam0_full_log.txt"]
    )
    for name in candidates:
        p = session_dir / name
        if p.exists():
            logger.debug("Using camera log: %s", p.name)
            return p
    # Search recursively one level up (PulseOX/ sub-folder case)
    for name in candidates:
        p = session_dir.parent / name
        if p.exists():
            return p
    logger.warning("No camera timestamp log found in %s", session_dir)
    return None


def find_pulseox_mat(session_dir: Path) -> Optional[Path]:
    """
    Locate pulseOx.mat for a given session.

    MR-NIRP-D stores it in a PulseOX/ sub-folder at the subject level
    (shared across sessions of the same subject), OR directly in the
    session directory.

    Search order:
      1. <session>/PulseOX/pulseOx.mat
      2. <subject>/PulseOX/pulseOx.mat   ← most common MR-NIRP-D layout
      3. <session>/pulseOx.mat
      4. <subject>/pulseOx.mat
    """
    candidates = [
        session_dir / "PulseOX" / "pulseOx.mat",
        session_dir.parent / "PulseOX" / "pulseOx.mat",   # subject-level
        session_dir / "pulseOx.mat",
        session_dir.parent / "pulseOx.mat",
        session_dir / "pulseox.mat",                        # case variants
        session_dir.parent / "pulseox.mat",
    ]
    for p in candidates:
        if p.exists():
            logger.debug("Found pulseOx.mat: %s", p)
            return p
    logger.warning("pulseOx.mat not found for session %s", session_dir)
    return None


# ---------------------------------------------------------------------------
# Step 3–6: Synchronise, filter, interpolate, normalise
# ---------------------------------------------------------------------------


def sync_pulseox_to_camera(
    pulseox: PulseOxRecord,
    t_cam: np.ndarray,
    fps_cam: float = 30.0,
    bandpass_low: float = 0.75,
    bandpass_high: float = 3.0,
    bandpass_order: int = 4,
) -> Optional[SyncedGT]:
    """
    Core synchronisation pipeline (Steps 3–6).

    Parameters
    ----------
    pulseox   : loaded PulseOxRecord (BVP at 125 Hz with Unix timestamps)
    t_cam     : (N_frames,) Unix timestamps from cam0_full_log.txt
    fps_cam   : nominal camera frame rate (used for filter design only)
    bandpass_* : HR band parameters

    Returns
    -------
    SyncedGT with per-frame BVP aligned to camera timestamps, or None.

    Implementation details
    ──────────────────────
    RELATIVE vs ABSOLUTE timestamps
    ────────────────────────────────
    If pulseOx timestamps are relative (0-based, as happens when the 'time'
    field is absent from the .mat), we cannot directly align them with the
    absolute Unix t_cam timestamps.

    Detection: if max(t_pulseox) < 1e8 (i.e. < year ~1973 in epoch)
    then they're relative.

    Resolution strategy:
      a) Use the 'hr_bpm' field directly for BPM (skip waveform alignment).
      b) Construct a synthetic waveform at the known HR for training purposes.
      c) If cam0_full_log.txt exists alongside a cam0_full.pkl, the pkl may
         carry absolute offsets — we don't rely on pkl to keep dependencies light.

    We apply strategy (b): synthesise a ground-truth sine-wave BVP at the
    reference HR when absolute alignment is not possible.  This is flagged
    in the SyncedGT.n_valid_frames field (set to 0 for synthetic).
    """
    t_po = pulseox.timestamps
    bvp_po = pulseox.bvp_raw

    # ── Step 3a: Determine if timestamps are absolute or relative ──────────
    is_absolute = t_po.max() > 1e8   # ~year 1973 as a sentinel

    if is_absolute:
        # Find temporal overlap between pulseOx and camera
        cam_start, cam_end = t_cam[0], t_cam[-1]
        mask = (t_po >= cam_start - 1.0) & (t_po <= cam_end + 1.0)
        if mask.sum() < 2:
            logger.warning(
                "No pulseOx overlap with camera window "
                "[%.3f, %.3f] vs [%.3f, %.3f]",
                cam_start, cam_end, t_po[0], t_po[-1],
            )
            # Fall back to synthetic if reference HR known
            return _synthetic_gt(pulseox, t_cam, fps_cam)

        t_po_win = t_po[mask]
        bvp_po_win = bvp_po[mask]

    else:
        # Relative timestamps: offset so the pulseOx window spans [cam[0], cam[-1]]
        logger.warning(
            "pulseOx timestamps appear relative (max=%.2f) — aligning by duration",
            t_po.max(),
        )
        cam_duration = t_cam[-1] - t_cam[0]
        po_duration = t_po[-1] - t_po[0]

        if po_duration < cam_duration * 0.5:
            logger.warning("PulseOx duration (%.1fs) much shorter than camera (%.1fs)",
                           po_duration, cam_duration)
            return _synthetic_gt(pulseox, t_cam, fps_cam)

        # Shift relative timestamps to start at cam[0]
        t_po_win = t_po + t_cam[0]
        bvp_po_win = bvp_po

    # ── Step 4: Bandpass filter at pulseOx sample rate ────────────────────
    nyq = pulseox.fs / 2.0
    lo = np.clip(bandpass_low / nyq, 1e-4, 0.9999)
    hi = np.clip(bandpass_high / nyq, 1e-4, 0.9999)
    if lo >= hi:
        logger.warning("Degenerate bandpass [%.4f, %.4f] — skipping filter", lo, hi)
        bvp_filtered = bvp_po_win
    else:
        b, a = scipy.signal.butter(bandpass_order, [lo, hi], btype="band")
        try:
            bvp_filtered = scipy.signal.filtfilt(b, a, bvp_po_win)
        except Exception as e:
            logger.warning("Bandpass filter failed: %s — using raw BVP", e)
            bvp_filtered = bvp_po_win

    # ── Step 5: Interpolate to camera frame timestamps ────────────────────
    try:
        interp_fn = scipy.interpolate.interp1d(
            t_po_win, bvp_filtered,
            kind="linear",
            bounds_error=False,
            fill_value=(bvp_filtered[0], bvp_filtered[-1]),  # edge extension
        )
        bvp_cam = interp_fn(t_cam).astype(np.float32)
    except Exception as e:
        logger.error("Interpolation failed: %s", e)
        return _synthetic_gt(pulseox, t_cam, fps_cam)

    # ── Step 6: Z-score normalise ─────────────────────────────────────────
    std = bvp_cam.std()
    if std > 1e-8:
        bvp_cam = ((bvp_cam - bvp_cam.mean()) / std).astype(np.float32)
    else:
        logger.warning("BVP std ≈ 0 after interpolation — signal may be flat")

    # Count valid frames (within pulseOx coverage)
    valid_mask = (t_cam >= t_po_win[0]) & (t_cam <= t_po_win[-1])
    n_valid = int(valid_mask.sum())

    # Effective camera fps
    fps_eff = (len(t_cam) - 1) / max(t_cam[-1] - t_cam[0], 1e-9)

    return SyncedGT(
        bvp_cam=bvp_cam,
        t_cam=t_cam,
        n_valid_frames=n_valid,
        fs_cam=fps_eff,
        hr_ref_bpm=pulseox.hr_ref_bpm,
        spo2=pulseox.spo2,
    )


def _synthetic_gt(
    pulseox: PulseOxRecord,
    t_cam: np.ndarray,
    fps_cam: float,
) -> Optional[SyncedGT]:
    """
    Fallback: synthesise a ground-truth BVP sine wave at the reference HR.

    Used when absolute timestamp alignment is impossible.
    The resulting waveform has correct frequency but no phase relationship
    to real physiology — adequate for frequency-domain loss training.
    """
    hr_ref = pulseox.hr_ref_bpm
    if hr_ref is None or hr_ref < 20 or hr_ref > 250:
        logger.error("Cannot synthesise GT: no valid hr_ref_bpm")
        return None

    logger.warning(
        "Using SYNTHETIC BVP at %.1f BPM (no absolute timestamp alignment)", hr_ref
    )
    hr_hz = hr_ref / 60.0
    t_rel = t_cam - t_cam[0]
    bvp = np.sin(2 * np.pi * hr_hz * t_rel).astype(np.float32)
    # Add harmonic for realism
    bvp += 0.15 * np.sin(2 * np.pi * 2 * hr_hz * t_rel + 0.4).astype(np.float32)
    bvp = bvp / (bvp.std() + 1e-8)

    return SyncedGT(
        bvp_cam=bvp,
        t_cam=t_cam,
        n_valid_frames=0,   # 0 flags synthetic origin
        fs_cam=fps_cam,
        hr_ref_bpm=hr_ref,
        spo2=pulseox.spo2,
    )


# ---------------------------------------------------------------------------
# Step 7: BPM estimation from synced BVP (evaluation only)
# ---------------------------------------------------------------------------


def bvp_to_bpm(
    bvp_cam: np.ndarray,
    fps: float,
    method: str = "welch",
    low_hz: float = 0.75,
    high_hz: float = 3.0,
) -> float:
    """
    Estimate BPM from a camera-rate BVP waveform (Step 7).

    method: 'welch' (default) or 'fft'

    Welch method averages multiple overlapping periodograms, reducing
    spectral noise at the cost of some frequency resolution.  With a
    10-second clip at 30 fps we have 300 samples → 5-second segments
    (150 samples) → frequency resolution ≈ 0.2 Hz ≈ 12 BPM.

    Parabolic interpolation between FFT bins gives sub-bin accuracy
    (≈ ±0.2 BPM) without requiring longer recording windows.
    """
    from src.utils.signal_processing import estimate_bpm, bandpass_filter, normalise_signal
    try:
        sig = bandpass_filter(bvp_cam.astype(np.float64), fps, low_hz, high_hz)
        sig = normalise_signal(sig, "zscore")
        return estimate_bpm(sig, fps, method=method, peak_pick="parabolic",
                            low_hz=low_hz, high_hz=high_hz)
    except Exception as e:
        logger.warning("BPM estimation failed: %s", e)
        return float("nan")


# ---------------------------------------------------------------------------
# High-level convenience: load everything for one session
# ---------------------------------------------------------------------------


def load_session_gt(
    session_dir: Path,
    n_frames: int,
    fps_cam: float = 30.0,
    prefer_full_log: bool = True,
) -> Optional[SyncedGT]:
    """
    One-call entry point: given a session directory, return the synced GT.

    Parameters
    ----------
    session_dir  : path to one session (contains NIR/ PGMs)
    n_frames     : number of PGM frames (used to validate alignment)
    fps_cam      : nominal camera fps (used only for fallback)
    prefer_full_log : use cam0_full_log over cam0_partial_log

    Returns
    -------
    SyncedGT or None
    """
    # 1. Find and load camera timestamps
    cam_log = find_camera_log(session_dir, prefer_full=prefer_full_log)
    if cam_log is None:
        logger.error("No camera log for %s", session_dir)
        return None

    t_cam = load_camera_timestamps(cam_log)
    if t_cam is None:
        return None

    # Validate frame count
    if abs(len(t_cam) - n_frames) > max(5, int(0.05 * n_frames)):
        logger.warning(
            "Camera timestamps (%d) vs PGM frames (%d) mismatch in %s — "
            "using min of both",
            len(t_cam), n_frames, session_dir,
        )
        # Trim to the shorter of the two
        n_use = min(len(t_cam), n_frames)
        t_cam = t_cam[:n_use]

    # 2. Find and load pulseOx.mat
    mat_path = find_pulseox_mat(session_dir)
    if mat_path is None:
        logger.error("pulseOx.mat not found for %s", session_dir)
        return None

    pulseox = load_pulseox_mat(mat_path)
    if pulseox is None:
        return None

    # 3–6. Synchronise
    synced = sync_pulseox_to_camera(pulseox, t_cam, fps_cam=fps_cam)
    if synced is None:
        return None

    # Pad or trim to n_frames
    N = len(synced.bvp_cam)
    if N > len(t_cam):
        synced.bvp_cam = synced.bvp_cam[:len(t_cam)]
    elif N < len(t_cam):
        pad = len(t_cam) - N
        synced.bvp_cam = np.concatenate(
            [synced.bvp_cam, np.full(pad, synced.bvp_cam[-1], dtype=np.float32)]
        )

    valid_pct = 100.0 * synced.n_valid_frames / max(len(t_cam), 1)
    logger.info(
        "GT synced for %s: %d frames  valid=%.0f%%  hr_ref=%s BPM",
        session_dir.name, len(synced.bvp_cam), valid_pct,
        f"{synced.hr_ref_bpm:.1f}" if synced.hr_ref_bpm else "—",
    )
    return synced


# ---------------------------------------------------------------------------
# MAT field extraction helpers
# ---------------------------------------------------------------------------


def _extract_array(
    mat: dict,
    keys: List[str],
    path: Path,
) -> Optional[np.ndarray]:
    """Try each key name; return the first found as a numpy array."""
    for k in keys:
        if k in mat:
            val = mat[k]
            # Handle MATLAB struct objects (scipy squeeze_me flattens them)
            if hasattr(val, "__array__"):
                return np.asarray(val, dtype=np.float64)
            elif hasattr(val, "ravel"):
                return val.astype(np.float64)
    return None


def _extract_scalar(mat: dict, keys: List[str]) -> Optional[float]:
    """Try each key name; return as a scalar float."""
    for k in keys:
        if k in mat:
            val = mat[k]
            try:
                v = float(np.squeeze(np.asarray(val)))
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return None

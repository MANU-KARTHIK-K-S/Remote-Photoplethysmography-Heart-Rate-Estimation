#!/usr/bin/env python3
"""
scripts/inspect_mat.py
────────────────────────
Diagnostic tool — inspect a pulseOx.mat file and camera timestamp log.

Run this BEFORE preprocessing to verify your GT files are readable
and to understand the exact field names and sample rates in your dataset.

Usage
─────
    # Inspect a single .mat file:
    python scripts/inspect_mat.py --mat /path/to/Subject1/PulseOX/pulseOx.mat

    # Inspect .mat + camera log together (full GT derivation preview):
    python scripts/inspect_mat.py \\
        --mat  /path/to/Subject1/PulseOX/pulseOx.mat \\
        --log  /path/to/Subject1/session1/cam0_full_log.txt \\
        --plot          # save waveform PNG to current directory

    # Inspect entire dataset (all subjects):
    python scripts/inspect_mat.py --dataset-root /path/to/mr-nirp-d

Output for each file
─────────────────────
    ┌─────────────────────────────────────────────┐
    │  pulseOx.mat inspection                     │
    ├─────────────────────────────────────────────┤
    │  Key          Shape          dtype  Preview  │
    │  time         (1, 1500)      float64 1540... │
    │  data         (1, 1500)      float64 0.81... │
    │  rate         (1, 1)         float64 125.0   │
    │  spo2         (1, 1)         float64 98.3    │
    │  hr_bpm       (1, 1)         float64 72.5    │
    ├─────────────────────────────────────────────┤
    │  Sample rate  : 125.0 Hz                    │
    │  Duration     : 12.00 s                     │
    │  HR reference : 72.5 BPM                    │
    │  SpO2         : 98.3 %                      │
    │  Timestamps   : absolute (Unix epoch)       │
    │  Estimated HR : 72.1 BPM  (Welch FFT)       │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Pretty-printer for .mat contents
# ---------------------------------------------------------------------------

def inspect_mat(mat_path: Path, verbose: bool = True) -> dict:
    """Load and print contents of a pulseOx.mat file."""
    try:
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=False)
    except Exception as e:
        print(f"  ERROR loading {mat_path}: {e}")
        return {}

    user_keys = {k: v for k, v in mat.items() if not k.startswith("_")}

    print(f"\n{'═'*60}")
    print(f"  pulseOx.mat: {mat_path}")
    print(f"{'═'*60}")
    print(f"  {'Key':<18} {'Shape':<18} {'dtype':<10} Preview")
    print(f"  {'─'*56}")

    summary = {}
    for k, v in sorted(user_keys.items()):
        arr = np.asarray(v)
        preview = ""
        if arr.size > 0:
            flat = arr.ravel().astype(float)
            if arr.size == 1:
                preview = f"{flat[0]:.4g}"
            else:
                preview = f"[{flat[0]:.4g} … {flat[-1]:.4g}]  mean={flat.mean():.4g}"
        print(f"  {k:<18} {str(arr.shape):<18} {str(arr.dtype):<10} {preview}")
        summary[k] = arr

    # Infer key facts
    print(f"\n  Analysis:")

    # Sample rate
    rate = None
    for rk in ["rate", "fs", "Fs", "sample_rate", "hz"]:
        if rk in summary:
            rate = float(np.squeeze(summary[rk]))
            break
    if rate:
        print(f"  Sample rate : {rate:.1f} Hz")

    # BVP data
    bvp = None
    for dk in ["data", "bvp", "BVP", "ppg", "PPG", "signal","pulseOxRecord"]:
        if dk in summary:
            bvp = summary[dk].ravel().astype(float)
            break
    if bvp is not None and rate:
        dur = len(bvp) / rate
        print(f"  BVP samples : {len(bvp)} → {dur:.2f} s @ {rate:.0f} Hz")

        # Estimate HR from the BVP
        lo, hi = 0.75, 3.0
        nyq = rate / 2.0
        b, a = scipy.signal.butter(4, [lo/nyq, hi/nyq], btype="band")
        try:
            bvp_f = scipy.signal.filtfilt(b, a, bvp)
            freqs, power = scipy.signal.welch(bvp_f, fs=rate,
                                              nperseg=min(len(bvp_f), int(5*rate)))
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.sum() > 0:
                peak = freqs[mask][np.argmax(power[mask])]
                hr_est = peak * 60.0
                print(f"  Estimated HR: {hr_est:.1f} BPM  (Welch FFT in [{lo},{hi}] Hz)")
        except Exception:
            pass

    # Timestamps
    t = None
    for tk in ["time", "Time", "t", "timestamps","pulseOxTime"]:
        if tk in summary:
            t = summary[tk].ravel().astype(float)
            break
    if t is not None:
        is_abs = t.max() > 1e8
        print(f"  Timestamps  : {'absolute (Unix epoch)' if is_abs else 'relative (0-based)'}")
        if is_abs:
            from datetime import datetime, timezone

            dt = datetime.fromtimestamp(t[0], tz=timezone.utc)
            print(f"  t_start     : {t[0]:.3f}  ({dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")

    # SpO2 & reference HR
    for sk in ["spo2", "SpO2"]:
        if sk in summary:
            print(f"  SpO2        : {float(np.squeeze(summary[sk])):.1f} %")
    for hk in ["hr_bpm", "hr", "HR"]:
        if hk in summary:
            print(f"  HR ref      : {float(np.squeeze(summary[hk])):.1f} BPM")

    return summary


# ---------------------------------------------------------------------------
# Camera log inspector
# ---------------------------------------------------------------------------

def inspect_cam_log(log_path: Path) -> np.ndarray | None:
    """Parse and describe a cam0_full_log.txt file."""
    if not log_path.exists():
        print(f"  Camera log not found: {log_path}")
        return None

    raw = log_path.read_text().strip().replace("\n", " ")
    try:
        ts_list = ast.literal_eval(raw)
    except Exception as e:
        print(f"  ERROR parsing {log_path}: {e}")
        return None

    ts = np.array(ts_list, dtype=np.float64)
    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(ts[0], tz=timezone.utc)

    print(f"\n  Camera log: {log_path.name}")
    print(f"  Frames     : {len(ts)}")
    print(f"  Effective fps: {fps:.2f} Hz")
    print(f"  t_start    : {ts[0]:.3f}  ({dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
    print(f"  Duration   : {ts[-1]-ts[0]:.2f} s")

    if len(ts) > 1:
        diffs = np.diff(ts)
        dropped = (diffs > 2.0/fps).sum() if fps > 0 else 0
        print(f"  Dropped frames (>2x interval): {dropped}")
    return ts


# ---------------------------------------------------------------------------
# Full GT derivation preview
# ---------------------------------------------------------------------------

def preview_gt_derivation(mat_path: Path, log_path: Path, n_cam_frames: int | None = None):
    """Show the 6-step GT derivation for a given session."""
    from src.data.gt_loader import (
        load_pulseox_mat, load_camera_timestamps,
        sync_pulseox_to_camera, bvp_to_bpm,
    )

    print(f"\n{'═'*60}")
    print(f"  GT Derivation Preview")
    print(f"{'═'*60}")
    print(f"  Step 1: Load pulseOx.mat ...")
    pulseox = load_pulseox_mat(mat_path)
    if pulseox is None:
        print("  FAILED"); return

    print(f"         N={len(pulseox.bvp_raw)} samples  fs={pulseox.fs:.0f} Hz  "
          f"hr_ref={pulseox.hr_ref_bpm}")

    print(f"  Step 2: Parse camera timestamps ...")
    t_cam = load_camera_timestamps(log_path)
    if t_cam is None:
        print("  FAILED"); return
    fps_cam = (len(t_cam)-1)/(t_cam[-1]-t_cam[0]) if len(t_cam) > 1 else 30.0
    print(f"         N={len(t_cam)} frames  fps≈{fps_cam:.1f}")

    print(f"  Step 3-6: Bandpass → sync → interpolate → normalise ...")
    synced = sync_pulseox_to_camera(pulseox, t_cam, fps_cam=fps_cam)
    if synced is None:
        print("  FAILED"); return
    print(f"           bvp_cam.shape={synced.bvp_cam.shape}  "
          f"valid={synced.n_valid_frames}/{len(t_cam)} frames "
          f"({'abs' if synced.n_valid_frames>0 else 'synthetic'})")

    print(f"  Step 7: BPM estimation (Welch PSD) ...")
    bpm_est = bvp_to_bpm(synced.bvp_cam, fps_cam)
    ref = pulseox.hr_ref_bpm
    err = abs(bpm_est - ref) if ref else float("nan")
    ref_str = f"{ref:.1f}" if ref is not None else "N/A"
    print(f"         Estimated = {bpm_est:.1f} BPM  |  Reference = {ref_str} BPM  |  "
          f"Error = {err:.1f} BPM")
    print(f"\n  ✓ GT derivation pipeline OK")


# ---------------------------------------------------------------------------
# Dataset-wide scan
# ---------------------------------------------------------------------------

def scan_dataset(dataset_root: Path):
    """Scan all subjects and report GT availability."""
    import re
    print(f"\nScanning dataset: {dataset_root}")
    print(f"{'─'*70}")
    print(f"  {'Subject/Session':<35} {'Mat':>5} {'CamLog':>8} {'Frames':>8} {'HR ref':>8}")
    print(f"{'─'*70}")

    for subj_dir in sorted(dataset_root.iterdir()):
        if not subj_dir.is_dir(): continue
        m = re.search(r"(?:subject|subj|p)_?0*(\d+)", subj_dir.name, re.IGNORECASE)
        if not m: continue

        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir(): continue
            nir = sess_dir / "NIR"
            n_pgm = len(list(nir.glob("*.pgm"))) if nir.exists() else 0
            if n_pgm == 0: continue

            # Find mat
            mat_candidates = [
                sess_dir.parent / "PulseOX" / "pulseOx.mat",
                sess_dir / "PulseOX" / "pulseOx.mat",
                sess_dir / "pulseOx.mat",
            ]
            mat_ok = any(p.exists() for p in mat_candidates)
            mat_p = next((p for p in mat_candidates if p.exists()), None)

            # Find cam log
            cam_ok = any((sess_dir/n).exists() for n in
                         ["cam0_full_log.txt","cam0_partial_log.txt"])

            # Reference HR
            hr_ref = "—"
            if mat_p:
                try:
                    mat = scipy.io.loadmat(str(mat_p), squeeze_me=True)
                    for hk in ["hr_bpm","hr","HR"]:
                        if hk in mat:
                            hr_ref = f"{float(np.squeeze(mat[hk])):.1f}"
                            break
                except Exception:
                    pass

            label = f"{subj_dir.name}/{sess_dir.name}"
            print(f"  {label:<35} {'✓' if mat_ok else '✗':>5} "
                  f"{'✓' if cam_ok else '✗':>8} {n_pgm:>8} {hr_ref:>8}")

    print(f"{'─'*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect MR-NIRP-D GT files")
    parser.add_argument("--mat",  type=Path, default=None,
                        help="Path to pulseOx.mat")
    parser.add_argument("--log",  type=Path, default=None,
                        help="Path to cam0_full_log.txt")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="Scan entire dataset root")
    parser.add_argument("--plot", action="store_true",
                        help="Save BVP waveform plot as PNG")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.dataset_root:
        scan_dataset(args.dataset_root)
        return

    if args.mat is None and args.log is None:
        parser.print_help()
        sys.exit(1)

    if args.mat:
        summary = inspect_mat(args.mat, verbose=args.verbose)

    if args.log:
        ts = inspect_cam_log(args.log)

    if args.mat and args.log:
        preview_gt_derivation(args.mat, args.log)

    if args.plot and args.mat:
        _plot_bvp(args.mat, args.log)


def _plot_bvp(mat_path: Path, log_path: Path | None):
    """Save a BVP waveform plot showing GT derivation."""
    try:
        import matplotlib.pyplot as plt
        from src.data.gt_loader import load_pulseox_mat, load_camera_timestamps, sync_pulseox_to_camera

        pulseox = load_pulseox_mat(mat_path)
        if pulseox is None: return
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

        # Raw BVP at pulseOx rate
        t_po = pulseox.timestamps - pulseox.timestamps[0]
        axes[0].plot(t_po, pulseox.bvp_raw, lw=0.8, color="steelblue",
                     label=f"Raw pulseOx BVP  ({pulseox.fs:.0f} Hz)")
        axes[0].set_ylabel("Amplitude"); axes[0].legend()
        axes[0].set_title(f"pulseOx.mat: {mat_path.name}")

        # Synced BVP at camera rate
        if log_path and log_path.exists():
            t_cam = load_camera_timestamps(log_path)
            fps = (len(t_cam)-1)/(t_cam[-1]-t_cam[0]) if len(t_cam)>1 else 30.0
            synced = sync_pulseox_to_camera(pulseox, t_cam, fps_cam=fps)
            if synced is not None:
                t_rel = synced.t_cam - synced.t_cam[0]
                axes[1].plot(t_rel, synced.bvp_cam, lw=1.2, color="tomato",
                             label=f"Synced to camera ({fps:.0f} fps)")
                axes[1].set_ylabel("Z-score"); axes[1].set_xlabel("Time (s)")
                axes[1].legend()
                axes[1].set_title("Bandpass-filtered + interpolated to camera timestamps")

        plt.tight_layout()
        save_path = Path("bvp_inspection.png")
        fig.savefig(save_path, dpi=120)
        print(f"\n  Plot saved: {save_path.resolve()}")
        plt.close(fig)
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    main()

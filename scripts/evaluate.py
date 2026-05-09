#!/usr/bin/env python3
"""
scripts/evaluate.py
────────────────────
Comprehensive evaluation of a trained PhysFormer-NIR checkpoint.

Outputs
───────
* Console table: per-subject MAE, RMSE, pearson_r, SNR
* CSV file:      per-clip predictions vs GT BPM
* Plots saved to eval_outputs/:
    - bland_altman.png   — agreement analysis between pred and GT BPM
    - scatter_bpm.png    — scatter plot of pred vs GT BPM
    - snr_hist.png       — histogram of SNR values
    - rppg_examples.png  — 4 example waveforms (best / worst / median)
* Prints a formatted summary table

Diagnostic sections
───────────────────
Section A: Overall aggregate metrics
Section B: Per-subject breakdown  (identifies outlier subjects)
Section C: Error analysis  (where does the model fail?)
Section D: Spectral quality  (SNR distribution)

Usage
─────
    python scripts/evaluate.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/physformer_nir_best.pth \
        --split test \
        [--save-dir eval_outputs]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import build_dataloaders
from src.evaluation.metrics import HRMetrics, rppg_to_bpm
from src.models.physformer_nir import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_bland_altman(pred_bpms, gt_bpms, save_path: Path):
    """Bland-Altman agreement plot between predicted and GT BPM."""
    import matplotlib.pyplot as plt
    mean = (pred_bpms + gt_bpms) / 2.0
    diff = pred_bpms - gt_bpms
    md = diff.mean()
    sd = diff.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean, diff, alpha=0.5, s=15, color="steelblue", label="Clips")
    ax.axhline(md, color="tomato", lw=2, label=f"Mean diff: {md:+.2f} BPM")
    ax.axhline(md + 1.96 * sd, color="tomato", lw=1.5, ls="--",
               label=f"±1.96 SD: [{md-1.96*sd:.1f}, {md+1.96*sd:.1f}]")
    ax.axhline(md - 1.96 * sd, color="tomato", lw=1.5, ls="--")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Mean BPM ((Pred + GT) / 2)")
    ax.set_ylabel("Difference (Pred − GT) [BPM]")
    ax.set_title("Bland-Altman Agreement Plot")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_scatter(pred_bpms, gt_bpms, save_path: Path):
    """Scatter: GT BPM (x) vs Predicted BPM (y)."""
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    r, _ = pearsonr(pred_bpms, gt_bpms) if len(pred_bpms) > 2 else (0, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    mn, mx = min(pred_bpms.min(), gt_bpms.min()) - 5, max(pred_bpms.max(), gt_bpms.max()) + 5
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="Perfect")
    ax.scatter(gt_bpms, pred_bpms, alpha=0.4, s=15, color="steelblue")
    ax.set_xlabel("GT BPM"); ax.set_ylabel("Predicted BPM")
    ax.set_title(f"Predicted vs GT BPM  (r={r:.3f})")
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_rppg_examples(results, save_path: Path, fps: float, n: int = 4):
    """Plot best, worst, and median rPPG waveform examples."""
    import matplotlib.pyplot as plt
    from src.utils.signal_processing import normalise_signal, bandpass_filter

    errors = [abs(r.pred_bpm - r.gt_bpm) for r in results]
    sorted_idx = np.argsort(errors)
    pick_idxs = [
        sorted_idx[0],               # best
        sorted_idx[len(sorted_idx)//4],    # 25th percentile
        sorted_idx[len(sorted_idx)//2],    # median
        sorted_idx[-1],              # worst
    ]
    labels = ["Best", "25th pct", "Median", "Worst"]

    fig, axes = plt.subplots(n, 1, figsize=(14, n * 3), sharex=False)
    for ax, idx, label in zip(axes, pick_idxs, labels):
        r = results[int(idx)]
        T = len(r.gt_rppg)
        t = np.arange(T) / fps
        gt_f = normalise_signal(bandpass_filter(r.gt_rppg, fps))
        pd_f = normalise_signal(bandpass_filter(r.pred_rppg, fps))
        ax.plot(t, gt_f, label="GT BVP", color="steelblue", lw=1.2, alpha=0.85)
        ax.plot(t, pd_f, label="Pred rPPG", color="tomato", lw=1.2, alpha=0.85)
        ax.set_title(
            f"{label} | Subj {r.subject_id} | GT={r.gt_bpm:.1f} BPM  Pred={r.pred_bpm:.1f} BPM  "
            f"Err={abs(r.pred_bpm-r.gt_bpm):.1f} BPM  SNR={r.snr_db:.1f} dB"
        )
        ax.set_ylabel("Normalised amp."); ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Time (s)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_snr_hist(results, save_path: Path):
    import matplotlib.pyplot as plt
    snrs = np.array([r.snr_db for r in results])
    snrs = snrs[np.isfinite(snrs)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(snrs, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(snrs.mean(), color="tomato", lw=2, label=f"Mean={snrs.mean():.1f} dB")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Count")
    ax.set_title("Distribution of rPPG Signal-to-Noise Ratio"); ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(results, fps: float):
    """Identify systematic failure modes."""
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "subject": r.subject_id,
            "session": r.session,
            "gt_bpm": r.gt_bpm,
            "pred_bpm": r.pred_bpm,
            "abs_err": abs(r.pred_bpm - r.gt_bpm),
            "snr_db": r.snr_db,
            "waveform_pearson": r.waveform_pearson,
        })
    df = pd.DataFrame(rows)

    logger.info("\n── Error Analysis ─────────────────────────────────")
    # High-error clips
    high_err = df[df["abs_err"] > 10.0]
    logger.info(
        "Clips with >10 BPM error: %d / %d (%.1f%%)",
        len(high_err), len(df), 100 * len(high_err) / max(len(df), 1),
    )

    # HR range analysis
    bins = [(45, 75, "low (<75)"), (75, 100, "normal (75–100)"), (100, 180, "high (>100)")]
    for lo, hi, name in bins:
        subset = df[(df["gt_bpm"] >= lo) & (df["gt_bpm"] < hi)]
        if len(subset) > 0:
            logger.info(
                "  HR %s: MAE=%.2f BPM (n=%d)", name, subset["abs_err"].mean(), len(subset)
            )

    # SNR vs error correlation
    snr_valid = df[np.isfinite(df["snr_db"])]
    if len(snr_valid) > 5:
        from scipy.stats import pearsonr
        r, p = pearsonr(-snr_valid["snr_db"], snr_valid["abs_err"])
        logger.info("SNR vs absolute error: r=%.3f (p=%.4f) — low SNR → higher error", r, p)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhysFormer-NIR checkpoint")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--save-dir", default="eval_outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fps = float(cfg["dataset"]["fps"])

    # Build model and load checkpoint
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    logger.info("Loaded checkpoint (epoch %d, val_MAE=%.2f)", ckpt.get("epoch", -1), ckpt.get("val_mae", -1))

    # Build dataloaders
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = test_loader if args.split == "test" else val_loader

    # Inference
    metrics = HRMetrics(fps=fps)
    logger.info("Running inference on %s split ...", args.split)

    with torch.no_grad():
        for frames, gt, meta in loader:
            frames = frames.to(device)
            pred = model(frames).cpu().numpy()
            for b in range(pred.shape[0]):
                mb = {k: (v[b] if hasattr(v, "__getitem__") else v) for k, v in meta.items()}
                metrics.add(pred[b], gt[b].numpy(), mb)

    # A. Aggregate metrics
    logger.info("\n── SECTION A: Aggregate Metrics ────────────────────")
    summary = metrics.log_summary(args.split)

    # B. Per-subject breakdown
    logger.info("\n── SECTION B: Per-subject Breakdown ────────────────")
    per_subj = metrics.per_subject_summary()
    for subj, m in per_subj.items():
        logger.info("  Subject %2d: MAE=%.2f BPM  pearson=%.3f  n=%d",
                    subj, m["MAE"], m["pearson_r"], m["n_clips"])

    results = metrics.results

    # C. Error analysis
    logger.info("\n── SECTION C: Error Analysis ───────────────────────")
    df = error_analysis(results, fps)
    df.to_csv(save_dir / "per_clip_results.csv", index=False)
    logger.info("Per-clip CSV saved to %s", save_dir / "per_clip_results.csv")

    # D. Plots
    logger.info("\n── SECTION D: Generating Plots ────────────────────")
    pred_bpms = np.array([r.pred_bpm for r in results])
    gt_bpms = np.array([r.gt_bpm for r in results])
    valid = np.isfinite(pred_bpms) & np.isfinite(gt_bpms)

    if valid.sum() > 2:
        plot_bland_altman(pred_bpms[valid], gt_bpms[valid], save_dir / "bland_altman.png")
        plot_scatter(pred_bpms[valid], gt_bpms[valid], save_dir / "scatter_bpm.png")
        plot_snr_hist(results, save_dir / "snr_hist.png")

    if len(results) >= 4:
        plot_rppg_examples(results, save_dir / "rppg_examples.png", fps)

    # Final summary table
    print("\n" + "=" * 55)
    print(f"  PhysFormer-NIR  |  MR-NIRP-D  |  Split: {args.split.upper()}")
    print("=" * 55)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:>10.4f}")
        else:
            print(f"  {k:<25} {v:>10}")
    print("=" * 55)
    print(f"  Plots saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()

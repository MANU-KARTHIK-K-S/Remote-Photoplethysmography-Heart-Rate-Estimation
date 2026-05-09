"""
src/evaluation/metrics.py
──────────────────────────
Evaluation metrics for HR estimation.

Per-clip BPM metrics (used at test time)
─────────────────────────────────────────
  MAE        Mean Absolute Error (BPM)   — primary metric
  RMSE       Root Mean Squared Error (BPM)
  MAPE       Mean Absolute Percentage Error (%)
  pearson_r  Pearson correlation between predicted and GT BPM vectors
  SNR        Signal-to-Noise Ratio of the predicted rPPG waveform (dB)

Per-waveform metrics (used during training validation)
───────────────────────────────────────────────────────
  pearson_waveform  Pearson correlation between predicted and GT rPPG waveforms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.utils.signal_processing import (
    bandpass_filter,
    compute_snr,
    estimate_bpm,
    normalise_signal,
    pearson_correlation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-clip result container
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    subject_id: int
    session: str
    start_frame: int
    pred_bpm: float
    gt_bpm: float
    pred_rppg: np.ndarray   # (T,)
    gt_rppg: np.ndarray     # (T,)
    snr_db: float = 0.0
    waveform_pearson: float = 0.0


# ---------------------------------------------------------------------------
# BPM estimation from rPPG waveform
# ---------------------------------------------------------------------------


def rppg_to_bpm(
    rppg: np.ndarray,
    fps: float,
    method: str = "welch",
    peak_pick: str = "parabolic",
    apply_bandpass: bool = True,
) -> float:
    """
    Estimate BPM from a 1-D rPPG waveform.
    Optionally applies bandpass filtering first.
    """
    sig = rppg.copy().astype(np.float64)
    if apply_bandpass:
        try:
            sig = bandpass_filter(sig, fps)
        except Exception as e:
            logger.debug("Bandpass filter failed: %s", e)
    sig = normalise_signal(sig, method="zscore")
    return estimate_bpm(sig, fps, method=method, peak_pick=peak_pick)


def gt_signal_to_bpm(gt: np.ndarray, fps: float) -> float:
    """Estimate ground-truth BPM from a BVP signal."""
    return rppg_to_bpm(gt, fps, method="welch")


# ---------------------------------------------------------------------------
# Aggregate metrics computation
# ---------------------------------------------------------------------------


class HRMetrics:
    """
    Accumulates per-clip results and computes aggregate metrics.

    Usage
    ─────
        metrics = HRMetrics(fps=30.0)
        for pred_rppg, gt_rppg, meta in results:
            metrics.add(pred_rppg, gt_rppg, meta)
        summary = metrics.compute()
    """

    def __init__(self, fps: float = 30.0, method: str = "welch"):
        self.fps = fps
        self.method = method
        self.results: List[ClipResult] = []

    def add(
        self,
        pred_rppg: np.ndarray,
        gt_rppg: np.ndarray,
        meta: Optional[Dict] = None,
    ) -> ClipResult:
        """
        Process one clip: estimate BPM for pred and GT, compute waveform metrics.

        Parameters
        ----------
        pred_rppg : (T,) predicted rPPG waveform
        gt_rppg   : (T,) ground-truth BVP waveform
        meta      : optional dict from Dataset with subject/session info

        Returns
        -------
        ClipResult for this clip (also stored internally)
        """
        meta = meta or {}
        pred_bpm = rppg_to_bpm(pred_rppg, self.fps, method=self.method)
        gt_bpm = rppg_to_bpm(gt_rppg, self.fps, method=self.method)

        # Signal-level SNR
        try:
            snr = compute_snr(pred_rppg, self.fps, gt_bpm)
        except Exception:
            snr = float("nan")

        # Waveform-level Pearson
        wp = pearson_correlation(
            normalise_signal(pred_rppg, "zscore"),
            normalise_signal(gt_rppg, "zscore"),
        )

        result = ClipResult(
            subject_id=int(meta.get("subject", -1)),
            session=str(meta.get("session", "unknown")),
            start_frame=int(meta.get("start_frame", -1)),
            pred_bpm=pred_bpm,
            gt_bpm=gt_bpm,
            pred_rppg=pred_rppg.copy(),
            gt_rppg=gt_rppg.copy(),
            snr_db=snr,
            waveform_pearson=wp,
        )
        self.results.append(result)
        return result

    def compute(self) -> Dict[str, float]:
        """
        Aggregate all stored ClipResults into a metrics summary dict.

        Returns
        ───────
        {
          'MAE':        float   (BPM)
          'RMSE':       float   (BPM)
          'MAPE':       float   (%)
          'pearson_r':  float
          'SNR_mean':   float   (dB)
          'waveform_pearson': float
          'n_clips':    int
        }
        """
        if not self.results:
            logger.warning("HRMetrics.compute() called with no results")
            return {}

        pred_bpms = np.array([r.pred_bpm for r in self.results])
        gt_bpms = np.array([r.gt_bpm for r in self.results])
        snrs = np.array([r.snr_db for r in self.results])
        wps = np.array([r.waveform_pearson for r in self.results])

        # Filter out NaN
        valid = np.isfinite(pred_bpms) & np.isfinite(gt_bpms)
        if valid.sum() == 0:
            logger.error("All BPM estimates are NaN — check model output")
            return {"MAE": float("nan"), "n_clips": 0}

        p, g = pred_bpms[valid], gt_bpms[valid]
        abs_err = np.abs(p - g)

        mae = float(abs_err.mean())
        rmse = float(np.sqrt((abs_err ** 2).mean()))
        mape = float((abs_err / (g + 1e-9) * 100).mean())
        rho = float(pearson_correlation(p, g))
        snr_mean = float(np.nanmean(snrs))
        wp_mean = float(np.nanmean(wps))

        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "pearson_r": rho,
            "SNR_mean": snr_mean,
            "waveform_pearson": wp_mean,
            "n_clips": int(valid.sum()),
        }
        return metrics

    def reset(self):
        self.results.clear()

    def log_summary(self, split: str = "test"):
        m = self.compute()
        logger.info(
            "[%s] MAE=%.2f BPM  RMSE=%.2f  MAPE=%.1f%%  Pearson=%.3f  SNR=%.1f dB  n=%d",
            split, m.get("MAE", 0), m.get("RMSE", 0),
            m.get("MAPE", 0), m.get("pearson_r", 0),
            m.get("SNR_mean", 0), m.get("n_clips", 0),
        )
        return m

    def per_subject_summary(self) -> Dict[int, Dict[str, float]]:
        """Compute per-subject MAE and pearson for analysis."""
        subjects = set(r.subject_id for r in self.results)
        out = {}
        for subj in sorted(subjects):
            subj_results = [r for r in self.results if r.subject_id == subj]
            preds = np.array([r.pred_bpm for r in subj_results])
            gts = np.array([r.gt_bpm for r in subj_results])
            valid = np.isfinite(preds) & np.isfinite(gts)
            if valid.sum() == 0:
                continue
            p, g = preds[valid], gts[valid]
            out[subj] = {
                "MAE": float(np.abs(p - g).mean()),
                "pearson_r": float(pearson_correlation(p, g)),
                "n_clips": int(valid.sum()),
            }
        return out


# ---------------------------------------------------------------------------
# Waveform-level validation metric (used during training)
# ---------------------------------------------------------------------------


def batch_waveform_mae_bpm(
    pred_batch: np.ndarray,
    gt_batch: np.ndarray,
    fps: float,
    method: str = "welch",
) -> float:
    """
    Fast batch BPM-MAE for use inside training validation loop.

    pred_batch, gt_batch: (B, T) numpy arrays
    Returns mean |pred_bpm − gt_bpm| over the batch.
    """
    errors = []
    for pred, gt in zip(pred_batch, gt_batch):
        try:
            p_bpm = rppg_to_bpm(pred, fps, method=method)
            g_bpm = rppg_to_bpm(gt, fps, method=method)
            if np.isfinite(p_bpm) and np.isfinite(g_bpm):
                errors.append(abs(p_bpm - g_bpm))
        except Exception:
            pass
    return float(np.mean(errors)) if errors else float("nan")

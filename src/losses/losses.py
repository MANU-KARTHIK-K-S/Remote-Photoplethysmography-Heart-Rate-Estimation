"""
src/losses/losses.py
─────────────────────
Loss functions for rPPG / HR estimation.

Three complementary losses are combined
──────────────────────────────────────
1. **NegPearsonLoss** (primary, waveform fidelity)
   Maximises Pearson correlation between predicted and GT rPPG signals.
   Standard choice in the rPPG literature (e.g. PhysFormer, TS-CAN).
   Range: [−1, 1] → we minimise (−ρ) so the model aims for ρ → 1.

2. **FreqDomainLoss** (secondary, spectral regulariser)
   Measures KL-divergence between predicted and GT power spectra in the
   HR frequency band.  Forces the model to place spectral energy at the
   correct HR frequency even if the waveform phase is imperfect.
   Based on: PhysFormer (Yu et al. 2022), Eq. 5-6.

3. **SNRLoss** (auxiliary, harmonic suppression)
   Penalises energy outside the true HR peak + its first harmonic.
   Helps suppress motion-artefact frequency components that satisfy
   correlation but scatter spectral energy.

Combined loss
─────────────
    L = w_pearson × L_pearson + w_freq × L_freq + w_snr × L_snr

References
──────────
[1] Yu Z. et al. "PhysFormer" CVPR 2022  (Neg-Pearson + frequency CE)
[2] Liu X. et al. "TS-CAN" NeurIPS 2020  (Neg-Pearson primary loss)
[3] de Haan G. & Jeanne V. IEEE T-BME 2013  (SNR definition for rPPG)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Negative Pearson Correlation Loss
# ---------------------------------------------------------------------------


class NegPearsonLoss(nn.Module):
    """
    Loss = 1 − Pearson(pred, gt)   (minimising → maximising correlation).

    Computed per sample in the batch, then averaged.

    Notes
    ─────
    * pred and gt must have the same length T.
    * We add a small ε for numerical stability when std ≈ 0.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : (B, T) predicted rPPG signal
        gt   : (B, T) ground-truth BVP signal

        Returns
        -------
        scalar loss
        """
        # Centre signals
        pred_c = pred - pred.mean(dim=1, keepdim=True)
        gt_c = gt - gt.mean(dim=1, keepdim=True)

        # Pearson correlation per sample
        num = (pred_c * gt_c).sum(dim=1)
        den = torch.sqrt(
            (pred_c ** 2).sum(dim=1) * (gt_c ** 2).sum(dim=1) + self.eps
        )
        rho = num / den                   # (B,)
        loss = (1.0 - rho).mean()
        return loss


# ---------------------------------------------------------------------------
# 2. Frequency-Domain Cross-Entropy Loss
# ---------------------------------------------------------------------------


def _rppg_to_psd(signal: torch.Tensor, fps: float, n_bins: int) -> torch.Tensor:
    """
    Convert (B, T) signal tensor to (B, n_bins) normalised power spectrum
    in the HR frequency band [0.75, 3.0] Hz.

    Uses rfft with a Hann window; restricts to the HR band before softmax.
    Differentiable — no NumPy used.
    """
    B, T = signal.shape
    # Hann window (on device)
    window = torch.hann_window(T, device=signal.device, dtype=signal.dtype)
    windowed = signal * window.unsqueeze(0)  # (B, T)

    # rfft
    fft = torch.fft.rfft(windowed, n=T, norm="ortho")       # (B, T//2+1) complex
    power = fft.abs() ** 2                                    # (B, T//2+1)

    # Frequency array
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(signal.device)  # (T//2+1,)

    # Restrict to HR band
    low_hz, high_hz = 0.75, 3.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    power_band = power[:, mask]  # (B, n_band)

    # Interpolate / select to exactly n_bins for consistent target alignment
    if power_band.shape[1] != n_bins:
        power_band = F.interpolate(
            power_band.unsqueeze(1), size=n_bins, mode="linear", align_corners=False
        ).squeeze(1)

    # Normalise to probability distribution
    prob = power_band / (power_band.sum(dim=1, keepdim=True) + 1e-9)
    return prob


class FreqDomainLoss(nn.Module):
    """
    KL-divergence between predicted and GT power spectra in the HR band.

    L_freq = KL( P_gt ‖ P_pred )

    where P_gt and P_pred are normalised PSD vectors in [0.75, 3.0] Hz.
    Using KL rather than cross-entropy makes it symmetric-ish and avoids
    log(0) issues via the soft clamp.

    Reference: PhysFormer (Yu et al. 2022), Eq. 5.
    """

    def __init__(self, fps: float = 30.0, n_bins: int = 200, eps: float = 1e-9):
        super().__init__()
        self.fps = fps
        self.n_bins = n_bins
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        P_pred = _rppg_to_psd(pred, self.fps, self.n_bins).clamp(min=self.eps)
        P_gt = _rppg_to_psd(gt, self.fps, self.n_bins).clamp(min=self.eps)

        # KL(P_gt ‖ P_pred) — note: asymmetric, GT is the reference
        kl = (P_gt * (P_gt.log() - P_pred.log())).sum(dim=1).mean()
        return kl


# ---------------------------------------------------------------------------
# 3. SNR Loss
# ---------------------------------------------------------------------------


class SNRLoss(nn.Module):
    """
    Promotes spectral energy concentration at the true HR frequency.

    L_snr = −SNR_db = −10 log10( signal_power / noise_power )

    signal_power = PSD energy in a window around the GT HR frequency ± harmonic
    noise_power  = remaining PSD energy in the HR band

    This loss directly penalises the model for spreading energy to
    motion-artefact frequencies that happen to correlate with the GT waveform.
    """

    def __init__(
        self,
        fps: float = 30.0,
        n_bins: int = 200,
        harmonic_bins: int = 5,       # ±bins around HR peak to count as signal
        eps: float = 1e-8,
    ):
        super().__init__()
        self.fps = fps
        self.n_bins = n_bins
        self.harmonic_bins = harmonic_bins
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        B, T = pred.shape
        P_pred = _rppg_to_psd(pred, self.fps, self.n_bins)   # (B, n_bins)
        P_gt = _rppg_to_psd(gt, self.fps, self.n_bins)       # (B, n_bins)  reference

        # Find the GT HR bin (argmax of GT PSD)
        hr_bin = P_gt.argmax(dim=1)  # (B,)

        snr_vals = []
        hb = self.harmonic_bins
        for b in range(B):
            peak = hr_bin[b].item()
            # Signal bins: HR peak ± hb  and  first harmonic ± hb
            harmonic2 = min(self.n_bins - 1, int(2 * peak))
            lo1, hi1 = max(0, int(peak) - hb), min(self.n_bins, int(peak) + hb + 1)
            lo2, hi2 = max(0, harmonic2 - hb), min(self.n_bins, harmonic2 + hb + 1)

            sig_mask = torch.zeros(self.n_bins, dtype=torch.bool, device=pred.device)
            sig_mask[lo1:hi1] = True
            sig_mask[lo2:hi2] = True

            sig_power = P_pred[b][sig_mask].sum()
            noise_power = P_pred[b][~sig_mask].sum()
            snr_db = 10.0 * torch.log10(sig_power / (noise_power + self.eps))
            snr_vals.append(snr_db)

        # Minimise negative SNR
        return -torch.stack(snr_vals).mean()


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------


class CombinedRPPGLoss(nn.Module):
    """
    Weighted combination of Pearson + Frequency-Domain + SNR losses.

    w_pearson, w_freq, w_snr are loaded from config.
    """

    def __init__(
        self,
        fps: float = 30.0,
        n_bins: int = 200,
        pearson_weight: float = 1.0,
        freq_weight: float = 0.5,
        snr_weight: float = 0.2,
    ):
        super().__init__()
        self.pearson_loss = NegPearsonLoss()
        self.freq_loss = FreqDomainLoss(fps=fps, n_bins=n_bins)
        self.snr_loss = SNRLoss(fps=fps, n_bins=n_bins)
        self.w_pearson = pearson_weight
        self.w_freq = freq_weight
        self.w_snr = snr_weight

    def forward(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns
        -------
        total_loss : scalar tensor (differentiable)
        component_dict : {'pearson': float, 'freq': float, 'snr': float}
        """
        l_p = self.pearson_loss(pred, gt)
        l_f = self.freq_loss(pred, gt)
        l_s = self.snr_loss(pred, gt)

        total = self.w_pearson * l_p + self.w_freq * l_f + self.w_snr * l_s

        components = {
            "loss/pearson": l_p.item(),
            "loss/freq": l_f.item(),
            "loss/snr": l_s.item(),
            "loss/total": total.item(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_loss(cfg: dict) -> CombinedRPPGLoss:
    lcfg = cfg["training"]["loss"]
    ds_cfg = cfg["dataset"]
    return CombinedRPPGLoss(
        fps=float(ds_cfg["fps"]),
        pearson_weight=float(lcfg.get("pearson_weight", 1.0)),
        freq_weight=float(lcfg.get("freq_weight", 0.5)),
        snr_weight=float(lcfg.get("snr_weight", 0.2)),
    )

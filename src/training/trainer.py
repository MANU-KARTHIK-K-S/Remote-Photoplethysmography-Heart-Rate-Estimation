"""
src/training/trainer.py
────────────────────────
Training loop for PhysFormer-NIR on MR-NIRP-D.

Quality-aware loss
──────────────────
The DataLoader (collate_with_quality) returns per-clip loss_weight tensors:
  - Normal clips (well-lit):       loss_weight ≈ 1.0
  - Dark clips (CLAHE-enhanced):   loss_weight ≈ 0.3 (cfg.dark_clip_loss_weight)
  - All-dark clips:                discarded at preprocessing stage

This lets the model learn from partially-dark sessions without being
dominated by low-SNR signals.

Diagnostics logged each step / epoch
──────────────────────────────────────
  loss breakdown   : pearson / freq / snr component weights
  gradient norms   : max / mean / min — flags exploding/vanishing
  dark vs bright   : separate MAE for dark-weighted vs normal clips
  per-subject MAE  : detects subject-specific failure modes
  rPPG waveform    : predicted vs GT plot uploaded to W&B every val epoch
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.evaluation.metrics import HRMetrics
from src.losses.losses import CombinedRPPGLoss
from src.utils.signal_processing import normalise_signal

logger = logging.getLogger(__name__)


# ── LR Schedule ────────────────────────────────────────────────────────────

class WarmupCosineRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_epochs, then CosineAnnealingWarmRestarts."""
    def __init__(self, optimizer, warmup_epochs, cosine_T0, eta_min=1e-6):
        self.warmup = warmup_epochs
        self._cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=cosine_T0, T_mult=2, eta_min=eta_min)
        super().__init__(optimizer)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup:
            return [b * (e + 1) / max(1, self.warmup) for b in self.base_lrs]
        self._cosine.last_epoch = e - self.warmup
        return self._cosine.get_last_lr()

    def step(self, epoch=None):
        super().step(epoch)
        if self.last_epoch >= self.warmup:
            self._cosine.step()


# ── Gradient diagnostics ────────────────────────────────────────────────────

def _grad_stats(model: nn.Module) -> Dict[str, float]:
    norms = [p.grad.data.norm(2).item()
             for p in model.parameters() if p.grad is not None]
    if not norms:
        return {}
    return {"grad/max": max(norms), "grad/mean": sum(norms)/len(norms),
            "grad/min": min(norms)}


# ── Quality-aware per-sample loss ──────────────────────────────────────────

def quality_weighted_loss(
    criterion: CombinedRPPGLoss,
    pred: torch.Tensor,           # (B, T)
    gt:   torch.Tensor,           # (B, T)
    weights: torch.Tensor,        # (B,)  per-clip quality weights
    use_weighting: bool,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute per-sample loss then compute weighted mean.

    Dark clips (weight≈0.3) contribute less to the gradient than normal clips.
    This is equivalent to under-sampling dark clips without discarding them —
    the model still observes their waveform structure.
    """
    B = pred.shape[0]
    losses, comps_acc = [], {"loss/pearson": 0.0, "loss/freq": 0.0, "loss/snr": 0.0}

    for i in range(B):
        l, c = criterion(pred[i:i+1], gt[i:i+1])
        w = float(weights[i]) if use_weighting else 1.0
        losses.append(w * l)
        for k, v in c.items():
            comps_acc[k] = comps_acc.get(k, 0.0) + v * w

    total_w = float(weights.sum()) if use_weighting else float(B)
    total_w = max(total_w, 1e-6)
    total   = torch.stack(losses).sum() / total_w
    for k in comps_acc:
        comps_acc[k] /= total_w
    comps_acc["loss/total"] = total.item()
    return total, comps_acc


# ── Trainer ────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: CombinedRPPGLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
    ):
        self.model      = model.to(device)
        self.criterion  = criterion
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg    = cfg
        self.device = device
        self.fps    = float(cfg["dataset"]["fps"])

        tr = cfg["training"]
        self.epochs      = tr["epochs"]
        self.grad_accum  = tr["gradient_accumulation_steps"]
        self.clip_grad   = tr["clip_grad_norm"]
        self.log_every   = tr["log_every_n_steps"]
        self.val_every   = tr["val_every_n_epochs"]
        self.mixed_prec  = tr.get("mixed_precision", True) and device.type == "cuda"
        self.use_qual_w  = tr.get("use_quality_weighting", True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=tr["learning_rate"],
            weight_decay=tr["weight_decay"],
            betas=(0.9, 0.999),
        )
        self.scheduler = WarmupCosineRestarts(
            self.optimizer,
            warmup_epochs=tr["warmup_epochs"],
            cosine_T0=max(10, self.epochs // 4),
        )
        self.scaler  = GradScaler(enabled=self.mixed_prec)
        self.ckpt_dir = Path(tr["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_mae     = float("inf")
        self.patience_counter = 0
        self.patience         = 15

        self.val_metrics = HRMetrics(fps=self.fps)
        self.use_wandb   = self._init_wandb(cfg)
        try:
            import wandb; self._wb = wandb
        except ImportError:
            self._wb = None

    # ── W&B ──────────────────────────────────────────────────────────────

    def _init_wandb(self, cfg):
        try:
            import wandb
            w = cfg.get("wandb", {})
            wandb.init(project=w.get("project","hr-nir"), entity=w.get("entity"),
                       tags=w.get("tags",[]), config=cfg)
            wandb.watch(self.model, log="gradients", log_freq=100)
            logger.info("W&B run: %s", wandb.run.url)
            return True
        except Exception as e:
            logger.info("W&B disabled (%s)", e)
            return False

    def _log(self, d: Dict, step: int):
        if self.use_wandb and self._wb:
            self._wb.log(d, step=step)

    # ── Training step ─────────────────────────────────────────────────────

    def _train_step(self, frames, gt, weights):
        with autocast(enabled=self.mixed_prec):
            pred = self.model(frames)
            loss, comps = quality_weighted_loss(
                self.criterion, pred, gt, weights, self.use_qual_w)
            loss = loss / self.grad_accum
        self.scaler.scale(loss).backward()
        return loss * self.grad_accum, comps

    # ── Train epoch ───────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        running: Dict[str, float] = {}
        gs = epoch * len(self.train_loader)
        step = 0
        t0 = time.time()
        self.optimizer.zero_grad()

        for bi, batch in enumerate(self.train_loader):
            frames, gt, weights, _ = batch          # collate_with_quality
            frames  = frames.to(self.device,  non_blocking=True)
            gt      = gt.to(self.device,      non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)

            loss, comps = self._train_step(frames, gt, weights)
            for k, v in comps.items():
                running[k] = running.get(k, 0.0) + v

            accum_done = (bi + 1) % self.grad_accum == 0 or \
                         (bi + 1) == len(self.train_loader)
            if accum_done:
                self.scaler.unscale_(self.optimizer)
                grad_st = _grad_stats(self.model)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                step += 1

                if step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = (time.time() - t0) / max(step, 1)
                    log_d = {**comps, **grad_st,
                             "train/lr": lr, "train/step_sec": elapsed}
                    self._log(log_d, gs + step)
                    logger.info(
                        "Ep%03d  step%4d  |  loss=%.4f  (p=%.4f f=%.4f s=%.4f)  "
                        "lr=%.1e  ∇max=%.3f",
                        epoch, step, comps["loss/total"],
                        comps["loss/pearson"], comps["loss/freq"], comps["loss/snr"],
                        lr, grad_st.get("grad/max", 0),
                    )

        n = max(len(self.train_loader), 1)
        return {k: v / n for k, v in running.items()}

    # ── Validation epoch ──────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        self.model.eval()
        self.val_metrics.reset()
        dark_errs, bright_errs = [], []

        for batch in self.val_loader:
            frames, gt, weights, meta_list = batch
            frames = frames.to(self.device, non_blocking=True)
            with autocast(enabled=self.mixed_prec):
                pred = self.model(frames)
            pred_np = pred.cpu().numpy()
            gt_np   = gt.numpy()

            for b in range(pred_np.shape[0]):
                m = meta_list[b] if isinstance(meta_list[b], dict) else {}
                r = self.val_metrics.add(pred_np[b], gt_np[b], m)
                err = abs(r.pred_bpm - r.gt_bpm)
                (dark_errs if float(weights[b]) < 0.7 else bright_errs).append(err)

        summary = self.val_metrics.log_summary("val")

        # ── Dark vs bright diagnostic ─────────────────────────────────────
        if dark_errs:
            logger.info("  Dark clip  MAE=%.2f BPM  (n=%d clips)", np.mean(dark_errs), len(dark_errs))
        if bright_errs:
            logger.info("  Normal clip MAE=%.2f BPM  (n=%d clips)", np.mean(bright_errs), len(bright_errs))
        self._log({
            "val/dark_MAE":   np.mean(dark_errs)   if dark_errs   else float("nan"),
            "val/bright_MAE": np.mean(bright_errs) if bright_errs else float("nan"),
        }, epoch)

        # ── Per-subject diagnostic ────────────────────────────────────────
        for sid, m in self.val_metrics.per_subject_summary().items():
            logger.info("  Subject%d: MAE=%.2f  pearson=%.3f  n=%d",
                        sid, m["MAE"], m["pearson_r"], m["n_clips"])
            self._log({f"val/subj{sid}/MAE": m["MAE"]}, epoch)

        # ── Waveform plot ─────────────────────────────────────────────────
        if self.use_wandb and self.val_metrics.results:
            self._waveform_plot(self.val_metrics.results[0], epoch)

        return summary

    def _waveform_plot(self, result, epoch):
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            t = np.arange(len(result.gt_rppg)) / self.fps
            axes[0].plot(t, normalise_signal(result.gt_rppg), "steelblue", lw=1,
                         label="GT BVP (from pulseOx)")
            axes[0].legend(); axes[0].set_ylabel("GT BVP")
            axes[1].plot(t, normalise_signal(result.pred_rppg), "tomato", lw=1,
                         label="Predicted rPPG")
            axes[1].legend(); axes[1].set_ylabel("Pred"); axes[1].set_xlabel("Time (s)")
            fig.suptitle(f"Ep{epoch} | Subj{result.subject_id} | "
                         f"GT={result.gt_bpm:.1f} pred={result.pred_bpm:.1f} BPM  "
                         f"r={result.waveform_pearson:.3f}")
            plt.tight_layout()
            self._wb.log({"val/waveform": self._wb.Image(fig)}, step=epoch)
            plt.close(fig)
        except Exception as e:
            logger.debug("Waveform plot failed: %s", e)

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save(self, epoch, val_mae, tag="best"):
        p = self.ckpt_dir / f"physformer_nir_{tag}.pth"
        torch.save({
            "epoch": epoch, "val_mae": val_mae, "cfg": self.cfg,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
        }, p)
        logger.info("Checkpoint: %s  (val_MAE=%.2f BPM)", p.name, val_mae)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        logger.info("Loaded %s (epoch %d, val_MAE=%.2f)", path,
                    ckpt.get("epoch",-1), ckpt.get("val_mae", -1))
        return ckpt.get("epoch", 0)

    # ── Main loop ─────────────────────────────────────────────────────────

    def fit(self) -> Dict:
        best = {}
        for epoch in range(1, self.epochs + 1):
            logger.info("═"*55 + f"\n  Epoch {epoch}/{self.epochs}")
            train_m = self.train_epoch(epoch)
            self._log({f"train/{k}": v for k, v in train_m.items()}, epoch)
            self.scheduler.step()

            if epoch % self.val_every == 0 or epoch == self.epochs:
                val_summary = self.validate(epoch)
                val_mae = val_summary.get("MAE", float("inf"))
                self._log({f"val/{k}": v for k, v in val_summary.items()}, epoch)
                self._save(epoch, val_mae, "last")

                if val_mae < self.best_val_mae:
                    self.best_val_mae = val_mae
                    best = val_summary
                    self._save(epoch, val_mae, "best")
                    self.patience_counter = 0
                    logger.info("★ New best val MAE = %.2f BPM", val_mae)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

        if self.use_wandb and self._wb:
            self._wb.finish()
        logger.info("Training done. Best val_MAE=%.2f BPM", self.best_val_mae)
        return best

#!/usr/bin/env python3
"""
scripts/train.py
─────────────────
Train PhysFormer-NIR on MR-NIRP-D (NIR PGM frames, pulseOx GT).

Usage
─────
    python scripts/train.py
    python scripts/train.py training.learning_rate=2e-4 training.epochs=60
    python scripts/train.py --resume checkpoints/physformer_nir_last.pth
"""

from __future__ import annotations
import argparse, logging, os, random, sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def load_config(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        if "=" not in ov:
            continue
        kpath, val = ov.split("=", 1)
        keys = kpath.strip().split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        for cast in (int, float):
            try: val = cast(val); break
            except ValueError: pass
        if val in ("true", "false"):
            val = val == "true"
        d[keys[-1]] = val
        logger.info("Override: %s = %s", kpath, val)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["training"].get("seed", 42), args.deterministic)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    if device.type == "cuda":
        logger.info("GPU: %s  (%.1f GB VRAM)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        logger.warning("No CUDA — training on CPU (slow)")

    logger.info("═"*55)
    logger.info("HR Estimation from NIR — PhysFormer-NIR | MR-NIRP-D")
    logger.info("Input: PGM frames  GT: pulseOx.mat + cam timestamps")
    logger.info("═"*55)

    from src.data.dataset import build_dataloaders
    from src.losses.losses import build_loss
    from src.models.physformer_nir import build_model
    from src.training.trainer import Trainer

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info("Train: %d clips | Val: %d clips | Test: %d clips",
                len(train_loader.dataset),
                len(val_loader.dataset),
                len(test_loader.dataset))

    model     = build_model(cfg)
    criterion = build_loss(cfg)
    trainer   = Trainer(model, criterion, train_loader, val_loader, cfg, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    best = trainer.fit()

    # ── Test set evaluation ──────────────────────────────────────────────
    logger.info("Running test evaluation ...")
    best_ckpt = Path(cfg["training"]["checkpoint_dir"]) / "physformer_nir_best.pth"
    if best_ckpt.exists():
        trainer.load_checkpoint(str(best_ckpt))

    from src.evaluation.metrics import HRMetrics
    test_m = HRMetrics(fps=float(cfg["dataset"]["fps"]))
    model.eval()
    with torch.no_grad():
        for frames, gt, weights, meta_list in test_loader:
            pred = model(frames.to(device)).cpu().numpy()
            for b in range(pred.shape[0]):
                mb = meta_list[b] if isinstance(meta_list[b], dict) else {}
                test_m.add(pred[b], gt[b].numpy(), mb)

    test_m.log_summary("test")
    summary = test_m.compute()

    print("\n" + "═"*50)
    print("  TEST RESULTS  — PhysFormer-NIR / MR-NIRP-D")
    print("═"*50)
    for k, v in summary.items():
        print(f"  {k:<25} {v:.4f}" if isinstance(v, float) else f"  {k:<25} {v}")
    print("═"*50)


if __name__ == "__main__":
    main()

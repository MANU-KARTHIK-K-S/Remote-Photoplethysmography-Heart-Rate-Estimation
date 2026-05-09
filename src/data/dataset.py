"""
src/data/dataset.py
────────────────────
PyTorch Dataset for MR-NIRP-D — reads preprocessed HDF5 shards.

HDF5 shard layout (written by preprocess_dataset.py)
──────────────────────────────────────────────────────
  mr_nirp_subj{ID:02d}.h5
    /clips/
      {0}/
        frames    float32 (T, 2, H, W)   [intensity + temporal-diff]
        gt        float32 (T,)
        quality/  attrs: dark_frame_ratio, loss_weight, ...
        meta      JSON: {subject, session, start_frame, fps, loss_weight, ...}
      {1}/  ...

Quality-aware training
──────────────────────
Each clip carries a `loss_weight` scalar (0.3 – 1.0) stored in HDF5.
The DataLoader collator stacks these weights alongside frames and GT so the
Trainer can apply them as per-sample loss multipliers:
    L_total = sum(loss_weight_i * L_clip_i) / sum(loss_weight_i)

This down-weights dark clips without discarding them, allowing the model
to still learn from partially dark sessions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.gcs_utils import download_file

logger = logging.getLogger(__name__)


class MRNIRPDataset(Dataset):
    """
    MR-NIRP-D NIR rPPG dataset reader.

    Parameters
    ----------
    hdf5_dir        : local directory with HDF5 shards
    subject_ids     : list of integer subject IDs to include
    split           : 'train' | 'val' | 'test'
    clip_length     : frames per clip (must match preprocessing)
    use_diff_channel: return (T, 2, H, W) if True, (T, 1, H, W) if False
    gcs_cfg         : optional dict for auto-downloading missing shards
    """

    def __init__(
        self,
        hdf5_dir: str,
        subject_ids: List[int],
        split: str = "train",
        clip_length: int = 300,
        use_diff_channel: bool = True,
        gcs_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.hdf5_dir = Path(hdf5_dir)
        self.hdf5_dir.mkdir(parents=True, exist_ok=True)
        self.subject_ids = subject_ids
        self.split = split
        self.clip_length = clip_length
        self.use_diff_channel = use_diff_channel
        self.gcs_cfg = gcs_cfg or {}

        self._index: List[Tuple[Path, str, float]] = []  # (shard, key, loss_weight)
        self._build_index()

        n_dark = sum(1 for _, _, w in self._index if w < 0.9)
        logger.info(
            "MRNIRPDataset [%s]: %d clips (%d dark-weighted) | subjects %s",
            split, len(self._index), n_dark, subject_ids,
        )

    def _build_index(self):
        for sid in self.subject_ids:
            shard = self.hdf5_dir / f"mr_nirp_subj{sid:02d}.h5"
            if not shard.exists():
                shard = self._try_download(sid, shard)
            if shard is None or not shard.exists():
                logger.warning("Shard missing for subject %d — skipping", sid)
                continue

            with h5py.File(shard, "r") as hf:
                if "clips" not in hf:
                    continue
                for key in sorted(hf["clips"].keys(), key=int):
                    grp = hf[f"clips/{key}"]
                    T = grp["frames"].shape[0]
                    if T < self.clip_length:
                        continue
                    # Read loss_weight from quality attrs or meta JSON
                    lw = 1.0
                    if "quality" in grp:
                        lw = float(grp["quality"].attrs.get("loss_weight", 1.0))
                    else:
                        try:
                            meta = json.loads(grp["meta"][()])
                            lw = float(meta.get("loss_weight", 1.0))
                        except Exception:
                            pass
                    self._index.append((shard, key, lw))

    def _try_download(self, sid: int, local: Path) -> Optional[Path]:
        if not self.gcs_cfg:
            return None
        gcs_path = (f"gs://{self.gcs_cfg['bucket_name']}/"
                    f"{self.gcs_cfg['processed_prefix']}/mr_nirp_subj{sid:02d}.h5")
        try:
            return download_file(gcs_path, str(self.hdf5_dir),
                                 project_id=self.gcs_cfg.get("project_id"))
        except Exception as e:
            logger.warning("GCS download failed for subject %d: %s", sid, e)
            return None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        shard, key, loss_weight = self._index[idx]

        with h5py.File(shard, "r") as hf:
            grp = hf[f"clips/{key}"]
            frames = grp["frames"][:].astype(np.float32)   # (T, 2, H, W)
            gt = grp["gt"][:].astype(np.float32)            # (T,)
            meta = json.loads(grp["meta"][()])

        frames, gt = self._ensure_length(frames, gt)

        if self.split == "train":
            frames, gt = self._augment(frames, gt)

        # Channel selection
        if not self.use_diff_channel:
            frames = frames[:, :1, :, :]   # intensity only
        # else: already (T, 2, H, W)

        meta["loss_weight"] = loss_weight
        return torch.from_numpy(frames), torch.from_numpy(gt), meta

    def _ensure_length(self, frames, gt):
        T, C, H, W = frames.shape
        L = self.clip_length
        if T > L:
            start = np.random.randint(0, T - L) if self.split == "train" else (T - L) // 2
            return frames[start:start+L], gt[start:start+L]
        elif T < L:
            pad = L - T
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
            gt = np.concatenate([gt, np.full(pad, gt[-1])])
        return frames, gt

    def _augment(self, frames, gt):
        """
        Safe augmentations for rPPG:
        - Horizontal flip           (BVP amplitude symmetric)
        - Temporal reversal (15%)   (BVP waveform still valid)
        - Gaussian noise  (σ=0.008) (small relative to BVP ~0.01 amplitude)
        - Brightness jitter ±8%     (on intensity channel only)

        NOT applied:
        - Time-stretch  (would shift BPM)
        - Frame drop    (breaks temporal continuity)
        - Colour jitter (single channel)
        """
        # Horizontal flip
        if np.random.rand() < 0.5:
            frames = frames[:, :, :, ::-1].copy()

        # Temporal reversal
        if np.random.rand() < 0.15:
            frames = frames[::-1].copy()
            gt = gt[::-1].copy()
            # Recompute diff channel after reversal
            diff = np.zeros_like(frames[:, 0:1])
            diff[1:] = frames[1:, 0:1] - frames[:-1, 0:1]
            frames[:, 1:2] = diff

        # Additive Gaussian noise (only on intensity channel)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.008, frames[:, 0:1].shape).astype(np.float32)
            frames[:, 0:1] = np.clip(frames[:, 0:1] + noise, 0.0, 1.0)

        # Brightness jitter (intensity channel only)
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.92, 1.08)
            frames[:, 0:1] = np.clip(frames[:, 0:1] * alpha, 0.0, 1.0)

        return frames, gt


# ---------------------------------------------------------------------------
# Quality-aware collator (passes loss_weight to Trainer)
# ---------------------------------------------------------------------------

def collate_with_quality(batch):
    """
    Custom collate_fn that preserves per-clip loss_weight from metadata.
    Returns (frames, gt, loss_weights, meta_list).
    """
    frames_list, gt_list, meta_list = zip(*batch)
    frames = torch.stack(frames_list)
    gt = torch.stack(gt_list)
    loss_weights = torch.tensor(
        [float(m.get("loss_weight", 1.0)) for m in meta_list],
        dtype=torch.float32
    )
    return frames, gt, loss_weights, list(meta_list)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from config."""
    ds_cfg = cfg["dataset"]
    tr_cfg = cfg["training"]
    gcs_cfg = {
        "bucket_name": cfg["gcs"]["bucket_name"],
        "processed_prefix": cfg["gcs"]["processed_prefix"],
        "project_id": cfg["gcs"]["project_id"],
    }
    hdf5_dir = cfg["gcs"]["local_cache_dir"] + "/hdf5"
    clip_length = int(ds_cfg["clip_length_frames"])
    use_diff = bool(cfg["model"].get("use_temporal_diff_channel", True))

    # Support both explicit lists and "auto" (auto resolved by preprocess step)
    def _get_subjects(key):
        val = ds_cfg.get(key, [])
        if val == "auto" or not val:
            # Fall back to all_subjects if split lists not set
            return ds_cfg.get("all_subjects", [])
        return list(val)

    def _make(subjects, split, shuffle):
        ds = MRNIRPDataset(
            hdf5_dir=hdf5_dir,
            subject_ids=subjects,
            split=split,
            clip_length=clip_length,
            use_diff_channel=use_diff,
            gcs_cfg=gcs_cfg,
        )
        return DataLoader(
            ds,
            batch_size=tr_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=tr_cfg["num_workers"],
            pin_memory=tr_cfg["pin_memory"],
            drop_last=(split == "train"),
            collate_fn=collate_with_quality,
            persistent_workers=(tr_cfg["num_workers"] > 0),
        )

    return (
        _make(_get_subjects("train_subjects"), "train", shuffle=True),
        _make(_get_subjects("val_subjects"),   "val",   shuffle=False),
        _make(_get_subjects("test_subjects"),  "test",  shuffle=False),
    )

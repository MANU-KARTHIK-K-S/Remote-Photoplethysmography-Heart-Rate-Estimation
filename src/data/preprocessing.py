"""
src/data/preprocessing.py
──────────────────────────
MR-NIRP-D preprocessing pipeline — PGM frame sequences + pulseOx GT.

Pipeline per session
────────────────────
1. Discover PGMs in <session>/NIR/*.pgm
2. Load and quality-score each frame via PGMSessionLoader
   (dark frames → CLAHE enhancement)
3. Load GT via gt_loader:
     - parse cam0_full_log.txt for per-frame Unix timestamps
     - load pulseOx.mat BVP waveform + timestamps
     - bandpass-filter at pulseOx rate → interpolate to camera timestamps
4. Slice into overlapping 10-s clips (T=300 frames)
5. Discard clips with >40% dark frames; remaining dark clips → loss_weight=0.3
6. Write HDF5 shards: /clips/{idx}/{frames(T,2,H,W), gt(T,), quality, meta}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from src.data.pgm_loader import (
    ClipQuality, FrameQuality,
    PGMSessionLoader, compute_clip_quality,
)
from src.data.gt_loader import load_session_gt

logger = logging.getLogger(__name__)


def add_temporal_diff_channel(frames: np.ndarray) -> np.ndarray:
    """
    Input:  (T, H, W)   float32
    Output: (T, 2, H, W) float32  — [intensity, I_t - I_{t-1}]
    Ref: PhysFormer (Yu et al. CVPR 2022).
    """
    T, H, W = frames.shape
    diff = np.zeros_like(frames)
    diff[1:] = frames[1:] - frames[:-1]
    return np.stack([frames, diff], axis=1).astype(np.float32)


def slice_into_clips(
    frames: np.ndarray,
    gt: np.ndarray,
    frame_qualities: List[FrameQuality],
    clip_length: int,
    overlap_ratio: float,
    fps: float,
    min_valid_frame_ratio: float,
    dark_clip_loss_weight: float,
) -> List[Dict]:
    T = frames.shape[0]
    step = max(1, int(clip_length * (1.0 - overlap_ratio)))
    clips = []
    for start in range(0, T - clip_length + 1, step):
        end = start + clip_length
        clip_q = compute_clip_quality(frame_qualities[start:end], dark_clip_loss_weight)
        # Keep dark clips (they've been session-level stretched, BVP is preserved).
        # Only discard clips where ALL frames are dead (loss_weight at minimum)
        # AND quality is near zero (truly uninformative).
        if clip_q.loss_weight <= dark_clip_loss_weight and clip_q.mean_quality < 0.15:
            continue
        clips.append({
            "frames": add_temporal_diff_channel(frames[start:end]),
            "gt":     gt[start:end],
            "quality": clip_q,
            "start_frame": start,
        })
    return clips


def append_clips_to_hdf5(
    clips: List[Dict],
    shard_path: Path,
    subject_id: int,
    session_name: str,
    fps: float,
):
    mode = "a" if shard_path.exists() else "w"
    with h5py.File(shard_path, mode) as hf:
        base = len(hf.require_group("clips"))
        for i, clip in enumerate(clips):
            grp = hf["clips"].create_group(str(base + i))
            grp.create_dataset("frames", data=clip["frames"],
                               compression="gzip", compression_opts=4)
            grp.create_dataset("gt", data=clip["gt"])
            q: ClipQuality = clip["quality"]
            qg = grp.create_group("quality")
            for attr, val in [
                ("dark_frame_ratio", q.dark_frame_ratio),
                ("mean_quality",     q.mean_quality),
                ("has_dark_frames",  int(q.has_dark_frames)),
                ("loss_weight",      q.loss_weight),
            ]:
                qg.attrs[attr] = val
            meta = {
                "subject": int(subject_id), "session": session_name,
                "start_frame": int(clip["start_frame"]), "fps": float(fps),
                "loss_weight": float(q.loss_weight),
                "dark_frame_ratio": float(q.dark_frame_ratio),
            }
            grp.create_dataset("meta", data=json.dumps(meta))
    logger.debug("Appended %d clips → %s", len(clips), shard_path.name)


def process_session(
    session_dir: Path,
    subject_id: int,
    out_shard: Path,
    cfg: Dict,
) -> int:
    """Process one session → HDF5 clips. Returns number of clips written."""
    ds  = cfg["dataset"]
    pre = cfg["preprocessing"]
    fps = float(ds["fps"])

    nir_dir = session_dir / "NIR"
    if not nir_dir.exists() or not list(nir_dir.glob("*.pgm")):
        nir_dir = session_dir
        if not list(nir_dir.glob("*.pgm")):
            logger.warning("No PGMs in %s — skipping", session_dir)
            return 0

    # ── 1. Load NIR frames ─────────────────────────────────────────────────
    loader = PGMSessionLoader(
        nir_dir=nir_dir,
        face_roi_size=int(ds["face_roi_size"]),
        dark_threshold=float(pre.get("dark_frame_threshold", 0.04)),
        dark_enhancement=str(pre.get("dark_enhancement", "clahe")),
        clahe_clip_limit=float(pre.get("clahe_clip_limit", 3.0)),
        clahe_tile_size=tuple(pre.get("clahe_tile_size", [8, 8])),
        normalize_method=str(pre.get("normalize_method", "percentile_clip")),
        percentile_lo=float(pre.get("percentile_lo", 1.0)),
        percentile_hi=float(pre.get("percentile_hi", 99.0)),
        face_detector=str(pre.get("face_detector", "mediapipe")),
        face_conf_threshold=float(pre.get("face_conf_threshold", 0.4)),
    )
    frames, frame_qualities = loader.load()
    if frames is None:
        return 0
    T = frames.shape[0]

    # ── 2. Load GT (pulseOx.mat + camera timestamps) ───────────────────────
    synced = load_session_gt(session_dir, n_frames=T, fps_cam=fps)
    if synced is None:
        logger.error("GT loading failed for %s — skipping", session_dir)
        return 0
    gt = synced.bvp_cam

    # Align lengths
    n_use = min(T, len(gt))
    frames = frames[:n_use]
    gt     = gt[:n_use]
    frame_qualities = frame_qualities[:n_use]

    # ── 3. Slice clips ─────────────────────────────────────────────────────
    clips = slice_into_clips(
        frames=frames, gt=gt,
        frame_qualities=frame_qualities,
        clip_length=int(ds["clip_length_frames"]),
        overlap_ratio=float(ds["overlap_ratio"]),
        fps=fps,
        min_valid_frame_ratio=float(pre.get("min_valid_frame_ratio", 0.6)),
        dark_clip_loss_weight=float(cfg["training"].get("dark_clip_loss_weight", 0.3)),
    )

    if not clips:
        logger.warning("No valid clips for %s", session_dir)
        return 0

    dark_n = sum(1 for c in clips if c["quality"].has_dark_frames)
    logger.info("Subject%d | %s → %d clips (%d dark-weighted, synced_valid=%d%%)",
                subject_id, session_dir.name, len(clips), dark_n,
                int(100*synced.n_valid_frames/max(T,1)))

    # ── 4. Write HDF5 ──────────────────────────────────────────────────────
    append_clips_to_hdf5(clips, out_shard, subject_id, session_dir.name, fps)
    return len(clips)


def discover_sessions(subject_dir: Path) -> List[Path]:
    sessions = []
    for d in sorted(subject_dir.iterdir()):
        if not d.is_dir(): continue
        nir = d / "NIR"
        if (nir.is_dir() and list(nir.glob("*.pgm"))) or list(d.glob("*.pgm")):
            sessions.append(d)
    return sessions


def find_subject_dirs(dataset_root: Path, subject_ids: List[int]) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for d in sorted(dataset_root.iterdir()):
        if not d.is_dir(): continue
        m = re.search(r"(?:subject|subj|p|sub)_?0*(\d+)", d.name, re.IGNORECASE)
        if m:
            sid = int(m.group(1))
            if sid in subject_ids:
                mapping[sid] = d
    missing = set(subject_ids) - set(mapping.keys())
    if missing:
        logger.warning("Subject dirs not found: %s", sorted(missing))
    return mapping

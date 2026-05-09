#!/usr/bin/env python3
"""
scripts/preprocess_dataset.py
───────────────────────────────
Preprocessing for MR-NIRP-D from GCS.

REAL BUCKET LAYOUT (docs-ingest-bucket):
  subject14_garage_still_975/
    PulseOX_subject14_garage_still_975/
      PulseOX/
        pulseOx.mat           ← 125 Hz BVP reference
        cam0_full_log.txt     ← per-frame Unix timestamps
    subject14_garage_still_975/
      000000.pgm … NNNNNN.pgm   ← NIR frames

RUN ON T4 VM:
  cd ~/hr_nir_estimation
  python scripts/preprocess_dataset.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse, logging, sys, tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess")


# ── GCS helpers ─────────────────────────────────────────────────────────────

def gcs_download_bytes(bucket_name, blob_name, project_id=None):
    from src.utils.gcs_utils import get_gcs_client
    return get_gcs_client(project_id).bucket(bucket_name).blob(blob_name).download_as_bytes()


def list_pgm_blobs(bucket_name, prefix, project_id=None):
    """Return sorted blob names for all .pgm files under prefix."""
    import re
    from src.utils.gcs_utils import get_gcs_client
    blobs = get_gcs_client(project_id).list_blobs(bucket_name, prefix=prefix.rstrip("/")+"/")
    pgms = sorted(
        [b.name for b in blobs if b.name.lower().endswith(".pgm")],
        key=lambda x: int(re.search(r"\d+", Path(x).stem).group())
                      if re.search(r"\d+", Path(x).stem) else 0
    )
    return pgms


def stream_pgm_frames(bucket_name, blob_names, project_id, batch=50):
    """Download PGM blobs in batches, yield float32 (H,W) arrays."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch(blob_name):
        raw = gcs_download_bytes(bucket_name, blob_name, project_id)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if img is None:
            with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tf:
                tf.write(raw); tmp = tf.name
            from src.data.pgm_loader import read_pgm_frame
            img_f = read_pgm_frame(tmp); Path(tmp).unlink(missing_ok=True)
            return img_f
        if img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        return img.astype(np.float32) / 255.0

    for i in range(0, len(blob_names), batch):
        chunk = blob_names[i:i+batch]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(fetch, b): j for j, b in enumerate(chunk)}
            buf = {}
            for fut in as_completed(futs):
                buf[futs[fut]] = fut.result()
        for j in range(len(chunk)):
            yield buf[j]


# ── Session processor ────────────────────────────────────────────────────────

def process_one_session(session, out_shard, cfg, bucket, proj):
    """
    Memory-efficient streaming batch processing with full quality logic.
    Restores use of MIN_CONTRAST, DEAD_THRESHOLD, and normalisation percentiles.
    """
    import cv2, gc
    from src.data.pgm_loader import (
        SessionNormaliser, _FaceDetectorAdapter, _interpolate_bboxes,
        FrameQuality, MIN_CONTRAST, DARK_THRESHOLD, DEAD_THRESHOLD,
    )
    from src.data.gt_loader import (
        load_pulseox_mat, load_camera_timestamps, sync_pulseox_to_camera,
    )
    from src.data.preprocessing import append_clips_to_hdf5, add_temporal_diff_channel

    ds, pre, tr = cfg["dataset"], cfg["preprocessing"], cfg["training"]
    fps      = float(ds["fps"])
    roi_sz   = int(ds["face_roi_size"])
    clip_len = int(ds["clip_length_frames"])
    overlap  = float(ds["overlap_ratio"])
    dark_lw  = float(tr.get("dark_clip_loss_weight", 0.3))
    min_vfr  = float(pre.get("min_valid_frame_ratio", 0.4))

    logger.info("━"*55)
    logger.info("Subject %d | %s", session.subject_id, session.session_label)

    # 1. Load GT & Timestamps (Small files)
    mat_blob = session.pulseox_mat_path.replace(f"gs://{bucket}/", "")
    mat_bytes = gcs_download_bytes(bucket, mat_blob, proj)
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tf:
        tf.write(mat_bytes); mat_tmp = tf.name
    pulseox = load_pulseox_mat(mat_tmp)
    Path(mat_tmp).unlink(missing_ok=True)
    
    log_blob = session.cam_log_path.replace(f"gs://{bucket}/", "")
    log_text = gcs_download_bytes(bucket, log_blob, proj).decode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tf:
        tf.write(log_text); log_tmp = tf.name
    t_cam = load_camera_timestamps(log_tmp)
    Path(log_tmp).unlink(missing_ok=True)
    
    pgm_prefix = session.pgm_prefix.replace(f"gs://{bucket}/", "")
    pgm_blobs = list_pgm_blobs(bucket, pgm_prefix, proj)
    n_frames = min(len(pgm_blobs), len(t_cam))

    # 2. Sync Ground Truth
    fps_cam = (len(t_cam)-1)/(t_cam[-1]-t_cam[0]) if len(t_cam)>1 else fps
    synced = sync_pulseox_to_camera(pulseox, t_cam[:n_frames], fps_cam=fps_cam)
    gt_arr = synced.bvp_cam[:n_frames]

    # 3. Batch Configuration
    step = max(1, int(clip_len * (1.0 - overlap)))
    batch_size = 600 
    detector = _FaceDetectorAdapter("haar", 0.4)
    normaliser = None
    rng = np.random.default_rng(42)
    total_clips = 0
    
    # 4. Iterative Batch Processing
    for start_idx in range(0, n_frames - clip_len + 1, batch_size):
        # We fetch enough frames to cover the clips starting in this batch
        end_idx = min(start_idx + batch_size + clip_len, n_frames)
        current_blobs = pgm_blobs[start_idx:end_idx]
        
        logger.info(f"  Streaming frames {start_idx} to {end_idx}...")
        raw_chunk = list(stream_pgm_frames(bucket, current_blobs, proj, batch=100))
        
        # Initialize normaliser with original config
        if normaliser is None:
            normaliser = SessionNormaliser(
                [f for f in raw_chunk if f is not None],
                dark_thr=float(pre.get("dark_frame_threshold", DARK_THRESHOLD)),
                dead_thr=float(pre.get("dead_frame_threshold", DEAD_THRESHOLD)),
                p_lo=float(pre.get("percentile_lo", 1.0)),
                p_hi=float(pre.get("percentile_hi", 99.0)),
            )
        
        # Process frames and collect quality metrics
        processed_chunk = []
        chunk_qualities = []
        for f in raw_chunk:
            if f is None:
                processed_chunk.append(np.zeros((roi_sz, roi_sz), np.float32))
                chunk_qualities.append(FrameQuality(is_flat=True))
                continue

            normed, is_dark = normaliser.apply(f, rng)
            # Re-implementing original face-detect/ROI logic
            boost = np.clip(f * 20.0, 0, 1) if f.mean() < 0.05 else f
            bbox = detector.detect(boost)
            
            H, W = normed.shape[:2]
            if bbox is None:
                py, px = int(H*0.15), int(W*0.15)
                roi = normed[py:H-py, px:W-px]
            else:
                x, y, bw, bh = bbox
                px2, py2 = int(bw*0.1), int(bh*0.1)
                roi = normed[max(0,y-py2):min(H,y+bh+py2),
                             max(0,x-px2):min(W,x+bw+px2)]
            
            processed_chunk.append(cv2.resize(roi, (roi_sz, roi_sz), interpolation=cv2.INTER_AREA))
            
            # Utilizing MIN_CONTRAST and original quality mapping
            chunk_qualities.append(FrameQuality(
                mean=float(normed.mean()), std=float(normed.std()),
                is_dark=is_dark, is_flat=float(normed.std()) < MIN_CONTRAST,
                enhanced=is_dark,
            ))

        chunk_frames = np.stack(processed_chunk).astype(np.float32)
        
        # 5. Slice and write batch clips
        batch_clips = []
        for local_start in range(0, len(chunk_frames) - clip_len + 1, step):
            global_start = start_idx + local_start
            if global_start < start_idx and start_idx != 0: continue
            if global_start + clip_len > n_frames: break

            l_end = local_start + clip_len
            from src.data.pgm_loader import compute_clip_quality
            clip_q = compute_clip_quality(chunk_qualities[local_start:l_end], dark_lw)
            
            if clip_q.loss_weight <= dark_lw and clip_q.mean_quality < 0.15:
                continue

            batch_clips.append({
                "frames": add_temporal_diff_channel(chunk_frames[local_start:l_end]),
                "gt":     gt_arr[global_start:global_start+clip_len],
                "quality": clip_q,
                "start_frame": global_start,
            })

        if batch_clips:
            append_clips_to_hdf5(batch_clips, out_shard, session.subject_id, session.session_label, fps)
            total_clips += len(batch_clips)

        # 6. Explicit Memory Clearance
        del raw_chunk, chunk_frames, processed_chunk, batch_clips
        gc.collect() 

    return total_clips


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gcs     = cfg["gcs"]
    bucket  = gcs["bucket_name"]
    proj    = gcs.get("project_id")
    out_dir = Path(gcs["local_cache_dir"]) / "hdf5"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.data.gcs_discovery import discover_sessions, split_sessions, print_discovery_summary

    sessions = discover_sessions(bucket, gcs.get("raw_prefix",""), proj)
    if not sessions:
        logger.error("No sessions found in gs://%s", bucket); sys.exit(1)

    ratio = tuple(cfg["dataset"].get("subject_split_ratio", [0.6, 0.2, 0.2]))
    sessions = split_sessions(sessions, ratio=ratio)
    if args.subjects:
        sessions = [s for s in sessions if s.subject_id in args.subjects]

    print_discovery_summary(sessions)

    total = 0
    for sess in sessions:
        shard = out_dir / f"mr_nirp_subj{sess.subject_id:02d}.h5"
        n = process_one_session(sess, shard, cfg, bucket, proj)
        total += n

    logger.info("Total clips: %d", total)

    if not args.no_upload:
        from src.utils.gcs_utils import upload_file
        for shard in sorted(out_dir.glob("mr_nirp_subj*.h5")):
            upload_file(shard, bucket, gcs["processed_prefix"], project_id=proj)
            logger.info("Uploaded %s", shard.name)

    # Report
    import h5py
    print("\n" + "═"*55)
    print("  Preprocessing Complete")
    print("═"*55)
    for shard in sorted(out_dir.glob("mr_nirp_subj*.h5")):
        with h5py.File(shard,"r") as hf:
            clips = hf.get("clips",{})
            n = len(clips)
            dark = sum(1 for k in clips
                       if "quality" in hf[f"clips/{k}"]
                       and hf[f"clips/{k}/quality"].attrs.get("has_dark_frames",0))
        print(f"  {shard.name}: {n:4d} clips  ({dark} dark-weighted)")
    print(f"  TOTAL: {total} clips")
    print("═"*55)
    print("\n  Next: python scripts/train.py --config configs/config.yaml\n")


if __name__ == "__main__":
    main()

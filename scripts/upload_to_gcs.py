#!/usr/bin/env python3
"""
scripts/upload_to_gcs.py
──────────────────────────
Upload the locally-downloaded MR-NIRP-D PGM dataset to your GCS bucket.

Run this ONCE after downloading the dataset from Google Drive.
The raw PGMs are uploaded preserving the Subject{N}/session/NIR/ structure.
After this, use preprocess_dataset.py --from-gcs to process them.

Usage
─────
    # Upload everything (all subjects):
    python scripts/upload_to_gcs.py \\
        --config configs/config.yaml \\
        --dataset-root /path/to/downloaded/mr-nirp-d

    # Upload specific subjects only:
    python scripts/upload_to_gcs.py \\
        --config configs/config.yaml \\
        --dataset-root /path/to/mr-nirp-d \\
        --subjects 1 2

    # Dry run (list files without uploading):
    python scripts/upload_to_gcs.py \\
        --config configs/config.yaml \\
        --dataset-root /path/to/mr-nirp-d \\
        --dry-run

What gets uploaded
──────────────────
  *.pgm   — NIR frame images
  *.csv   — ground-truth BVP signals
  *.txt   — HR ground truth files
  *.mat   — MATLAB GT files (if any)

Destination in GCS
──────────────────
  gs://{bucket}/{raw_prefix}/Subject1/resting_indoor/NIR/frame_000000.pgm
  gs://{bucket}/{raw_prefix}/Subject1/resting_indoor/gt_bvp.csv
  ...

Performance notes
─────────────────
~28,000 PGM files × ~75 KB each ≈ 2 GB total.
Upload uses 32 parallel threads → typically ~5 min on a 100 Mbps link.
Resume-safe: existing GCS objects are skipped unless --force is set.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.gcs_utils import get_gcs_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("upload")

UPLOAD_EXTENSIONS = {".pgm", ".PGM", ".csv", ".CSV", ".txt", ".mat"}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_dataset_files(
    dataset_root: Path,
    subjects: Optional[List[int]] = None,
) -> List[Tuple[Path, str]]:
    """
    Walk dataset_root and return (local_path, subject_name) pairs
    for all files that should be uploaded.

    Filters by subject IDs if provided.
    """
    import re

    files = []
    for item in sorted(dataset_root.iterdir()):
        if not item.is_dir():
            continue
        m = re.search(r"(?:subject|subj|p|sub)_?0*(\d+)", item.name, re.IGNORECASE)
        if m is None:
            continue
        sid = int(m.group(1))
        if subjects and sid not in subjects:
            continue
        # Walk all files under this subject dir
        for fpath in sorted(item.rglob("*")):
            if fpath.is_file() and fpath.suffix.lower() in {e.lower() for e in UPLOAD_EXTENSIONS}:
                files.append((fpath, item.name))

    return files


# ---------------------------------------------------------------------------
# Existence check (skip already-uploaded objects)
# ---------------------------------------------------------------------------

def blob_exists(bucket, blob_name: str) -> bool:
    """Return True if the blob already exists in GCS."""
    blob = bucket.blob(blob_name)
    return blob.exists()


# ---------------------------------------------------------------------------
# Upload worker
# ---------------------------------------------------------------------------

def upload_one(
    local_path: Path,
    dataset_root: Path,
    bucket,
    raw_prefix: str,
    force: bool,
    dry_run: bool,
) -> Tuple[str, bool, str]:
    """
    Upload a single file.
    Returns (blob_name, success, status_message).
    """
    rel = local_path.relative_to(dataset_root)
    blob_name = f"{raw_prefix.rstrip('/')}/{rel.as_posix()}"

    if dry_run:
        return blob_name, True, "dry-run"

    if not force and blob_exists(bucket, blob_name):
        return blob_name, True, "skipped (exists)"

    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        return blob_name, True, "uploaded"
    except Exception as e:
        return blob_name, False, f"FAILED: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload MR-NIRP-D PGM files to GCS bucket"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root", required=True,
                        help="Local path containing Subject1/, Subject2/ …")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel upload threads (default: 32)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload even if blob already exists in GCS")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gcs_cfg = cfg["gcs"]
    project_id = gcs_cfg.get("project_id")
    bucket_name = gcs_cfg["bucket_name"]
    raw_prefix = gcs_cfg["raw_prefix"]

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        logger.error("dataset-root does not exist: %s", dataset_root)
        sys.exit(1)

    # Discover files
    subjects = args.subjects or cfg["dataset"]["all_subjects"]
    logger.info("Scanning %s for subjects %s ...", dataset_root, subjects)
    files = find_dataset_files(dataset_root, subjects)

    if not files:
        logger.error("No files found. Check --dataset-root and file extensions.")
        sys.exit(1)

    pgm_count = sum(1 for f, _ in files if f.suffix.lower() == ".pgm")
    gt_count = sum(1 for f, _ in files if f.suffix.lower() in {".csv", ".txt", ".mat"})
    total_size_mb = sum(f.stat().st_size for f, _ in files) / 1e6

    print(f"\n{'═'*60}")
    print(f"  Upload plan: {len(files):,} files → gs://{bucket_name}/{raw_prefix}")
    print(f"  PGM frames : {pgm_count:,}")
    print(f"  GT files   : {gt_count:,}")
    print(f"  Total size : {total_size_mb:,.0f} MB")
    if args.dry_run:
        print(f"  Mode       : DRY RUN (no files will be uploaded)")
    print(f"{'═'*60}\n")

    if args.dry_run:
        for f, subj in files[:10]:
            print(f"  [DRY-RUN] Would upload: {f.relative_to(dataset_root)}")
        if len(files) > 10:
            print(f"  ... and {len(files)-10} more files")
        return

    # Get GCS bucket
    client = get_gcs_client(project_id)
    bucket = client.bucket(bucket_name)

    # Parallel upload
    succeeded, failed, skipped = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                upload_one, local_path, dataset_root, bucket,
                raw_prefix, args.force, False
            ): local_path
            for local_path, _ in files
        }
        with tqdm(total=len(futures), unit="file", desc="Uploading") as pbar:
            for fut in as_completed(futures):
                blob_name, ok, status = fut.result()
                if ok:
                    if "skipped" in status:
                        skipped += 1
                    else:
                        succeeded += 1
                else:
                    failed += 1
                    logger.warning("%s", status)
                pbar.update(1)
                pbar.set_postfix(ok=succeeded, skip=skipped, fail=failed)

    print(f"\n{'═'*60}")
    print(f"  Upload complete")
    print(f"  Uploaded : {succeeded:,} files")
    print(f"  Skipped  : {skipped:,} (already exist in GCS)")
    print(f"  Failed   : {failed:,}")
    if failed > 0:
        print(f"  ⚠ Re-run with --force to retry failed uploads")
    print(f"{'═'*60}")
    print(f"\n  Next step:")
    print(f"  python scripts/preprocess_dataset.py --config configs/config.yaml --from-gcs\n")


if __name__ == "__main__":
    main()

"""
src/data/gcs_discovery.py
──────────────────────────
Discovers the real MR-NIRP-D subject/session structure from the GCS bucket.

REAL bucket layout (as uploaded):
  gs://docs-ingest-bucket/
    subject14_garage_still_975/
      PulseOX_subject14_garage_still_975/
        PulseOX/
          pulseOx.mat
          cam0_full_log.txt
          cam0_partial_log.txt
      subject14_garage_still_975/          ← NIR PGM frames (OPTION A)
        000000.pgm … NNNNNN.pgm
      NIR/                                 ← NIR PGM frames (OPTION B)
        000000.pgm …
    subject3_office_walking_234/
      ...

Subject naming: subject{ID}_{scene}_{condition}_{session_num}
  e.g. subject14_garage_still_975 → ID=14, session=garage_still_975

This module:
  1. Lists all top-level subject prefixes from the bucket
  2. For each, finds:
     - pulseOx.mat path
     - cam0_full_log.txt path
     - NIR PGM frame prefix (tries multiple layouts)
  3. Returns a list of SessionRecord objects ready for preprocessing
  4. Performs subject-independent train/val/test split
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session record
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    """All GCS paths and metadata for one MR-NIRP-D session."""
    subject_id: int
    subject_folder: str          # e.g. "subject14_garage_still_975"
    session_label: str           # e.g. "garage_still_975"
    # GCS paths (gs://bucket/...)
    pgm_prefix: str              # prefix where .pgm files live
    pulseox_mat_path: str        # gs:// path to pulseOx.mat
    cam_log_path: str            # gs:// path to cam0_full_log.txt
    n_pgm_files: int = 0
    split: str = "train"         # train / val / test (assigned later)


# ---------------------------------------------------------------------------
# Bucket scanner
# ---------------------------------------------------------------------------

def discover_sessions(
    bucket_name: str,
    raw_prefix: str = "",
    project_id: Optional[str] = None,
) -> List[SessionRecord]:
    """
    Scan the GCS bucket and return all valid SessionRecord objects.

    A valid session must have:
      - At least one .pgm file (NIR frames)
      - pulseOx.mat
      - cam0_full_log.txt (or cam0_partial_log.txt)
    """
    from src.utils.gcs_utils import get_gcs_client

    client = get_gcs_client(project_id)
    bucket = client.bucket(bucket_name)

    logger.info("Scanning gs://%s/%s ...", bucket_name, raw_prefix or "(root)")

    # ── List all blobs once (more efficient than per-prefix listing) ────────
    prefix = raw_prefix.rstrip("/") + "/" if raw_prefix else ""
    all_blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    logger.info("Found %d total blobs in bucket", len(all_blobs))

    # ── Build a map: subject_folder → {blob_names} ──────────────────────────
    folder_blobs: Dict[str, List[str]] = {}
    subj_pat = re.compile(r"^(?:[^/]+/)?subject(\d+)_\S+")

    for blob in all_blobs:
        name = blob.name
        parts = name.split("/")
        # Subject folder is first (or second if raw_prefix) non-empty part
        offset = 1 if raw_prefix else 0
        if len(parts) <= offset:
            continue
        subj_folder = parts[offset]
        if not subj_folder.lower().startswith("subject"):
            continue
        folder_blobs.setdefault(subj_folder, []).append(name)

    logger.info("Found %d subject folders", len(folder_blobs))

    # ── For each subject folder, find the three required paths ───────────────
    sessions: List[SessionRecord] = []

    for subj_folder in sorted(folder_blobs.keys()):
        blobs = folder_blobs[subj_folder]

        # Parse subject ID from folder name
        m = re.match(r"subject(\d+)_(.+)", subj_folder, re.IGNORECASE)
        if not m:
            logger.debug("Skipping non-subject folder: %s", subj_folder)
            continue
        sid = int(m.group(1))
        session_label = m.group(2)

        bucket_root = f"gs://{bucket_name}"

        # ── Find pulseOx.mat ─────────────────────────────────────────────────
        pulseox = _find_blob(blobs, "pulseOx.mat", bucket_root)
        if pulseox is None:
            pulseox = _find_blob(blobs, "pulseox.mat", bucket_root)
        if pulseox is None:
            logger.warning("No pulseOx.mat for %s — skipping", subj_folder)
            continue

        # ── Find camera timestamp log ────────────────────────────────────────
        cam_log = _find_blob(blobs, "cam0_full_log.txt", bucket_root)
        if cam_log is None:
            cam_log = _find_blob(blobs, "cam0_partial_log.txt", bucket_root)
        if cam_log is None:
            logger.warning("No cam log for %s — skipping", subj_folder)
            continue

        # ── Find NIR PGM frames prefix ────────────────────────────────────────
        pgm_prefix = _find_pgm_prefix(blobs, subj_folder, bucket_root, bucket_name)
        if pgm_prefix is None:
            logger.warning("No PGM frames found for %s — skipping", subj_folder)
            continue

        # Count PGMs
        n_pgm = sum(1 for b in blobs if b.endswith(".pgm") or b.endswith(".PGM"))

        sessions.append(SessionRecord(
            subject_id=sid,
            subject_folder=subj_folder,
            session_label=session_label,
            pgm_prefix=pgm_prefix,
            pulseox_mat_path=pulseox,
            cam_log_path=cam_log,
            n_pgm_files=n_pgm,
        ))
        logger.info(
            "  ✓ Subject %2d | %-35s | %5d PGMs",
            sid, subj_folder, n_pgm,
        )

    logger.info("Total valid sessions: %d", len(sessions))
    return sessions


def _find_blob(blobs: List[str], filename: str, bucket_root: str) -> Optional[str]:
    """Find a blob by filename (case-insensitive) and return gs:// path."""
    fname_lower = filename.lower()
    for b in blobs:
        if Path(b).name.lower() == fname_lower:
            return f"{bucket_root}/{b}"
    return None


def _find_pgm_prefix(
    blobs: List[str],
    subj_folder: str,
    bucket_root: str,
    bucket_name: str,
) -> Optional[str]:
    """
    Find the GCS prefix where .pgm files live.

    CONFIRMED MR-NIRP-D layout (from bucket inspection):
      subject14_garage_still_975/NIR/000000.pgm  ← NIR/ subfolder

    Also handles fallbacks:
      subject14_garage_still_975/subject14_garage_still_975/000000.pgm
      subject14_garage_still_975/000000.pgm  (flat)
    """
    pgm_blobs = [b for b in blobs if b.lower().endswith(".pgm")]
    if not pgm_blobs:
        return None

    # Get unique directory prefixes containing PGMs
    prefixes = set()
    for b in pgm_blobs:
        prefixes.add("/".join(b.split("/")[:-1]))

    # Priority 1: NIR/ subfolder (confirmed real layout)
    for prefix in sorted(prefixes):
        if prefix.rstrip("/").endswith("/NIR") or prefix.rstrip("/").endswith("/nir"):
            return f"{bucket_root}/{prefix}"

    # Priority 2: subfolder matching subject name
    for prefix in sorted(prefixes, key=len, reverse=True):
        if subj_folder.lower() in prefix.lower():
            return f"{bucket_root}/{prefix}"

    # Fallback: any prefix with PGMs
    return f"{bucket_root}/{sorted(prefixes, key=len, reverse=True)[0]}"


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_sessions(
    sessions: List[SessionRecord],
    ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> List[SessionRecord]:
    """
    Subject-independent split: all sessions of a subject go to same split.
    Ensures no data leakage across splits.

    With ~4 subjects and ratio (0.6, 0.2, 0.2):
      2 subjects → train, 1 → val, 1 → test
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    # Get unique subject IDs
    unique_ids = sorted(set(s.subject_id for s in sessions))
    n = len(unique_ids)

    shuffled = rng.permutation(unique_ids).tolist()
    n_train = max(1, int(n * ratio[0]))
    n_val   = max(1, int(n * ratio[1]))
    # Remainder → test

    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train: n_train + n_val])
    test_ids  = set(shuffled[n_train + n_val:])

    # Ensure at least 1 subject in each split
    if not test_ids and len(val_ids) > 1:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        test_ids.add(moved)

    logger.info("Subject split:  train=%s  val=%s  test=%s",
                sorted(train_ids), sorted(val_ids), sorted(test_ids))

    for s in sessions:
        if s.subject_id in train_ids:
            s.split = "train"
        elif s.subject_id in val_ids:
            s.split = "val"
        else:
            s.split = "test"

    return sessions


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_discovery_summary(sessions: List[SessionRecord]):
    print("\n" + "═" * 75)
    print("  MR-NIRP-D Session Discovery Summary")
    print("═" * 75)
    print(f"  {'Subject':<6} {'Session Label':<30} {'PGMs':>6}  {'Split':<6}  GT")
    print("─" * 75)
    for s in sorted(sessions, key=lambda x: x.subject_id):
        print(f"  {s.subject_id:<6} {s.session_label:<30} {s.n_pgm_files:>6}  "
              f"{s.split:<6}  ✓")
    print("─" * 75)
    splits = {"train": 0, "val": 0, "test": 0}
    for s in sessions:
        splits[s.split] += 1
    print(f"  Total: {len(sessions)} sessions | "
          f"train={splits['train']} val={splits['val']} test={splits['test']}")
    print("═" * 75 + "\n")

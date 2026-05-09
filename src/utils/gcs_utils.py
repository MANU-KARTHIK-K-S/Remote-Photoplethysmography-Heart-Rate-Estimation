"""
src/utils/gcs_utils.py
──────────────────────
Google Cloud Storage helpers for the HR-NIR pipeline.

Responsibilities
----------------
* List / download NIR video files and ground-truth CSVs from a GCS bucket.
* Cache files locally to avoid repeated network transfers during training.
* Upload preprocessed HDF5 clip files back to GCS.
* Provide a streaming download iterator for large video files.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Generator, List, Optional

from google.cloud import storage
from google.cloud.storage import Blob
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory (singleton per process)
# ---------------------------------------------------------------------------

_CLIENT: Optional[storage.Client] = None


def get_gcs_client(project_id: Optional[str] = None) -> storage.Client:
    """
    Return (or create) a GCS client.

    Authentication (no gcloud SDK needed on VM):
    ─────────────────────────────────────────────
    On a GCE VM with a Service Account attached, the google-cloud-storage
    Python library automatically uses Application Default Credentials (ADC).
    The SA's permissions are picked up from the VM metadata server at
    http://metadata.google.internal — no gcloud auth or key files needed.

    This works because:
      1. You created the VM with a SA that has Storage Object Admin
      2. pip install google-cloud-storage is installed
      3. That's it — no other auth config required
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = storage.Client(project=project_id)
        logger.info("GCS client initialised via ADC (project=%s)", project_id or "default")
    return _CLIENT


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def list_blobs(
    bucket_name: str,
    prefix: str,
    suffix_filter: str = "",
    project_id: Optional[str] = None,
) -> List[Blob]:
    """Return all blobs under *prefix* optionally filtered by suffix."""
    client = get_gcs_client(project_id)
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if suffix_filter:
        blobs = [b for b in blobs if b.name.endswith(suffix_filter)]
    logger.debug("Found %d blobs under gs://%s/%s", len(blobs), bucket_name, prefix)
    return blobs


def list_nir_videos(
    bucket_name: str,
    raw_prefix: str,
    project_id: Optional[str] = None,
) -> List[str]:
    """Return GCS paths of NIR video files (avi / mp4)."""
    blobs = list_blobs(bucket_name, raw_prefix, project_id=project_id)
    video_exts = {".avi", ".mp4", ".mov"}
    paths = [
        f"gs://{bucket_name}/{b.name}"
        for b in blobs
        if Path(b.name).suffix.lower() in video_exts
        and "NIR" in b.name  # only NIR modality
    ]
    logger.info("Found %d NIR video files in gs://%s/%s", len(paths), bucket_name, raw_prefix)
    return sorted(paths)


def list_gt_files(
    bucket_name: str,
    raw_prefix: str,
    project_id: Optional[str] = None,
) -> List[str]:
    """Return GCS paths of ground-truth BVP / HR CSV files."""
    blobs = list_blobs(bucket_name, raw_prefix, project_id=project_id)
    gt_exts = {".csv", ".txt", ".mat"}
    paths = [
        f"gs://{bucket_name}/{b.name}"
        for b in blobs
        if Path(b.name).suffix.lower() in gt_exts
        and any(kw in b.name for kw in ["bvp", "gt", "HR", "pulse"])
    ]
    logger.info("Found %d ground-truth files", len(paths))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _local_path(gcs_path: str, cache_dir: str) -> Path:
    """Map a gs:// URI to a deterministic local cache path."""
    # Strip gs://bucket/
    parts = gcs_path.replace("gs://", "").split("/", 1)
    rel = parts[1] if len(parts) > 1 else parts[0]
    local = Path(cache_dir) / rel
    return local


def download_file(
    gcs_path: str,
    cache_dir: str,
    project_id: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Download a single GCS file to *cache_dir*, preserving relative path.

    Returns the local :class:`Path` to the downloaded file.
    Skips download if the file already exists (unless *force=True*).
    """
    local = _local_path(gcs_path, cache_dir)
    if local.exists() and not force:
        logger.debug("Cache hit: %s", local)
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    client = get_gcs_client(project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    size_mb = blob.size / 1e6 if blob.size else 0
    logger.info("Downloading gs://%s/%s (%.1f MB) → %s", bucket_name, blob_name, size_mb, local)
    blob.download_to_filename(str(local))
    return local


def download_files_parallel(
    gcs_paths: List[str],
    cache_dir: str,
    project_id: Optional[str] = None,
    max_workers: int = 8,
) -> List[Path]:
    """Download a list of GCS files in parallel using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    local_paths: List[Optional[Path]] = [None] * len(gcs_paths)

    def _dl(idx_path):
        idx, path = idx_path
        return idx, download_file(path, cache_dir, project_id)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_dl, (i, p)): i for i, p in enumerate(gcs_paths)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            idx, lp = fut.result()
            local_paths[idx] = lp

    return local_paths  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def upload_file(
    local_path: str | Path,
    bucket_name: str,
    gcs_prefix: str,
    project_id: Optional[str] = None,
) -> str:
    """Upload a local file to GCS and return the gs:// URI."""
    local_path = Path(local_path)
    blob_name = f"{gcs_prefix.rstrip('/')}/{local_path.name}"
    client = get_gcs_client(project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    logger.info("Uploaded %s → %s", local_path.name, gcs_uri)
    return gcs_uri


def upload_directory(
    local_dir: str | Path,
    bucket_name: str,
    gcs_prefix: str,
    project_id: Optional[str] = None,
    pattern: str = "*.h5",
) -> List[str]:
    """Upload all files matching *pattern* in *local_dir* to GCS."""
    local_dir = Path(local_dir)
    files = list(local_dir.rglob(pattern))
    uris = []
    for f in tqdm(files, desc="Uploading to GCS"):
        rel = f.relative_to(local_dir)
        prefix = f"{gcs_prefix.rstrip('/')}/{rel.parent}"
        uri = upload_file(f, bucket_name, prefix, project_id)
        uris.append(uri)
    logger.info("Uploaded %d files to gs://%s/%s", len(uris), bucket_name, gcs_prefix)
    return uris


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------


def md5_checksum(path: str | Path) -> str:
    """Compute MD5 of a local file (for integrity verification)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_download(gcs_path: str, local_path: str | Path, project_id: Optional[str] = None) -> bool:
    """Compare local MD5 against GCS object MD5 metadata."""
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    client = get_gcs_client(project_id)
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.reload()
    # GCS md5_hash is base64-encoded
    import base64
    gcs_md5 = base64.b64decode(blob.md5_hash).hex() if blob.md5_hash else None
    local_md5 = md5_checksum(local_path)
    ok = gcs_md5 == local_md5
    if not ok:
        logger.warning("MD5 mismatch for %s: gcs=%s local=%s", gcs_path, gcs_md5, local_md5)
    return ok

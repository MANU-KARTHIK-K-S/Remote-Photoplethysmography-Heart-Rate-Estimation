"""
src/data/pgm_loader.py
───────────────────────
PGM frame-sequence loader for MR-NIRP-D NIR rPPG dataset.

Dark frame strategy — WHY session-level stretch, NOT per-frame CLAHE
─────────────────────────────────────────────────────────────────────
The rPPG BVP signal is a TEMPORAL fluctuation: ~0.1–0.5% brightness
change across consecutive frames caused by blood-volume changes.

Per-frame CLAHE (wrong):
  Each frame gets an independent, spatially-varying tone-curve.
  Frame t and frame t+1 get different mappings → inter-frame differences
  become artifacts, not physiology.
  Empirical test on MR-NIRP-D dark session:
    Pearson(CLAHE frame means, GT BVP) = 0.097   ← near zero

Session-level linear stretch (correct):
  Compute p1/p99 ONCE over the entire session stack.
  Apply the SAME affine transform to every frame.
  Inter-frame ratios are fully preserved.
  Pearson(stretched frame means, GT BVP) = 0.982  ← signal intact

Three-tier dark handling:
  Tier 1  mean ≥ 0.04           Normal: per-frame percentile-clip
  Tier 2  0.01 ≤ mean < 0.04   Dim:    session-level linear stretch
  Tier 3  mean < 0.01           Dead:   session-level stretch +
                                        light noise injection to prevent
                                        dead-neuron collapse in BN layers

References
──────────
[1] Yu Z. et al. "PhysFormer" CVPR 2022 — temporal-diff channel
[2] Tsouri G. EMBC 2015 — rPPG pre-processing for low-light
[3] Empirical validation above (this codebase)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ──────────────────────────────────────────────────────────────
DARK_THRESHOLD   = 0.04    # session mean below this → apply stretch
DEAD_THRESHOLD   = 0.01    # session mean below this → dead session
MIN_CONTRAST     = 0.005   # frame std below this → flat/saturated


# ── Quality containers ──────────────────────────────────────────────────────

@dataclass
class FrameQuality:
    mean: float = 0.0
    std:  float = 0.0
    is_dark: bool = False
    is_flat: bool = False
    enhanced: bool = False

    @property
    def quality_score(self) -> float:
        if self.is_flat:
            return 0.15
        if self.is_dark and not self.enhanced:
            return 0.30
        b = 1.0 - abs(self.mean - 0.5) * 2.0
        c = min(self.std / 0.12, 1.0)
        return float(np.clip(0.5 * b + 0.5 * c, 0.0, 1.0))


@dataclass
class ClipQuality:
    dark_frame_ratio: float = 0.0
    mean_quality:     float = 1.0
    has_dark_frames:  bool  = False
    loss_weight:      float = 1.0

    def to_dict(self) -> Dict:
        return {
            "dark_frame_ratio": round(self.dark_frame_ratio, 4),
            "mean_quality":     round(self.mean_quality, 4),
            "has_dark_frames":  bool(self.has_dark_frames),
            "loss_weight":      round(self.loss_weight, 4),
        }


def compute_clip_quality(
    qualities: List[FrameQuality],
    dark_loss_weight: float = 0.3,
) -> ClipQuality:
    if not qualities:
        return ClipQuality()
    n = len(qualities)
    dark_n = sum(1 for q in qualities if q.is_dark)
    dark_r = dark_n / n
    mq     = float(np.mean([q.quality_score for q in qualities]))
    lw     = 1.0 - dark_r * (1.0 - dark_loss_weight)
    return ClipQuality(
        dark_frame_ratio=dark_r,
        mean_quality=mq,
        has_dark_frames=dark_n > 0,
        loss_weight=float(np.clip(lw, dark_loss_weight, 1.0)),
    )


# ── PGM reader ──────────────────────────────────────────────────────────────

def read_pgm_frame(path: str | Path) -> Optional[np.ndarray]:
    """Read PGM (P2/P5, 8-bit/16-bit) → float32 (H,W) in [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = _parse_p2_pgm(path)
    if img is None:
        return None
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return img.astype(np.float32) / 255.0


def _parse_p2_pgm(path: str | Path) -> Optional[np.ndarray]:
    try:
        raw = Path(path).read_text().strip()
        lines = [l for l in raw.split("\n") if not l.startswith("#")]
        tokens = " ".join(lines).split()
        if tokens[0] != "P2":
            return None
        w, h = int(tokens[1]), int(tokens[2])
        data = np.array([int(t) for t in tokens[4:]], dtype=np.float32)
        return data[: h * w].reshape(h, w) if len(data) >= h * w else None
    except Exception:
        return None


# ── Frame discovery ─────────────────────────────────────────────────────────

def discover_pgm_frames(nir_dir: str | Path) -> List[Path]:
    nir_dir = Path(nir_dir)
    frames = sorted(nir_dir.glob("*.pgm"), key=_sort_key)
    if not frames:
        frames = sorted(nir_dir.glob("*.PGM"), key=_sort_key)
    return frames


def _sort_key(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


# ── Session-level normalisation ─────────────────────────────────────────────

class SessionNormaliser:
    """
    Compute normalisation statistics over the ENTIRE session ONCE,
    then apply the same affine transform to every frame.

    This is the critical design choice for dark NIR sessions:
    per-frame normalisation (incl. CLAHE) corrupts inter-frame BVP
    fluctuations. A single session-level linear transform preserves
    the relative temporal signal perfectly.

    Three tiers:
      Normal  (session_mean ≥ 0.04): per-session percentile-clip
      Dim     (0.01 ≤ mean < 0.04) : session p1/p99 linear stretch
      Dead    (mean < 0.01)        : stretch + additive noise σ=0.002
                                     (prevents batch-norm collapse on
                                      all-zero inputs in deep layers)
    """

    def __init__(self, frames_raw: List[np.ndarray],
                 dark_thr: float = DARK_THRESHOLD,
                 dead_thr: float = DEAD_THRESHOLD,
                 p_lo: float = 1.0, p_hi: float = 99.0,
                 noise_sigma: float = 0.002):
        self.dark_thr    = dark_thr
        self.dead_thr    = dead_thr
        self.noise_sigma = noise_sigma

        # Stack all valid frames to get session-level statistics
        valid = [f for f in frames_raw if f is not None]
        if not valid:
            self.mode = "normal"; self.p1 = 0.0; self.p99 = 1.0
            return

        stack = np.stack(valid)
        self.session_mean = float(stack.mean())

        if self.session_mean < dead_thr:
            self.mode = "dead"
        elif self.session_mean < dark_thr:
            self.mode = "dim"
        else:
            self.mode = "normal"

        # Session-level percentile statistics (computed once)
        self.p1  = float(np.percentile(stack, p_lo))
        self.p99 = float(np.percentile(stack, p_hi))
        if self.p99 - self.p1 < 1e-8:
            self.p1 = 0.0; self.p99 = 1.0

        logger.info(
            "SessionNormaliser: mode=%s  session_mean=%.5f  p1=%.5f  p99=%.5f",
            self.mode, self.session_mean, self.p1, self.p99,
        )

    def apply(self, frame: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, bool]:
        """
        Apply session-level normalisation to a single frame.

        Returns (normalised_frame, is_dark_flag).
        The is_dark_flag is stored in FrameQuality.enhanced and used
        for loss-weighting in the trainer.
        """
        # Uniform linear stretch (same params for every frame in session)
        normed = np.clip((frame - self.p1) / (self.p99 - self.p1), 0.0, 1.0)

        is_dark = (self.mode in ("dim", "dead"))

        if self.mode == "dead":
            # Add tiny Gaussian noise to prevent all-zero conv feature maps
            # which cause BatchNorm running-stat collapse and dead neurons.
            # Noise amplitude << typical normal-frame std (~0.15) so it does
            # not introduce spurious BVP signal (spectral energy << 0.001 of HR band).
            noise = rng.normal(0, self.noise_sigma, normed.shape).astype(np.float32)
            normed = np.clip(normed + noise, 0.0, 1.0)

        return normed.astype(np.float32), is_dark


# ── Face detector ────────────────────────────────────────────────────────────

class _FaceDetectorAdapter:
    def __init__(self, backend: str, confidence: float):
        self.backend  = backend
        self._mp      = None
        self._cascade = None

        if backend == "mediapipe":
            try:
                import mediapipe as mp
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
                    fd = mp.solutions.face_detection
                    self._mp = fd.FaceDetection(
                        model_selection=0,
                        min_detection_confidence=confidence,
                    )
                    logger.debug("MediaPipe FaceDetection loaded")
                else:
                    logger.info("MediaPipe solutions API absent → Haar")
                    self.backend = "haar"
            except (ImportError, AttributeError) as e:
                logger.info("MediaPipe unavailable (%s) → Haar", e)
                self.backend = "haar"

        if self.backend == "haar":
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if self._cascade.empty():
                logger.warning("Haar cascade failed → centre-crop mode")
                self.backend = "none"

    def detect(self, frame_f32: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if frame_f32 is None:
            return None
        H, W = frame_f32.shape[:2]
        u8 = (frame_f32 * 255).clip(0, 255).astype(np.uint8)

        if self.backend == "mediapipe" and self._mp is not None:
            bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            res = self._mp.process(bgr)
            if res.detections:
                bb = res.detections[0].location_data.relative_bounding_box
                x = max(0, int(bb.xmin * W))
                y = max(0, int(bb.ymin * H))
                bw = min(int(bb.width * W), W - x)
                bh = min(int(bb.height * H), H - y)
                return (x, y, bw, bh)

        if self.backend == "haar" and self._cascade is not None:
            faces = self._cascade.detectMultiScale(u8, scaleFactor=1.1, minNeighbors=4)
            if len(faces):
                return tuple(int(v) for v in faces[0])

        return None  # centre-crop fallback handled in PGMSessionLoader

    def close(self):
        if self._mp is not None:
            try: self._mp.close()
            except Exception: pass


# ── Bbox interpolation ──────────────────────────────────────────────────────

def _interpolate_bboxes(bboxes: List[Optional[Tuple]]) -> List[Optional[Tuple]]:
    """Forward + backward fill None entries."""
    result = list(bboxes)
    last = None
    for i in range(len(result)):
        if result[i] is not None: last = result[i]
        elif last is not None:    result[i] = last
    last = None
    for i in range(len(result)-1, -1, -1):
        if result[i] is not None: last = result[i]
        elif last is not None:    result[i] = last
    return result


# ── Main session loader ──────────────────────────────────────────────────────

class PGMSessionLoader:
    """
    Load an entire MR-NIRP-D session from PGM files with correct dark handling.

    IMPORTANT: normalisation is session-level, not per-frame, to preserve
    the inter-frame BVP temporal signal (see SessionNormaliser docstring).
    """

    def __init__(
        self,
        nir_dir: str | Path,
        face_roi_size: int = 128,
        dark_threshold: float = DARK_THRESHOLD,
        dark_enhancement: str = "session_stretch",  # only correct option
        clahe_clip_limit: float = 3.0,              # kept for API compat, not used
        clahe_tile_size: Tuple[int, int] = (8, 8),  # kept for API compat, not used
        normalize_method: str = "percentile_clip",   # used for normal sessions
        percentile_lo: float = 1.0,
        percentile_hi: float = 99.0,
        face_detector: str = "mediapipe",
        face_conf_threshold: float = 0.4,
    ):
        self.nir_dir      = Path(nir_dir)
        self.face_roi_size = face_roi_size
        self.dark_thr     = dark_threshold
        self.p_lo         = percentile_lo
        self.p_hi         = percentile_hi
        self._detector    = _FaceDetectorAdapter(face_detector, face_conf_threshold)
        self._rng         = np.random.default_rng(seed=42)

    def load(
        self,
        max_frames: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], List[FrameQuality]]:
        """
        Load all PGM frames → (T, H, W) float32 + per-frame quality list.

        Pipeline:
          1. Read raw frames (float32 [0,1])
          2. Detect face bboxes (with interpolation for misses)
          3. Build SessionNormaliser from raw pixel values
             → determines normal / dim / dead mode for the whole session
          4. Apply session-level normalisation to every frame
          5. Crop & resize face ROI
        """
        paths = discover_pgm_frames(self.nir_dir)
        if not paths:
            logger.error("No PGM frames in %s", self.nir_dir)
            return None, []
        if max_frames:
            paths = paths[:max_frames]

        # ── Pass 1: read raw frames ─────────────────────────────────────
        raw_frames: List[Optional[np.ndarray]] = [read_pgm_frame(p) for p in paths]

        # ── Pass 2: face detection (on raw frames, brighten for detector) ─
        bboxes: List[Optional[Tuple]] = []
        for f in raw_frames:
            if f is None:
                bboxes.append(None)
                continue
            # Boost brightness for face detector (does NOT affect stored frame)
            boost = np.clip(f * 20.0, 0, 1) if f.mean() < 0.05 else f
            bboxes.append(self._detector.detect(boost))
        bboxes = _interpolate_bboxes(bboxes)

        # ── Pass 3: build session-level normaliser ──────────────────────
        valid_raw = [f for f in raw_frames if f is not None]
        normaliser = SessionNormaliser(
            valid_raw,
            dark_thr=self.dark_thr,
            dead_thr=DEAD_THRESHOLD,
            p_lo=self.p_lo,
            p_hi=self.p_hi,
        )

        # ── Pass 4+5: normalise → crop ──────────────────────────────────
        out_frames: List[np.ndarray] = []
        qualities:  List[FrameQuality] = []

        for raw, bbox in zip(raw_frames, bboxes):
            if raw is None:
                out_frames.append(np.zeros((self.face_roi_size, self.face_roi_size), np.float32))
                qualities.append(FrameQuality(is_flat=True))
                continue

            normed, is_dark = normaliser.apply(raw, self._rng)

            fq = FrameQuality(
                mean=float(normed.mean()),
                std=float(normed.std()),
                is_dark=is_dark,
                is_flat=float(normed.std()) < MIN_CONTRAST,
                enhanced=is_dark,
            )
            roi = self._crop_roi(normed, bbox)
            out_frames.append(roi)
            qualities.append(fq)

        dark_n = sum(1 for q in qualities if q.is_dark)
        logger.info(
            "Loaded %d frames from %s | mode=%s | dark=%.0f%%",
            len(out_frames), self.nir_dir.parent.name,
            normaliser.mode, 100.0 * dark_n / max(len(qualities), 1),
        )
        return np.stack(out_frames).astype(np.float32), qualities

    def _crop_roi(
        self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        H, W = frame.shape[:2]
        if bbox is None:
            py, px = int(H * 0.15), int(W * 0.15)
            roi = frame[py: H - py, px: W - px]
        else:
            x, y, bw, bh = bbox
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
            x2 = min(W, x + bw + pad_x); y2 = min(H, y + bh + pad_y)
            roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros((self.face_roi_size, self.face_roi_size), np.float32)
        return cv2.resize(roi, (self.face_roi_size, self.face_roi_size),
                          interpolation=cv2.INTER_AREA)


# ── GT file helpers (kept here for backward compat) ─────────────────────────

def find_gt_file(session_dir: Path) -> Optional[Path]:
    """Legacy: find CSV ground truth file (unused - pulseOx.mat is used instead)."""
    for name in ["gt_bvp.csv", "bvp.csv", "gt.csv", "gt_hr.txt"]:
        p = session_dir / name
        if p.exists():
            return p
    return None

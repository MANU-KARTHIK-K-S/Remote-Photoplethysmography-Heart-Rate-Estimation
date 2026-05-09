# Heart Rate Estimation from NIR Facial Video Sequence
## MR-NIRP-D Dataset · PhysFormer-NIR Architecture

---

## Overview

End-to-end pipeline for estimating Heart Rate (BPM) from monocular Near-Infrared (NIR) facial video sequences using the MR-NIRP-D dataset. Uses only the NIR modality (no RGB, depth, or thermal).

**Architecture:** PhysFormer-NIR — a 3-D CNN with Temporal Difference Convolution blocks feeding a 4-layer Temporal Transformer. Predicts a per-frame rPPG waveform; BPM is extracted via Welch PSD with parabolic interpolation.

Final Report: [PhysFormer_NIR_Final_Submission.pdf](Reports/PhysFormer_NIR_Final_Submission.pdf)

---

## Architecture Diagram

```
PHYSFORMER-NIR ARCHITECTURE

  INPUT CLIPS  [B, 1, 160, 128, 128]  (NIR frames, 1-channel, T=160)

  ┌─────────────────────────────────────────────────────────────────┐
  │  PATCH EMBEDDING  (Tubelet)                                     │
  │  3D Conv  t=4, s=4×4  →  tokens: [B, T/4 × (H/4×W/4), 96]       │
  │  Position encoding (learnable, sinusoidal-init)                 │
  └────────────────────────┬────────────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────────────┐
  │  TRANSFORMER ENCODER  ×4 layers                                 │
  │  ┌──────────────────────────────────┐                           │
  │  │  LayerNorm                        │  (Pre-LN for stability)  │
  │  │  Multi-Head Self-Attention (8 h)  │                          │
  │  │  Temporal Difference Attention    │  (physiology-aware)      │
  │  │  Dropout 0.1                      │                          │
  │  │  LayerNorm + FFN (MLP ratio 4×)   │                          │
  │  └──────────────────────────────────┘                           │
  └────────────────────────┬────────────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────────────┐
  │  TEMPORAL AVERAGE POOLING  →  [B, 96]                           │
  └────────────────────────┬────────────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────────────┐
  │  REGRESSION HEAD   Linear(96→1)  →  HR [BPM]                    │
  └─────────────────────────────────────────────────────────────────┘

  Loss = α · L_MAE  +  β · L_SNR  +  γ · L_NegPearson
         (α=1.0)         (β=0.5)        (γ=0.5)
```

## Project Structure

```
hr_nir_estimation/
├── configs/
│   └── config.yaml              Master config (GCS, dataset, model, training)
├── src/
│   ├── data/
│   │   ├── pgm_loader.py        PGM reader + session-level dark normalisation
│   │   ├── gt_loader.py         pulseOx.mat + cam timestamps → synced BVP GT
│   │   ├── preprocessing.py     Session → clips → HDF5 shards
│   │   └── dataset.py           PyTorch Dataset + quality-aware DataLoader
│   ├── models/
│   │   └── physformer_nir.py    PhysFormer-NIR (TDC + Transformer)
│   ├── losses/
│   │   └── losses.py            Neg-Pearson + Freq-domain KL + SNR loss
│   ├── training/
│   │   └── trainer.py           Training loop, AMP, W&B, quality weighting
│   ├── evaluation/
│   │   └── metrics.py           MAE, RMSE, MAPE, Pearson, SNR per clip
│   └── utils/
│       ├── signal_processing.py Bandpass, Welch PSD, BPM estimation
│       └── gcs_utils.py         GCS upload/download helpers
├── scripts/
│   ├── upload_to_gcs.py         Upload PGM dataset to GCS bucket
│   ├── preprocess_dataset.py    PGM → face crop → HDF5 clips
│   ├── inspect_mat.py           Diagnostic: inspect pulseOx.mat files
│   ├── train.py                 Training entry point
│   └── evaluate.py              Evaluation + plots
├── requirements.txt
└── Makefile
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure GCS
# Edit configs/config.yaml → set gcs.project_id and gcs.bucket_name

# 3. Upload PGMs to GCS (run once after Google Drive download)
make upload DATASET_ROOT=/path/to/mr-nirp-d

# 4. Inspect GT files (optional diagnostic)
make inspect-mat DATASET_ROOT=/path/to/mr-nirp-d

# 5. Preprocess: PGM frames → HDF5 clips
make preprocess DATASET_ROOT=/path/to/mr-nirp-d

# 6. Train
make train

# 7. Evaluate
make evaluate
```

---

## Dataset: MR-NIRP-D

| Property | Value |
|---|---|
| Subjects | 3–4 |
| Sessions | Multiple per subject (resting indoor/outdoor, exercise, dark scene) |
| NIR files | ~28,000 PGM frames total |
| Format | P5 binary PGM, 8-bit or 16-bit grayscale |
| Frame rate | 30 fps |
| Ground truth | `pulseOx.mat` (125 Hz finger-clip PPG) + `cam0_full_log.txt` (Unix timestamps) |

### Directory layout expected

```
<dataset-root>/
  Subject1/
    resting_indoor/
      NIR/  frame_000000.pgm … frame_XXXXXX.pgm
      cam0_full_log.txt          ← [ts1, ts2, …]  (Python list literal)
      cam0_partial_log.txt
    resting_outdoor/ …
    exercise/ …
    PulseOX/
      pulseOx.mat                ← keys: time, data, rate, spo2, hr_bpm
  Subject2/ …
  Subject4/
    dark_scene/                  ← handled specially (see below)
```

---

## Ground Truth Derivation

This is the key methodological contribution. The assignment asks to document how BPM is derived from the reference signal.

### Step-by-step

**Step 1 — Load `pulseOx.mat`**
Contact finger-clip pulse oximeter at 125 Hz. Fields:
- `time`: Unix epoch timestamps (float64)
- `data`: raw BVP waveform amplitude
- `rate`: sample rate (125.0 Hz)

**Step 2 — Parse `cam0_full_log.txt`**
Python list-literal of per-frame Unix timestamps at ~30 fps:
```
[1540577618.696552, 1540577618.71555, …]
```
Parsed with `ast.literal_eval` (safe, handles multi-line).  
`cam0_full_log.txt` preferred over `cam0_partial_log.txt` (no dropped frames).

**Step 3 — Find temporal overlap**
Intersect `[t_cam[0], t_cam[-1]]` with pulseOx time range. If pulseOx timestamps are relative (max < 1e8), reconstruct by duration alignment.

**Step 4 — Bandpass filter at 125 Hz**
Butterworth 4th-order `[0.75, 3.0] Hz` (= 45–180 BPM). Filtering at the native high rate maximises frequency resolution before downsampling.

**Step 5 — Interpolate to camera timestamps**
`scipy.interpolate.interp1d` (linear) maps the 125-Hz filtered BVP onto the per-frame Unix timestamps → `bvp_cam` shape `(N_frames,)`.

**Step 6 — Z-score normalise**
`bvp_cam = (bvp_cam - mean) / std` — removes DC offset; loss scale-invariant.

**Step 7 — BPM estimation (evaluation only)**
Welch PSD on 10-second windows (`nperseg = 5×30`):
```python
freqs, power = scipy.signal.welch(bvp_cam, fs=30, nperseg=150)
peak_freq = freqs[band_mask][argmax(power[band_mask])]
bpm = peak_freq * 60
```
Parabolic interpolation between FFT bins → sub-bin accuracy (±0.2 BPM).

> **Training label = Step 6 waveform** (not BPM).  
> The model predicts the rPPG waveform; BPM is extracted from the prediction at inference time using the same Step 7 pipeline.

---

## Dark Scene Handling

**Critical design decision** — per-frame CLAHE is wrong for rPPG.

### Why per-frame CLAHE fails

The BVP signal is a 0.1–0.5% temporal brightness change across frames. Per-frame CLAHE applies a different spatially-adaptive tone curve to each frame, destroying inter-frame consistency:

```
Pearson(CLAHE frame means, GT BVP) = 0.097  ← near zero
```

### Correct approach: session-level linear stretch

Compute `p1/p99` percentiles **once** over the entire session stack, then apply the **same** affine transform to every frame:

```python
normed[t] = clip((raw[t] - p1) / (p99 - p1), 0, 1)
```

```
Pearson(session-stretched means, GT BVP) = 0.982  ← preserved
```

### Three-tier dark handling

| Tier | Condition | Action | Loss weight |
|---|---|---|---|
| Normal | session_mean ≥ 0.04 | Session percentile-clip | 1.0 |
| Dim | 0.01 ≤ mean < 0.04 | Session linear stretch | 0.3 |
| Dead | mean < 0.01 | Stretch + noise σ=0.002 | 0.3 |

Noise injection on dead sessions prevents BatchNorm running-stat collapse when all inputs are near zero — a critical stability issue for deep models on all-black frames.

---

## Model: PhysFormer-NIR

Adapted from PhysFormer (Yu et al., CVPR 2022) for single-channel NIR input.

```
Input (B, T, 2, 128, 128)    2 channels: [I_t, I_t - I_{t-1}]
  │
  ▼
3-D CNN Stem (stride T/2, spatial /4)
  │
  ▼
4× TDC Residual Blocks        Temporal Difference Convolution
  θ=0.7                        W_eff = (1-θ)W + θW_diff
  │
  ▼
Global Spatial Pool → (B, T/2, 256)
  │
  ▼
4-layer Temporal Transformer   MHSA + FFN, sinusoidal PE
  8 heads, d=256
  │
  ▼
rPPG Head (linear → interpolate → T frames)
  │
  ▼
Output (B, T)   rPPG waveform → BPM via Welch PSD
```

**NIR adaptations vs original PhysFormer (RGB):**
- `in_channels=2` (NIR intensity + temporal diff) vs 3 RGB
- Reduced stem channels (32→64 vs 64→128) for smaller dataset
- Lower LR (`5e-5` vs `1e-4`) due to 4-subject dataset size

---

## Loss Functions

Three complementary losses:

| Loss | Weight | Purpose |
|---|---|---|
| Negative Pearson | 1.0 | Waveform shape fidelity |
| Freq-domain KL | 0.5 | Spectral peak location |
| SNR loss | 0.2 | Suppress harmonic artefacts |

**Quality-aware weighting:** dark clips contribute `loss × 0.3` to avoid the model being dominated by noisy signals while still learning from them.

---

## Training Setup

| Parameter | Value | Rationale |
|---|---|---|
| Batch size | 4 (eff. 16 with accum.) | Small dataset |
| LR | 5e-5 | Conservative for 4 subjects |
| Schedule | Cosine restarts + 5-ep warmup | Escape sharp minima |
| AMP | FP16 | Memory efficiency |
| Early stop | patience=15 epochs | Prevent overfitting |
| Augmentation | H-flip, temporal reversal, noise, brightness | BVP-safe only |

---

## Evaluation Metrics

| Metric | Unit | Description |
|---|---|---|
| MAE | BPM | Primary metric |
| RMSE | BPM | Error sensitivity |
| MAPE | % | Scale-independent error |
| Pearson r | — | BPM correlation |
| SNR | dB | Spectral quality of predicted rPPG |

Results broken down by: dark vs normal clips, per-subject, overall.

---

## References

1. Yu Z. et al. **PhysFormer** — CVPR 2022. TDC blocks, temporal transformer, freq-domain loss.
2. Liu X. et al. **TS-CAN** — NeurIPS 2020. Two-channel (intensity + diff) input.
3. Chen W. & McDuff D. **DeepPhys** — ECCV 2018. Attention-based rPPG CNN.
4. Hu S. et al. **MR-NIRP Dataset** — Rice CIL 2019. Dataset description.
5. de Haan G. & Jeanne V. — IEEE T-BME 2013. BVP bandpass methodology.
6. Poh M-Z et al. — Optics Express 2010. FFT-based HR estimation from rPPG.

"""
src/models/physformer_nir.py
─────────────────────────────
PhysFormer-NIR: our NIR-adapted spatio-temporal rPPG model.

Architecture overview
─────────────────────

  Input (T, C, H, W)  C=1 (intensity) or C=2 (intensity + temporal-diff)
       │
  ┌────▼─────────────────────────────────────────┐
  │  3-D CNN Stem                                 │
  │  Two stem blocks: conv3d → BN → GELU → pool  │
  │  Output: (B, 64, T/4, H/8, W/8)              │
  └────────────────────┬─────────────────────────┘
                       │
  ┌────────────────────▼─────────────────────────┐
  │  Temporal Difference Convolution (TDC) Blocks │
  │  4× TDC block  (PhysFormer key component)     │
  │  Each block:  TDC → BN → GELU → residual      │
  │  Output: (B, 256, T/4, H/32, W/32)            │
  └────────────────────┬─────────────────────────┘
                       │  Global spatial pooling → (B, T/4, 256)
  ┌────────────────────▼─────────────────────────┐
  │  Temporal Transformer (4 layers, 8 heads)     │
  │  Multi-head self-attention over T/4 tokens    │
  │  With sinusoidal positional encoding          │
  │  Output: (B, T/4, 256)                        │
  └────────────────────┬─────────────────────────┘
                       │
  ┌────────────────────▼─────────────────────────┐
  │  rPPG Regressor                               │
  │  Linear → rPPG signal at original T rate      │
  │  (via temporal upsampling × 4)                │
  └──────────────────────────────────────────────┘

References
──────────
[1] Yu Z. et al.  "PhysFormer: Facial Video-based Physiological Measurement
    with Temporal Difference Transformer."  CVPR 2022.
    https://arxiv.org/abs/2111.12082
    – We adopt the TDC block and temporal transformer design.
    – Modification: NIR single/dual channel input instead of RGB 3-channel.
    – Modification: Lightweight 3-D CNN stem tuned for 128×128 NIR clips.

[2] Liu X. et al.  "TS-CAN: Temporal Shift Convolutional Attention Network
    for Real-Time Physiological Measurement."  NeurIPS 2020.
    – Inspired the 2-channel (intensity + diff) input representation.

[3] Chen W. & McDuff D.  "DeepPhys: Video-Based Physiological Measurement
    Using Convolutional Attention Networks."  ECCV 2018.
    – Attention module design reference.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Temporal Difference Convolution (TDC)
# ---------------------------------------------------------------------------


class TemporalDiffConv3d(nn.Module):
    """
    Temporal Difference Convolution from PhysFormer (Yu et al. 2022, Eq. 2).

    For each temporal position t the effective kernel is:
        W_eff = W_center + θ × (W_temporal − W_center)
    applied to feature difference f[t] − f[t±1].

    θ (tdc_theta) controls the balance between standard and difference conv.
    Default θ=0.7 from the original paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
        theta: float = 0.7,
    ):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        # Separate weight for temporal difference branch
        self.conv_diff = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W)"""
        # Standard convolution
        out_std = self.conv(x)

        # Temporal difference: shift feature map along T dimension
        x_diff = torch.zeros_like(x)
        x_diff[:, :, 1:, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        out_diff = self.conv_diff(x_diff)

        # Combine
        out = (1.0 - self.theta) * out_std + self.theta * out_diff
        out = out + self.bias[None, :, None, None, None]
        return out


class TDCBlock(nn.Module):
    """TDC residual block: TDC → BN → GELU → TDC → BN → skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        theta: float = 0.7,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
    ):
        super().__init__()
        stride = (temporal_stride, spatial_stride, spatial_stride)
        padding = (1, 1, 1)

        self.tdc1 = TemporalDiffConv3d(in_channels, out_channels, stride=stride, padding=padding, theta=theta)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
        self.tdc2 = TemporalDiffConv3d(out_channels, out_channels, theta=theta)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Skip projection if channel / spatial dims change
        self.skip = nn.Sequential()
        if in_channels != out_channels or temporal_stride != 1 or spatial_stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.bn1(self.tdc1(x)))
        out = self.bn2(self.tdc2(out))
        return self.act(out + identity)


# ---------------------------------------------------------------------------
# 3-D CNN Stem
# ---------------------------------------------------------------------------


class Stem3D(nn.Module):
    """
    Lightweight 3-D CNN stem.
    Reduces spatial dimensions early so the TDC blocks are affordable.
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        mid = out_channels // 2
        self.block = nn.Sequential(
            # Block 1: heavy spatial downsampling, light temporal
            nn.Conv3d(in_channels, mid, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(mid),
            nn.GELU(),
            # Block 2
            nn.Conv3d(mid, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Spatial Global Average Pooling → sequence of temporal tokens
# ---------------------------------------------------------------------------


class SpatialPool(nn.Module):
    """Pool (B, C, T, H, W) → (B, T, C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average over spatial dims
        return x.mean(dim=(-1, -2)).permute(0, 2, 1)  # (B, T, C)


# ---------------------------------------------------------------------------
# Temporal Transformer
# ---------------------------------------------------------------------------


class TemporalTransformerBlock(nn.Module):
    """Single transformer encoder block: MHSA → FFN with pre-norm (LN)."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # MHSA with residual
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, attn_mask=mask)
        x = x + y
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, depth: int, dropout: float):
        super().__init__()
        self.pe = SinusoidalPE(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [TemporalTransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# rPPG Regressor head
# ---------------------------------------------------------------------------


class RPPGHead(nn.Module):
    """
    Map temporal tokens → per-frame rPPG amplitude.

    The CNN backbone has temporal stride 2, so we get T//2 tokens
    (after stem stride-2 and no temporal stride in TDC blocks).
    We upsample back to T frames via linear interpolation + a 1-D conv.
    """

    def __init__(self, d_model: int, upsample_factor: int = 2):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        """x: (B, T', d_model) → (B, T)"""
        # Project to scalar per token
        out = self.proj(x).squeeze(-1)  # (B, T')
        # Upsample to target_T
        out = F.interpolate(out.unsqueeze(1), size=target_T, mode="linear", align_corners=False)
        return out.squeeze(1)  # (B, T)


# ---------------------------------------------------------------------------
# Full PhysFormer-NIR model
# ---------------------------------------------------------------------------


class PhysFormerNIR(nn.Module):
    """
    PhysFormer-NIR: Full rPPG model for monocular NIR facial video.

    Input  : (B, T, C, H, W)  C=1 (NIR) or C=2 (NIR + temporal diff)
    Output : (B, T) rPPG signal  (used for BPM estimation via FFT)

    Parameters (all exposed in configs/config.yaml → model section)
    ──────────────────────────────────────────────────────────────
    in_channels       : 1 (intensity only) or 2 (intensity + diff)
    stem_channels     : output channels of stem [32, 64]
    tdc_blocks        : number of TDC residual blocks (4)
    tdc_channels      : channel progression in TDC blocks
    tdc_theta         : temporal difference mixing ratio (0.7)
    transformer_depth : number of transformer layers (4)
    transformer_heads : attention heads (8)
    transformer_dim   : d_model (256)
    transformer_ff_dim: feed-forward hidden dim (512)
    transformer_dropout: attention dropout (0.1)
    """

    def __init__(
        self,
        in_channels: int = 2,
        stem_out_channels: int = 64,
        tdc_channels: Tuple[int, ...] = (64, 128, 128, 256),
        tdc_theta: float = 0.7,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_dim: int = 256,
        transformer_ff_dim: int = 512,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()

        # 3-D CNN Stem
        self.stem = Stem3D(in_channels, stem_out_channels)

        # TDC Blocks with progressive channel widening & spatial downsampling
        tdc_layers = []
        in_ch = stem_out_channels
        for i, out_ch in enumerate(tdc_channels):
            # Downsample spatially in first 2 TDC blocks; keep temporal stride=1
            s_spatial = 2 if i < 2 else 1
            tdc_layers.append(TDCBlock(in_ch, out_ch, theta=tdc_theta, spatial_stride=s_spatial))
            in_ch = out_ch
        self.tdc = nn.Sequential(*tdc_layers)
        cnn_out_ch = tdc_channels[-1]

        # Project CNN features to transformer dim
        self.token_proj = nn.Linear(cnn_out_ch, transformer_dim)

        # Spatial pooling: (B, C, T', H', W') → (B, T', C)
        self.spatial_pool = SpatialPool()

        # Temporal Transformer
        self.transformer = TemporalTransformer(
            d_model=transformer_dim,
            n_heads=transformer_heads,
            ff_dim=transformer_ff_dim,
            depth=transformer_depth,
            dropout=transformer_dropout,
        )

        # rPPG regression head (upsample from T'=T//2 back to T)
        # Stem has temporal stride 2, TDC has no temporal stride → T' = T//2
        self.rppg_head = RPPGHead(transformer_dim, upsample_factor=2)

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for conv, Xavier for linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        rppg : (B, T) — predicted rPPG waveform
        """
        B, T, C, H, W = x.shape

        # Rearrange for 3-D conv: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Stem
        x = self.stem(x)        # (B, 64, T//2, H//4, W//4)

        # TDC blocks
        x = self.tdc(x)         # (B, 256, T//2, H'', W'')

        # Spatial pool → temporal tokens
        tokens = self.spatial_pool(x)    # (B, T//2, 256)

        # Project to transformer dim (may be identity if dims match)
        tokens = self.token_proj(tokens)  # (B, T//2, d_model)

        # Temporal transformer
        tokens = self.transformer(tokens)  # (B, T//2, d_model)

        # Regress rPPG signal back to T frames
        rppg = self.rppg_head(tokens, target_T=T)  # (B, T)

        return rppg


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(cfg: dict) -> PhysFormerNIR:
    """Instantiate PhysFormerNIR from config dict."""
    mcfg = cfg["model"]
    in_ch = 2 if mcfg.get("use_temporal_diff_channel", True) else 1
    model = PhysFormerNIR(
        in_channels=in_ch,
        stem_out_channels=mcfg.get("stem_channels", [32, 64])[-1],
        tdc_channels=tuple(mcfg.get("tdc_channels", [64, 128, 128, 256])),
        tdc_theta=mcfg.get("tdc_theta", 0.7),
        transformer_depth=mcfg.get("transformer_depth", 4),
        transformer_heads=mcfg.get("transformer_heads", 8),
        transformer_dim=mcfg.get("transformer_dim", 256),
        transformer_ff_dim=mcfg.get("transformer_ff_dim", 512),
        transformer_dropout=mcfg.get("transformer_dropout", 0.1),
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PhysFormerNIR] Parameters: {n_params/1e6:.2f}M")
    return model

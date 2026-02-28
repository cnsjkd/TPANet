from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchaudio.models import Conformer
except Exception:
    Conformer = None


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x).clamp_min(1e-20))


class SincFilterbank(nn.Module):
    """SincNet-style learnable band-pass filterbank.

    Input:
      x: (N, C, T)
    Output:
      y: (N, C, B, T)
    """

    def __init__(
        self,
        fs: int = 200,
        bands_hz: Tuple[Tuple[float, float], ...] = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 50)),
        kernel_size: int = 101,
        min_hz: float = 0.5,
        max_hz: float = 75.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd")

        self.fs = float(fs)
        self.kernel_size = int(kernel_size)
        self.min_hz = float(min_hz)
        self.max_hz = float(max_hz)
        self.num_bands = len(bands_hz)

        f1_init = torch.tensor([b[0] for b in bands_hz], dtype=torch.float32)
        f2_init = torch.tensor([b[1] for b in bands_hz], dtype=torch.float32)

        f1_init = f1_init.clamp(min=self.min_hz, max=self.max_hz - 1.0)
        band_init = (f2_init - f1_init).clamp(min=0.5)

        self.f1_param = nn.Parameter(inverse_softplus(f1_init - self.min_hz))
        self.band_param = nn.Parameter(inverse_softplus(band_init))

        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("n", n, persistent=False)

        w = 0.54 - 0.46 * torch.cos(2 * math.pi * (n + kernel_size // 2) / (kernel_size - 1))
        self.register_buffer("window", w, persistent=False)

    def _build_kernels(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        f1 = self.min_hz + F.softplus(self.f1_param)
        f2 = f1 + F.softplus(self.band_param)
        f2 = torch.clamp(f2, max=self.max_hz)

        f1n = (f1 / self.fs).to(device=device, dtype=dtype)
        f2n = (f2 / self.fs).to(device=device, dtype=dtype)

        n = self.n.to(device=device, dtype=dtype)[None, :]
        window = self.window.to(device=device, dtype=dtype)[None, :]

        def lowpass(fc: torch.Tensor) -> torch.Tensor:
            return 2 * fc[:, None] * torch.sinc(2 * fc[:, None] * n)

        bandpass = (lowpass(f2n) - lowpass(f1n)) * window
        bandpass = bandpass / (bandpass.abs().sum(dim=-1, keepdim=True) + 1e-8)
        return bandpass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, channels, _ = x.shape
        kernels = self._build_kernels(x.device, x.dtype)
        weight = kernels.repeat(channels, 1).view(channels * self.num_bands, 1, self.kernel_size)
        y = F.conv1d(x, weight, padding=self.kernel_size // 2, groups=channels)
        return y.view(n_batch, channels, self.num_bands, -1)


class DELike(nn.Module):
    """DE-like feature via log-variance per band and channel."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y - y.mean(dim=-1, keepdim=True)
        var = (y ** 2).mean(dim=-1)
        return torch.log(var + self.eps)


class SpatialGCNBlock(nn.Module):
    """Lightweight spatial GCN over EEG channels (shared across bands)."""

    def __init__(
        self,
        channels: int,
        bands: int,
        hidden: int = 16,
        beta: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1]")

        self.channels = int(channels)
        self.bands = int(bands)
        self.beta = float(beta)

        # Bias the initial adjacency toward self-connections.
        init_adj = 5.0 * torch.eye(self.channels, dtype=torch.float32)
        self.adj_logits = nn.Parameter(init_adj)
        self.lin1 = nn.Linear(1, int(hidden), bias=False)
        self.lin2 = nn.Linear(int(hidden), 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.zeros(()))

    def _normalized_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        logits = 0.5 * (self.adj_logits + self.adj_logits.transpose(0, 1))
        attn = F.softmax(logits, dim=-1).to(device=x.device, dtype=x.dtype)
        eye = torch.eye(self.channels, device=x.device, dtype=x.dtype)
        return (1.0 - self.beta) * eye + self.beta * attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, B)
        if x.ndim != 3 or x.size(1) != self.channels or x.size(2) != self.bands:
            raise ValueError(
                f"expected x shape (N,{self.channels},{self.bands}), got {tuple(x.shape)}"
            )

        a_hat = self._normalized_adjacency(x)
        h = x.unsqueeze(-1)  # (N, C, B, 1)
        h = torch.einsum("ij,njbf->nibf", a_hat, h)
        h = self.lin1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = torch.einsum("ij,njbf->nibf", a_hat, h)
        h = self.lin2(h).squeeze(-1)  # (N, C, B)
        h = self.dropout(h)
        return x + self.gamma * h


class TCNSmoother(nn.Module):
    """Learnable smoothing over window axis W."""

    def __init__(self, dim: int, kernel_size: int = 5, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd")

        blocks = []
        for _ in range(int(layers)):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            valid = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            x = x * valid.unsqueeze(-1)

        h = x.transpose(1, 2)
        for block in self.blocks:
            h = block(h) + h
        out = h.transpose(1, 2)

        if lengths is not None:
            valid = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
            out = out * valid.unsqueeze(-1)

        return self.norm(out)


@dataclasses.dataclass
class FrontendConfig:
    fs: int = 200
    kernel_size: int = 101
    bands_hz: Tuple[Tuple[float, float], ...] = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 50))


class LearnableDELDSLikeFrontend(nn.Module):
    """raw EEG windows -> learnable DE-like features -> learnable smoothing."""

    def __init__(
        self,
        channels: int = 62,
        cfg: Optional[FrontendConfig] = None,
        smoother_layers: int = 2,
        dropout: float = 0.1,
        use_gcn: bool = False,
        gcn_hidden: int = 16,
        gcn_beta: float = 0.2,
        gcn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cfg = cfg or FrontendConfig()
        self.channels = int(channels)
        self.num_bands = len(cfg.bands_hz)
        self.filterbank = SincFilterbank(
            fs=cfg.fs,
            bands_hz=cfg.bands_hz,
            kernel_size=cfg.kernel_size,
            min_hz=0.5,
            max_hz=75.0,
        )
        self.de_like = DELike()
        self.spatial_gcn = (
            SpatialGCNBlock(
                channels=self.channels,
                bands=self.num_bands,
                hidden=gcn_hidden,
                beta=gcn_beta,
                dropout=gcn_dropout,
            )
            if use_gcn
            else nn.Identity()
        )
        self.out_dim = self.channels * self.num_bands
        self.smoother = TCNSmoother(dim=self.out_dim, kernel_size=5, layers=smoother_layers, dropout=dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (B, W, C, T)
        batch, windows, channels, samples = x.shape
        if channels != self.channels:
            raise ValueError(f"expected channels={self.channels}, got {channels}")

        x = x.reshape(batch * windows, channels, samples)
        y = self.filterbank(x)
        feat = self.de_like(y)
        feat = self.spatial_gcn(feat)
        feat = feat.reshape(batch, windows, -1)
        return self.smoother(feat, lengths=lengths)


class EEGConformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        channels: int = 62,
        bands: int = 5,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 6,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        smoother_layers: int = 2,
        use_gcn: bool = False,
        gcn_hidden: int = 16,
        gcn_beta: float = 0.2,
        gcn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if Conformer is None:
            raise RuntimeError("torchaudio is required: pip install torchaudio")

        self.frontend = LearnableDELDSLikeFrontend(
            channels=channels,
            cfg=FrontendConfig(bands_hz=((1, 4), (4, 8), (8, 14), (14, 31), (31, 50))),
            smoother_layers=smoother_layers,
            dropout=dropout,
            use_gcn=use_gcn,
            gcn_hidden=gcn_hidden,
            gcn_beta=gcn_beta,
            gcn_dropout=gcn_dropout,
        )
        expected_dim = channels * bands
        if self.frontend.out_dim != expected_dim:
            raise ValueError(f"frontend out_dim={self.frontend.out_dim} != channels*bands={expected_dim}")

        self.proj = nn.Linear(expected_dim, d_model)
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel,
            dropout=dropout,
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    @staticmethod
    def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, steps, _ = x.shape
        valid = torch.arange(steps, device=x.device)[None, :] < lengths[:, None]
        x = x * valid.unsqueeze(-1)
        denom = lengths.clamp_min(1).to(x.dtype).unsqueeze(-1)
        return x.sum(dim=1) / denom

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feat = self.frontend(x, lengths=lengths)
        feat = self.proj(feat)
        out, out_len = self.conformer(feat, lengths)
        pooled = self.masked_mean(out, out_len)
        return self.head(pooled)

"""Conditioning normalization layers.

Extends Abel's normalization system with style-conditioned variants
used in generative architectures (vocoders, decoders).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Dimensionality, validate_dim


class AdaIN(nn.Module):
    """Adaptive Instance Normalization conditioned on a style vector.

    Applies: ``(1 + gamma) * InstanceNorm(x) + beta``
    where ``gamma, beta`` are linearly projected from the style vector.

    Forward shapes:
        x: [B, C, L] (dim=1) or [B, C, H, W] (dim=2)
        s: [B, style_dim]
        output: same shape as x
    """

    def __init__(self, style_dim: int, num_features: int, dim: int = 1) -> None:
        super().__init__()
        self.dim: Dimensionality = validate_dim(dim)
        self.norm: nn.Module = (
            nn.InstanceNorm1d(num_features, affine=False)
            if self.dim == 1
            else nn.InstanceNorm2d(num_features, affine=False)
        )
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        # Reshape for broadcasting: [B, 2*C] -> [B, 2*C, 1] or [B, 2*C, 1, 1]
        h = h.view(h.size(0), h.size(1), *((1,) * self.dim))
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

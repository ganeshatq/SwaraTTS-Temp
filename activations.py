from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import validate_dim

ActivationName = Literal["relu", "leaky_relu", "gelu", "silu", "snake", "identity"]


class Snake(nn.Module):
    """Snake activation for periodic-rich signals.

    Forward shape:
        [B, C, L] -> [B, C, L] (1D)
        [B, C, H, W] -> [B, C, H, W] (2D)
    """

    def __init__(
        self,
        channels: int,
        dim: int,
        alpha_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        valid_dim = validate_dim(dim)
        shape = (1, channels, 1) if valid_dim == 1 else (1, channels, 1, 1)
        self.alpha = nn.Parameter(torch.full(shape, alpha_init))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.alpha) + self.eps
        return x + torch.sin(alpha * x).pow(2) / alpha


def build_activation(
    name: str,
    channels: int,
    dim: int,
    negative_slope: float = 0.2,
    inplace: bool = True,
) -> nn.Module:
    """Build activation layer by name.

    Args:
        name: One of {'relu', 'leaky_relu', 'gelu', 'silu', 'snake', 'identity'}.
        channels: Channel count (used by Snake).
        dim: Spatial dimensionality (1 or 2).
        negative_slope: Slope for LeakyReLU.
        inplace: In-place flag for applicable activations.

    Returns:
        Instantiated activation module.
    """
    key = name.lower()
    if key == "identity":
        return nn.Identity()
    if key == "relu":
        return nn.ReLU(inplace=inplace)
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU(inplace=inplace)
    if key == "snake":
        return Snake(channels=channels, dim=dim)
    raise ValueError(f"Unsupported activation '{name}'")

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .activations import build_activation
from .base import BaseBlock, build_norm, compute_same_padding
from .sampling import (
    IdentitySampling,
    StaticDownsample,
    StaticUpsample,
    TrainableDownsample,
    TrainableUpsample,
)


class ResidualBlock(BaseBlock):
    """Dimension-agnostic residual block with optional bottleneck.

    Forward shape:
        1D: [B, C_in, L] -> [B, C_out, L']
        2D: [B, C_in, H, W] -> [B, C_out, H', W']

    Notes:
        - Sampling is injected via composition (main_sampler, skip_sampler).
        - When channels differ, shortcut projection uses a 1x1 convolution.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        bottleneck: bool = False,
        expansion: int = 4,
        norm: str = "group",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        main_sampler: nn.Module | None = None,
        skip_sampler: nn.Module | None = None,
        post_activation: bool = True,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__(dim=dim, in_channels=in_channels, out_channels=out_channels)
        if expansion < 1:
            raise ValueError("expansion must be >= 1")

        self.bottleneck = bottleneck
        self.post_activation = post_activation
        self.main_sampler = main_sampler if main_sampler is not None else IdentitySampling()
        self.skip_sampler = skip_sampler if skip_sampler is not None else IdentitySampling()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.project_shortcut = in_channels != out_channels

        if bottleneck:
            if out_channels % expansion != 0:
                raise ValueError("For bottleneck=True, out_channels must be divisible by expansion")
            mid_channels = out_channels // expansion

            self.conv1 = self.factory.conv(in_channels, mid_channels, kernel_size=1, bias=False)
            self.norm1 = build_norm(norm, mid_channels, self.dim, group_norm_groups=group_norm_groups)
            self.act1 = build_activation(activation, channels=mid_channels, dim=self.dim)

            self.conv2 = self.factory.conv(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=compute_same_padding(kernel_size, 1, self.dim),
                bias=False,
            )
            self.norm2 = build_norm(norm, mid_channels, self.dim, group_norm_groups=group_norm_groups)
            self.act2 = build_activation(activation, channels=mid_channels, dim=self.dim)

            self.conv3 = self.factory.conv(mid_channels, out_channels, kernel_size=1, bias=False)
            self.norm3 = build_norm(norm, out_channels, self.dim, group_norm_groups=group_norm_groups)
        else:
            self.conv1 = self.factory.conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=compute_same_padding(kernel_size, 1, self.dim),
                bias=False,
            )
            self.norm1 = build_norm(norm, out_channels, self.dim, group_norm_groups=group_norm_groups)
            self.act1 = build_activation(activation, channels=out_channels, dim=self.dim)

            self.conv2 = self.factory.conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=compute_same_padding(kernel_size, 1, self.dim),
                bias=False,
            )
            self.norm2 = build_norm(norm, out_channels, self.dim, group_norm_groups=group_norm_groups)
            self.act2 = build_activation(activation, channels=out_channels, dim=self.dim)

        if self.project_shortcut:
            self.shortcut_conv = self.factory.conv(in_channels, out_channels, kernel_size=1, bias=False)
            self.shortcut_norm = build_norm(norm, out_channels, self.dim, group_norm_groups=group_norm_groups)
        else:
            self.shortcut_conv = nn.Identity()
            self.shortcut_norm = nn.Identity()

        self.out_act = build_activation(activation, channels=out_channels, dim=self.dim)

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.act2(x)
            x = self.dropout(x)

            x = self.conv3(x)
            x = self.norm3(x)
            return x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_sampler(x)
        identity = self.shortcut_conv(identity)
        identity = self.shortcut_norm(identity)

        out = self.main_sampler(x)
        out = self._residual(out)
        out = out + identity

        if self.post_activation:
            out = self.out_act(out)
        return out


class ResNetUpsample(ResidualBlock):
    """Residual block with static interpolation upsampling on both paths."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        mode: str = "nearest",
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            main_sampler=StaticUpsample(dim=dim, scale_factor=scale_factor, mode=mode),
            skip_sampler=StaticUpsample(dim=dim, scale_factor=scale_factor, mode=mode),
            **kwargs,
        )


class ResNetUpsampleTrain(ResidualBlock):
    """Residual block with learnable transposed-convolution upsampling."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        kernel_size: int | tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            main_sampler=TrainableUpsample(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=scale_factor,
                kernel_size=kernel_size,
                bias=False,
            ),
            skip_sampler=TrainableUpsample(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=scale_factor,
                kernel_size=kernel_size,
                bias=False,
            ),
            **kwargs,
        )


class ResNetDownsample(ResidualBlock):
    """Residual block with static pooling downsampling on both paths."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        pool_type: Literal["avg", "max"] = "avg",
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            main_sampler=StaticDownsample(dim=dim, scale_factor=scale_factor, pool_type=pool_type),
            skip_sampler=StaticDownsample(dim=dim, scale_factor=scale_factor, pool_type=pool_type),
            **kwargs,
        )


class ResNetDownsampleTrain(ResidualBlock):
    """Residual block with learnable strided-convolution downsampling."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        kernel_size: int | tuple[int, int] = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            main_sampler=TrainableDownsample(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=scale_factor,
                kernel_size=kernel_size,
                bias=False,
            ),
            skip_sampler=TrainableDownsample(
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                scale_factor=scale_factor,
                kernel_size=kernel_size,
                bias=False,
            ),
            **kwargs,
        )

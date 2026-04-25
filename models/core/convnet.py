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

SamplePosition = Literal["before", "after"]


class ConvNormActBlock(BaseBlock):
    """Reusable Conv -> Norm -> Activation block.

    Forward shape:
        1D: [B, C_in, L] -> [B, C_out, L']
        2D: [B, C_in, H, W] -> [B, C_out, H', W']
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        groups: int = 1,
        bias: bool = False,
        norm: str = "group",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__(dim=dim, in_channels=in_channels, out_channels=out_channels)
        conv_padding = compute_same_padding(kernel_size, dilation, self.dim) if padding is None else padding

        self.conv = self.factory.conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = build_norm(
            norm_type=norm,
            num_channels=out_channels,
            dim=self.dim,
            group_norm_groups=group_norm_groups,
        )
        self.activation = build_activation(activation, channels=out_channels, dim=self.dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ConvNetBlock(BaseBlock):
    """Stack of ConvNormAct blocks with pluggable sampling.

    Forward shape:
        1D: [B, C_in, L] -> [B, C_out, L']
        2D: [B, C_in, H, W] -> [B, C_out, H', W']
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        hidden_channels: int | None = None,
        kernel_size: int | tuple[int, int] = 3,
        norm: str = "group",
        activation: str = "leaky_relu",
        dropout: float = 0.0,
        sampler: nn.Module | None = None,
        sample_position: SamplePosition = "before",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__(dim=dim, in_channels=in_channels, out_channels=out_channels)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if sample_position not in {"before", "after"}:
            raise ValueError("sample_position must be 'before' or 'after'")

        hidden = out_channels if hidden_channels is None else hidden_channels
        self.sampler = sampler if sampler is not None else IdentitySampling()
        self.sample_position = sample_position

        layers: list[nn.Module] = []
        for idx in range(num_layers):
            in_ch = in_channels if idx == 0 else hidden
            out_ch = out_channels if idx == num_layers - 1 else hidden
            layers.append(
                ConvNormActBlock(
                    dim=self.dim,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    norm=norm,
                    activation=activation,
                    dropout=dropout,
                    group_norm_groups=group_norm_groups,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sample_position == "before":
            x = self.sampler(x)
            x = self.layers(x)
        else:
            x = self.layers(x)
            x = self.sampler(x)
        return x


class ConvNetUpsample(ConvNetBlock):
    """ConvNet block with static interpolation upsampling."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        mode: str = "nearest",
        **kwargs,
    ) -> None:
        sampler = StaticUpsample(dim=dim, scale_factor=scale_factor, mode=mode)
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            sampler=sampler,
            sample_position="before",
            **kwargs,
        )


class ConvNetUpsampleTrain(ConvNetBlock):
    """ConvNet block with learnable transposed-convolution upsampling."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        kernel_size: int | tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        sampler = TrainableUpsample(
            dim=dim,
            in_channels=in_channels,
            out_channels=in_channels,
            scale_factor=scale_factor,
            kernel_size=kernel_size,
            bias=False,
        )
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            sampler=sampler,
            sample_position="before",
            **kwargs,
        )


class ConvNetDownsample(ConvNetBlock):
    """ConvNet block with static pooling downsampling."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        pool_type: Literal["avg", "max"] = "avg",
        **kwargs,
    ) -> None:
        sampler = StaticDownsample(dim=dim, scale_factor=scale_factor, pool_type=pool_type)
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            sampler=sampler,
            sample_position="after",
            **kwargs,
        )


class ConvNetDownsampleTrain(ConvNetBlock):
    """ConvNet block with learnable strided-convolution downsampling."""

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int | tuple[int, int] = 2,
        kernel_size: int | tuple[int, int] = 3,
        **kwargs,
    ) -> None:
        sampler = TrainableDownsample(
            dim=dim,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            kernel_size=kernel_size,
            bias=False,
        )
        super().__init__(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            sampler=sampler,
            sample_position="after",
            **kwargs,
        )

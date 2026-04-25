from __future__ import annotations

from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LayerFactory, as_dimensional, compute_same_padding, validate_dim

PoolType = Literal["avg", "max"]


class IdentitySampling(nn.Module):
    """No-op sampler.

    Forward shape:
        [B, C, *] -> [B, C, *]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StaticUpsample(nn.Module):
    """Deterministic interpolation-based upsampling.

    Forward shape:
        1D: [B, C, L] -> [B, C, L * s]
        2D: [B, C, H, W] -> [B, C, H * s_h, W * s_w]
    """

    def __init__(
        self,
        dim: int,
        scale_factor: Union[int, tuple[int, int]] = 2,
        mode: str | None = None,
        align_corners: bool | None = None,
    ) -> None:
        super().__init__()
        self.dim = validate_dim(dim)
        self.scale_factor = as_dimensional(scale_factor, self.dim, "scale_factor")
        self.mode = mode or "nearest"
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self.mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False if self.align_corners is None else self.align_corners
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, **kwargs)


class StaticDownsample(nn.Module):
    """Deterministic pooling-based downsampling.

    Forward shape:
        1D: [B, C, L] -> [B, C, floor(L / s)]
        2D: [B, C, H, W] -> [B, C, floor(H / s_h), floor(W / s_w)]
    """

    def __init__(
        self,
        dim: int,
        scale_factor: Union[int, tuple[int, int]] = 2,
        pool_type: PoolType = "avg",
    ) -> None:
        super().__init__()
        self.dim = validate_dim(dim)
        self.pool_type = pool_type
        factory = LayerFactory(self.dim)

        if pool_type == "avg":
            self.pool = factory.avg_pool(scale_factor, stride=scale_factor)
        elif pool_type == "max":
            self.pool = factory.max_pool(scale_factor, stride=scale_factor)
        else:
            raise ValueError(f"Unsupported pool_type '{pool_type}'. Use avg or max")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


def _default_transpose_kernel(stride: Union[int, tuple[int, int]], dim: int) -> Union[int, tuple[int, int]]:
    stride_dim = as_dimensional(stride, validate_dim(dim), "stride")
    if dim == 1:
        return int(stride_dim) * 2
    st = stride_dim  # type: ignore[assignment]
    return (int(st[0]) * 2, int(st[1]) * 2)


def _default_transpose_padding(stride: Union[int, tuple[int, int]], dim: int) -> Union[int, tuple[int, int]]:
    stride_dim = as_dimensional(stride, validate_dim(dim), "stride")
    if dim == 1:
        return int(stride_dim) // 2
    st = stride_dim  # type: ignore[assignment]
    return (int(st[0]) // 2, int(st[1]) // 2)


class TrainableUpsample(nn.Module):
    """Learnable transposed-convolution upsampling.

    Forward shape:
        1D: [B, C_in, L] -> [B, C_out, L']
        2D: [B, C_in, H, W] -> [B, C_out, H', W']
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int | None = None,
        scale_factor: Union[int, tuple[int, int]] = 2,
        kernel_size: Union[int, tuple[int, int]] | None = None,
        padding: Union[int, tuple[int, int]] | None = None,
        output_padding: Union[int, tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = validate_dim(dim)
        out_channels = in_channels if out_channels is None else out_channels

        stride = as_dimensional(scale_factor, self.dim, "scale_factor")
        kernel = kernel_size if kernel_size is not None else _default_transpose_kernel(stride, self.dim)
        pad = padding if padding is not None else _default_transpose_padding(stride, self.dim)

        self.conv_t = LayerFactory(self.dim).conv_transpose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_t(x)


class TrainableDownsample(nn.Module):
    """Learnable strided-convolution downsampling.

    Forward shape:
        1D: [B, C_in, L] -> [B, C_out, L']
        2D: [B, C_in, H, W] -> [B, C_out, H', W']
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int | None = None,
        scale_factor: Union[int, tuple[int, int]] = 2,
        kernel_size: Union[int, tuple[int, int]] = 3,
        padding: Union[int, tuple[int, int]] | None = None,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.dim = validate_dim(dim)
        out_channels = in_channels if out_channels is None else out_channels

        stride = as_dimensional(scale_factor, self.dim, "scale_factor")
        pad = padding if padding is not None else compute_same_padding(kernel_size, 1, self.dim)

        self.conv = LayerFactory(self.dim).conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=bias,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

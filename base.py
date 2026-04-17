from __future__ import annotations

from abc import ABC
from typing import Literal, Sequence, Tuple, Union

import torch.nn as nn

Dimensionality = Literal[1, 2]
SizeLike = Union[int, Sequence[int]]


def validate_dim(dim: int) -> Dimensionality:
    """Validate block dimensionality.

    Args:
        dim: Spatial dimensionality, either 1 or 2.

    Returns:
        The validated dimensionality.
    """
    if dim not in (1, 2):
        raise ValueError(f"dim must be 1 or 2, got {dim}")
    return dim  # type: ignore[return-value]


def as_dimensional(value: SizeLike, dim: Dimensionality, name: str) -> Union[int, Tuple[int, int]]:
    """Normalize scalar/sequence values for 1D or 2D layers.

    Args:
        value: Scalar or sequence value to normalize.
        dim: Spatial dimensionality (1 or 2).
        name: Name of the argument for error messages.

    Returns:
        For dim=1 returns int, for dim=2 returns (int, int).
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(v) for v in value)
        if len(values) != dim:
            raise ValueError(f"{name} expects {dim} values, got {len(values)}")
        if dim == 1:
            return values[0]
        return values  # type: ignore[return-value]

    scalar = int(value)
    if dim == 1:
        return scalar
    return (scalar, scalar)


def compute_same_padding(
    kernel_size: SizeLike,
    dilation: SizeLike,
    dim: Dimensionality,
) -> Union[int, Tuple[int, int]]:
    """Compute symmetric 'same-like' padding for odd kernels.

    Args:
        kernel_size: Kernel size (int or tuple).
        dilation: Dilation (int or tuple).
        dim: Spatial dimensionality.

    Returns:
        Padding value for Conv1d/Conv2d.
    """
    k = as_dimensional(kernel_size, dim, "kernel_size")
    d = as_dimensional(dilation, dim, "dilation")

    if dim == 1:
        return ((int(k) - 1) * int(d)) // 2

    k2 = k  # type: ignore[assignment]
    d2 = d  # type: ignore[assignment]
    return tuple(((ki - 1) * di) // 2 for ki, di in zip(k2, d2))  # type: ignore[arg-type]


class LayerFactory:
    """Factory that creates dimension-aware PyTorch layers.

    This class centralizes Conv/Norm/Pool selection to avoid 1D/2D
    code duplication across higher-level blocks.
    """

    def __init__(self, dim: int) -> None:
        self.dim = validate_dim(dim)

    def conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeLike,
        stride: SizeLike = 1,
        padding: SizeLike = 0,
        dilation: SizeLike = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> nn.Module:
        conv_cls = nn.Conv1d if self.dim == 1 else nn.Conv2d
        return conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=as_dimensional(kernel_size, self.dim, "kernel_size"),
            stride=as_dimensional(stride, self.dim, "stride"),
            padding=as_dimensional(padding, self.dim, "padding"),
            dilation=as_dimensional(dilation, self.dim, "dilation"),
            groups=groups,
            bias=bias,
        )

    def conv_transpose(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: SizeLike,
        stride: SizeLike = 1,
        padding: SizeLike = 0,
        output_padding: SizeLike = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: SizeLike = 1,
    ) -> nn.Module:
        conv_t_cls = nn.ConvTranspose1d if self.dim == 1 else nn.ConvTranspose2d
        return conv_t_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=as_dimensional(kernel_size, self.dim, "kernel_size"),
            stride=as_dimensional(stride, self.dim, "stride"),
            padding=as_dimensional(padding, self.dim, "padding"),
            output_padding=as_dimensional(output_padding, self.dim, "output_padding"),
            groups=groups,
            bias=bias,
            dilation=as_dimensional(dilation, self.dim, "dilation"),
        )

    def avg_pool(self, kernel_size: SizeLike, stride: SizeLike | None = None) -> nn.Module:
        pool_cls = nn.AvgPool1d if self.dim == 1 else nn.AvgPool2d
        stride_val = kernel_size if stride is None else stride
        return pool_cls(
            kernel_size=as_dimensional(kernel_size, self.dim, "kernel_size"),
            stride=as_dimensional(stride_val, self.dim, "stride"),
        )

    def max_pool(self, kernel_size: SizeLike, stride: SizeLike | None = None) -> nn.Module:
        pool_cls = nn.MaxPool1d if self.dim == 1 else nn.MaxPool2d
        stride_val = kernel_size if stride is None else stride
        return pool_cls(
            kernel_size=as_dimensional(kernel_size, self.dim, "kernel_size"),
            stride=as_dimensional(stride_val, self.dim, "stride"),
        )


def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = min(max(1, requested_groups), num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


def build_norm(
    norm_type: str,
    num_channels: int,
    dim: int,
    eps: float = 1e-5,
    group_norm_groups: int = 8,
) -> nn.Module:
    """Create normalization layer for 1D/2D tensors.

    Args:
        norm_type: One of {'batch', 'instance', 'group', 'none'}.
        num_channels: Number of feature channels.
        dim: Spatial dimensionality (1 or 2).
        eps: Numerical epsilon.
        group_norm_groups: Requested group count for GroupNorm.

    Returns:
        Instantiated normalization module.
    """
    valid_dim = validate_dim(dim)
    key = norm_type.lower()

    if key == "none":
        return nn.Identity()
    if key == "batch":
        return nn.BatchNorm1d(num_channels, eps=eps) if valid_dim == 1 else nn.BatchNorm2d(num_channels, eps=eps)
    if key == "instance":
        return (
            nn.InstanceNorm1d(num_channels, eps=eps, affine=True)
            if valid_dim == 1
            else nn.InstanceNorm2d(num_channels, eps=eps, affine=True)
        )
    if key == "group":
        groups = _resolve_group_count(num_channels, group_norm_groups)
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels, eps=eps)

    raise ValueError(f"Unsupported norm_type '{norm_type}'. Use batch|instance|group|none")


class BaseBlock(nn.Module, ABC):
    """Shared base class for modular ConvNet/ResNet blocks."""

    def __init__(self, dim: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dim: Dimensionality = validate_dim(dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factory = LayerFactory(self.dim)

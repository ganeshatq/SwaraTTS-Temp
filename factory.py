from __future__ import annotations

from dataclasses import dataclass

from .base import validate_dim
from .convnet import (
    ConvNetBlock,
    ConvNetDownsample,
    ConvNetDownsampleTrain,
    ConvNetUpsample,
    ConvNetUpsampleTrain,
)
from .resnet import (
    ResidualBlock,
    ResNetDownsample,
    ResNetDownsampleTrain,
    ResNetUpsample,
    ResNetUpsampleTrain,
)


@dataclass(frozen=True)
class BlockFactory:
    """High-level factory to instantiate 1D/2D modular blocks.

    Example:
        factory = BlockFactory(dim=1)
        block = factory.resnet_downsample_train(in_channels=128, out_channels=256)
    """

    dim: int

    def __post_init__(self) -> None:
        validate_dim(self.dim)

    def convnet(self, in_channels: int, out_channels: int, **kwargs) -> ConvNetBlock:
        return ConvNetBlock(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)

    def convnet_upsample(self, in_channels: int, out_channels: int, trainable: bool = False, **kwargs):
        if trainable:
            return ConvNetUpsampleTrain(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)
        return ConvNetUpsample(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)

    def convnet_downsample(self, in_channels: int, out_channels: int, trainable: bool = False, **kwargs):
        if trainable:
            return ConvNetDownsampleTrain(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)
        return ConvNetDownsample(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)

    def resnet(self, in_channels: int, out_channels: int, **kwargs) -> ResidualBlock:
        return ResidualBlock(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)

    def resnet_upsample(self, in_channels: int, out_channels: int, trainable: bool = False, **kwargs):
        if trainable:
            return ResNetUpsampleTrain(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)
        return ResNetUpsample(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)

    def resnet_downsample(self, in_channels: int, out_channels: int, trainable: bool = False, **kwargs):
        if trainable:
            return ResNetDownsampleTrain(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)
        return ResNetDownsample(dim=self.dim, in_channels=in_channels, out_channels=out_channels, **kwargs)

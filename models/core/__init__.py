"""SwaraTTS 2 — modular neural TTS package.

Public API
----------
Flow-matching sampler (novel contribution replacing EDM diffusion):
    OTCFMSampler            — top-level sampler (train + infer)
    StyleVectorFieldNet     — Transformer vector-field network
    SinusoidalTimeEmbedding — time embedding helper

Decoder / vocoder:
    ISTFTDecoder            — iSTFT-Net full decoder
    ISTFTGenerator          — iSTFT-Net generator core
    AdaINResBlock           — style-conditioned ResBlock (Generator)
    AdaINResidualBlock      — style-conditioned ResBlock (Decoder)

Source excitation:
    SourceModuleHnNSF       — harmonic-plus-noise source module
    SineGen                 — sine wave generator
    TorchSTFT               — differentiable STFT wrapper

Building blocks:
    ConvNetBlock / ConvNetUpsample / ConvNetUpsampleTrain
    ConvNetDownsample / ConvNetDownsampleTrain
    ResidualBlock / ResNetUpsample / ResNetUpsampleTrain
    ResNetDownsample / ResNetDownsampleTrain
    BlockFactory            — high-level factory

Normalization:
    AdaIN                   — Adaptive Instance Normalization

Activations:
    Snake / build_activation

Sampling layers:
    StaticUpsample / StaticDownsample
    TrainableUpsample / TrainableDownsample
    IdentitySampling

SwaraTTS2 model:
    SwaraTTS2               — main model class
"""

from .activations import Snake, build_activation, ActivationName
from .base import LayerFactory, BaseBlock, build_norm, validate_dim
from .convnet import (
    ConvNormActBlock,
    ConvNetBlock,
    ConvNetUpsample,
    ConvNetUpsampleTrain,
    ConvNetDownsample,
    ConvNetDownsampleTrain,
)
from .factory import BlockFactory
from .flow_matching import OTCFMSampler, StyleVectorFieldNet, SinusoidalTimeEmbedding
from .istftnet import (
    AdaINResBlock,
    AdaINResidualBlock,
    ISTFTGenerator,
    ISTFTDecoder,
)
from .norms import AdaIN
from .resnet import (
    ResidualBlock,
    ResNetUpsample,
    ResNetUpsampleTrain,
    ResNetDownsample,
    ResNetDownsampleTrain,
)
from .sampling import (
    IdentitySampling,
    StaticUpsample,
    StaticDownsample,
    TrainableUpsample,
    TrainableDownsample,
)
from .source import TorchSTFT, SineGen, SourceModuleHnNSF
from .swara_model import SwaraTTS2

__all__ = [
    # activations
    "Snake", "build_activation", "ActivationName",
    # base
    "LayerFactory", "BaseBlock", "build_norm", "validate_dim",
    # convnet
    "ConvNormActBlock", "ConvNetBlock",
    "ConvNetUpsample", "ConvNetUpsampleTrain",
    "ConvNetDownsample", "ConvNetDownsampleTrain",
    # factory
    "BlockFactory",
    # flow matching (novelty)
    "OTCFMSampler", "StyleVectorFieldNet", "SinusoidalTimeEmbedding",
    # istftnet
    "AdaINResBlock", "AdaINResidualBlock", "ISTFTGenerator", "ISTFTDecoder",
    # norms
    "AdaIN",
    # resnet
    "ResidualBlock",
    "ResNetUpsample", "ResNetUpsampleTrain",
    "ResNetDownsample", "ResNetDownsampleTrain",
    # sampling
    "IdentitySampling", "StaticUpsample", "StaticDownsample",
    "TrainableUpsample", "TrainableDownsample",
    # source
    "TorchSTFT", "SineGen", "SourceModuleHnNSF",
    # model
    "SwaraTTS2",
]
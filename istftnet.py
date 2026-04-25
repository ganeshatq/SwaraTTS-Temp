"""iSTFT-Net Generator and Decoder.

Adapts the StyleTTS2 iSTFT-Net vocoder architecture using Abel modular
layers.  The key adaptations are:

* ``LayerFactory`` for all convolution / transposed-convolution creation
* Abel ``Snake`` activation (with softplus-guarded alpha) instead of raw
  inline sin² expressions
* ``AdaIN`` from ``norms.py`` for style conditioning
* ``compute_same_padding`` for dilation-aware padding
* ``register_buffer`` for the STFT window (automatic device transfer)
* Device-agnostic code (no hard-coded ``'cuda'``)

Modules:
    AdaINResBlock       — multi-dilation style-conditioned ResBlock (Generator)
    AdaINResidualBlock  — style-conditioned residual block (Decoder)
    ISTFTGenerator      — iSTFT-based vocoder core
    ISTFTDecoder        — full decoder wrapping the generator
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from .activations import Snake
from .base import LayerFactory, compute_same_padding, validate_dim
from .norms import AdaIN
from .sampling import StaticUpsample
from .source import SourceModuleHnNSF, TorchSTFT


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _init_conv_weights(
    module: nn.Module, mean: float = 0.0, std: float = 0.01
) -> None:
    """Initialize convolution weights with a normal distribution."""
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


# ---------------------------------------------------------------------------
# AdaINResBlock — used inside the Generator
# ---------------------------------------------------------------------------


class AdaINResBlock(nn.Module):
    """Style-conditioned residual block with multi-dilation convolutions.

    Each dilation stage applies::

        xt = Snake(AdaIN(x, s))
        xt = DilatedConv(xt)
        xt = Snake(AdaIN(xt, s))
        xt = Conv(xt)
        x  = x + xt              # residual

    This is repeated for every dilation value in ``dilations``.

    Forward shapes:
        x: [B, C, L] (dim=1) or [B, C, H, W] (dim=2)
        s: [B, style_dim]
        output: same shape as x
    """

    def __init__(
        self,
        dim: int,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
        style_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dim = validate_dim(dim)
        factory = LayerFactory(self.dim)

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.acts2 = nn.ModuleList()

        for d in dilations:
            # Dilated convolution
            self.convs1.append(
                weight_norm(
                    factory.conv(
                        channels,
                        channels,
                        kernel_size,
                        dilation=d,
                        padding=compute_same_padding(kernel_size, d, self.dim),
                    )
                )
            )
            # Unit-dilation convolution
            self.convs2.append(
                weight_norm(
                    factory.conv(
                        channels,
                        channels,
                        kernel_size,
                        dilation=1,
                        padding=compute_same_padding(kernel_size, 1, self.dim),
                    )
                )
            )
            self.norms1.append(AdaIN(style_dim, channels, dim=self.dim))
            self.norms2.append(AdaIN(style_dim, channels, dim=self.dim))
            self.acts1.append(Snake(channels, dim=self.dim))
            self.acts2.append(Snake(channels, dim=self.dim))

        self.convs1.apply(_init_conv_weights)
        self.convs2.apply(_init_conv_weights)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.norms1,
            self.norms2,
            self.acts1,
            self.acts2,
        ):
            xt = a1(n1(x, s))
            xt = c1(xt)
            xt = a2(n2(xt, s))
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


# ---------------------------------------------------------------------------
# AdaINResidualBlock — used inside the Decoder
# ---------------------------------------------------------------------------


class AdaINResidualBlock(nn.Module):
    """Style-conditioned residual block for the decoder.

    Structure::

        main:     AdaIN → Act → [Upsample] → Conv → Drop → AdaIN → Act → Conv → Drop
        shortcut: [Interpolate ×2] → [1×1 Conv]
        output:   (main + shortcut) / sqrt(2)

    When ``upsample=True`` the main path uses a learnable depthwise
    ``ConvTranspose`` (stride 2) and the shortcut uses nearest-neighbour
    interpolation (×2).

    Forward shapes:
        x: [B, C_in, L]
        s: [B, style_dim]
        output: [B, C_out, L'] where L' = 2*L if upsample else L
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        style_dim: int = 64,
        upsample: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = validate_dim(dim)
        self.has_upsample = upsample
        self.learned_sc = in_channels != out_channels
        factory = LayerFactory(self.dim)

        # -- main path --------------------------------------------------------
        self.norm1 = AdaIN(style_dim, in_channels, dim=self.dim)
        self.norm2 = AdaIN(style_dim, out_channels, dim=self.dim)
        self.actv = nn.LeakyReLU(0.2)
        self.conv1 = weight_norm(
            factory.conv(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv2 = weight_norm(
            factory.conv(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.dropout = nn.Dropout(dropout)

        # Learnable depthwise upsample on main path
        if upsample:
            self.pool = weight_norm(
                factory.conv_transpose(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    groups=in_channels,
                    padding=1,
                    output_padding=1,
                )
            )
        else:
            self.pool = nn.Identity()

        # -- shortcut path ----------------------------------------------------
        if upsample:
            self.skip_upsample: nn.Module = StaticUpsample(
                dim=self.dim, scale_factor=2, mode="nearest"
            )
        else:
            self.skip_upsample = nn.Identity()

        if self.learned_sc:
            self.conv1x1 = weight_norm(
                factory.conv(in_channels, out_channels, kernel_size=1, bias=False)
            )

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        x = self.skip_upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# ---------------------------------------------------------------------------
# ISTFTGenerator — vocoder core
# ---------------------------------------------------------------------------


class ISTFTGenerator(nn.Module):
    """iSTFT-based neural vocoder generator.

    Upsamples hidden features through multiple stages, injects F0-conditioned
    harmonic source excitation at each stage, then projects to magnitude /
    phase and reconstructs waveform via inverse STFT.

    Forward shapes:
        x:   [B, C_in, T]  — hidden features from the decoder
        s:   [B, style_dim] — style vector
        f0:  [B, T_f0]      — fundamental frequency contour
        output: [B, 1, T_audio]
    """

    def __init__(
        self,
        style_dim: int = 64,
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        upsample_rates: Sequence[int] = (10, 6),
        upsample_initial_channel: int = 512,
        resblock_dilation_sizes: Sequence[Sequence[int]] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        upsample_kernel_sizes: Sequence[int] = (20, 12),
        gen_istft_n_fft: int = 20,
        gen_istft_hop_size: int = 5,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        factory = LayerFactory(dim=1)

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.post_n_fft = gen_istft_n_fft

        total_upsample = int(np.prod(upsample_rates)) * gen_istft_hop_size

        # -- source excitation ------------------------------------------------
        self.m_source = SourceModuleHnNSF(
            sample_rate=sample_rate,
            upsample_scale=total_upsample,
            harmonic_num=8,
            voiced_threshold=10,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=float(total_upsample))

        # -- upsampling + resblocks + noise injection -------------------------
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()

        source_channels = gen_istft_n_fft + 2  # magnitude + phase channels

        for i, (rate, kernel) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes)
        ):
            ch_in = upsample_initial_channel // (2**i)
            ch_out = upsample_initial_channel // (2 ** (i + 1))

            # Learnable transposed-convolution upsample
            self.ups.append(
                weight_norm(
                    factory.conv_transpose(
                        ch_in,
                        ch_out,
                        kernel_size=kernel,
                        stride=rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )

            # Multi-kernel residual blocks for this stage
            for k_rb, d_rb in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                self.resblocks.append(
                    AdaINResBlock(
                        dim=1,
                        channels=ch_out,
                        kernel_size=k_rb,
                        dilations=tuple(d_rb),
                        style_dim=style_dim,
                    )
                )

            # Noise (source) projection for this stage
            if i + 1 < len(upsample_rates):
                stride_f0 = int(np.prod(upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    factory.conv(
                        source_channels,
                        ch_out,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(
                    AdaINResBlock(
                        dim=1,
                        channels=ch_out,
                        kernel_size=7,
                        dilations=(1, 3, 5),
                        style_dim=style_dim,
                    )
                )
            else:
                # Last stage: no temporal downsampling needed
                self.noise_convs.append(
                    factory.conv(source_channels, ch_out, kernel_size=1)
                )
                self.noise_res.append(
                    AdaINResBlock(
                        dim=1,
                        channels=ch_out,
                        kernel_size=11,
                        dilations=(1, 3, 5),
                        style_dim=style_dim,
                    )
                )

        # -- post-convolution → spectrogram -----------------------------------
        last_ch = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = weight_norm(
            factory.conv(last_ch, self.post_n_fft + 2, kernel_size=7, padding=3)
        )

        self.ups.apply(_init_conv_weights)
        self.conv_post.apply(_init_conv_weights)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(
            filter_length=gen_istft_n_fft,
            hop_length=gen_istft_hop_size,
            win_length=gen_istft_n_fft,
        )

    # -- forward --------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor
    ) -> torch.Tensor:
        """Full forward with F0-conditioned source excitation.

        Args:
            x:  [B, C, T]   — hidden features
            s:  [B, style_dim] — style vector
            f0: [B, T_f0]   — fundamental frequency contour
        Returns:
            waveform: [B, 1, T_audio]
        """
        with torch.no_grad():
            # Upsample F0 to audio rate and generate harmonic source
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_up, 1]
            har_source, _, _ = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)  # [B, T_up]
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)  # [B, n_fft+2, T']

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)

            # Source injection
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)

            # Upsample main path
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            x = x + x_source

            # Multi-kernel ResBlock averaging
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)

        # Split into magnitude spectrum and phase
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])

        return self.stft.inverse(spec, phase)

    def forward_without_source(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass without F0 source injection (inference shortcut).

        Returns:
            spec:  [B, n_fft//2 + 1, T']
            phase: [B, n_fft//2 + 1, T']
        """
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)

        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return spec, phase

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


# ---------------------------------------------------------------------------
# ISTFTDecoder — full decoder with F0 / energy processing
# ---------------------------------------------------------------------------


def _random_smooth(
    x: torch.Tensor, kernel_sizes: Sequence[int]
) -> torch.Tensor:
    """Apply random moving-average smoothing (training augmentation).

    Picks a random kernel size from ``kernel_sizes``. A size of 0 means
    no smoothing.
    """
    k = kernel_sizes[random.randint(0, len(kernel_sizes) - 1)]
    if k == 0:
        return x
    kernel = torch.ones(1, 1, k, device=x.device) / k
    return F.conv1d(x.unsqueeze(1), kernel, padding=k // 2).squeeze(1)


class ISTFTDecoder(nn.Module):
    """Full iSTFT-Net decoder.

    Processes linguistic features (ASR output), F0 contour, and energy (N)
    through a style-conditioned encoder–decoder, then synthesises waveform
    via ``ISTFTGenerator``.

    Architecture::

        [asr, F0↓, N↓] → Encode → {Decode × (n-1)} → Decode+Upsample → Generator → waveform
                                     ↑ (asr_res, F0↓, N↓) injected before each non-upsample block

    Forward shapes:
        asr:      [B, dim_in, T]
        f0_curve: [B, T]
        n:        [B, T]          — energy / loudness
        s:        [B, style_dim]
        output:   [B, 1, T_audio]
    """

    def __init__(
        self,
        dim_in: int = 512,
        style_dim: int = 64,
        hidden_dim: int = 1024,
        asr_res_dim: int = 64,
        generator_dim: int = 512,
        num_decode_blocks: int = 4,
        # --- Generator parameters ---
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        upsample_rates: Sequence[int] = (10, 6),
        upsample_initial_channel: int = 512,
        resblock_dilation_sizes: Sequence[Sequence[int]] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        upsample_kernel_sizes: Sequence[int] = (20, 12),
        gen_istft_n_fft: int = 20,
        gen_istft_hop_size: int = 5,
        sample_rate: int = 24000,
        # --- Training augmentation ---
        f0_smoothing_kernels: Sequence[int] = (0, 3, 7),
        n_smoothing_kernels: Sequence[int] = (0, 3, 7, 15),
    ) -> None:
        super().__init__()
        factory = LayerFactory(dim=1)

        self.f0_smoothing_kernels = tuple(f0_smoothing_kernels)
        self.n_smoothing_kernels = tuple(n_smoothing_kernels)

        # -- F0 / Energy downsampling (stride-2 conv: T → T/2) ---------------
        self.f0_conv = weight_norm(
            factory.conv(1, 1, kernel_size=3, stride=2, padding=1)
        )
        self.n_conv = weight_norm(
            factory.conv(1, 1, kernel_size=3, stride=2, padding=1)
        )

        # -- ASR residual projection ------------------------------------------
        self.asr_res = nn.Sequential(
            weight_norm(factory.conv(dim_in, asr_res_dim, kernel_size=1))
        )

        # -- Encoder: [asr + F0↓ + N↓] → hidden_dim --------------------------
        #   input channels = dim_in + 1 (F0) + 1 (N) = dim_in + 2
        self.encode = AdaINResidualBlock(
            dim=1,
            in_channels=dim_in + 2,
            out_channels=hidden_dim,
            style_dim=style_dim,
        )

        # -- Decoder blocks ---------------------------------------------------
        #   Blocks receive concatenated [x, asr_res, F0↓, N↓] until the
        #   first upsample block.
        #   input channels = hidden_dim + asr_res_dim + 1 (F0) + 1 (N)
        inject_channels = hidden_dim + asr_res_dim + 2
        self.decode = nn.ModuleList()
        for i in range(num_decode_blocks):
            is_last = i == num_decode_blocks - 1
            out_ch = generator_dim if is_last else hidden_dim
            self.decode.append(
                AdaINResidualBlock(
                    dim=1,
                    in_channels=inject_channels,
                    out_channels=out_ch,
                    style_dim=style_dim,
                    upsample=is_last,
                )
            )

        # -- Generator --------------------------------------------------------
        self.generator = ISTFTGenerator(
            style_dim=style_dim,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
            gen_istft_hop_size=gen_istft_hop_size,
            sample_rate=sample_rate,
        )

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        n: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            asr:      [B, dim_in, T]  — linguistic features
            f0_curve: [B, T]          — F0 contour
            n:        [B, T]          — energy / loudness
            s:        [B, style_dim]  — style vector
        Returns:
            waveform: [B, 1, T_audio]
        """
        # Training-time smoothing augmentation
        if self.training:
            f0_curve = _random_smooth(f0_curve, self.f0_smoothing_kernels)
            n = _random_smooth(n, self.n_smoothing_kernels)

        # Downsample F0 and N by 2× via strided conv
        f0 = self.f0_conv(f0_curve.unsqueeze(1))  # [B, 1, T/2]
        n_feat = self.n_conv(n.unsqueeze(1))  # [B, 1, T/2]
        target_len = f0.shape[-1]

        # Keep the decoder operating at the same half-rate time axis for all
        # conditioning paths before the final upsample block restores frame rate.
        if asr.shape[-1] != target_len:
            asr = F.interpolate(
                asr,
                size=target_len,
                mode="linear",
                align_corners=False,
            )

        # Encode
        x = torch.cat([asr, f0, n_feat], dim=1)  # [B, dim_in+2, T/2]
        x = self.encode(x, s)  # [B, hidden_dim, T/2]

        # ASR residual projection
        asr_res = self.asr_res(asr)  # [B, asr_res_dim, T/2]

        # Decode with residual injection
        inject_residual = True
        for block in self.decode:
            if inject_residual:
                x = torch.cat([x, asr_res, f0, n_feat], dim=1)
            x = block(x, s)
            if block.has_upsample:
                inject_residual = False

        # Generate waveform
        return self.generator(x, s, f0_curve)

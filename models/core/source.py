"""Source excitation modules for neural vocoder synthesis.

Provides harmonic-plus-noise source signals conditioned on F0 contour,
used as excitation input to neural vocoders (iSTFTNet, HiFi-GAN).

Modules:
    TorchSTFT         — differentiable STFT / inverse-STFT wrapper
    SineGen            — sine waveform generator from F0
    SourceModuleHnNSF  — harmonic + noise source module
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window


class TorchSTFT(nn.Module):
    """Differentiable STFT / inverse-STFT wrapper.

    The analysis window is stored as a registered buffer so it follows
    ``model.to(device)`` calls automatically.

    Forward shapes:
        input_data: [B, T]
        output (via inverse): [B, 1, T]
    """

    def __init__(
        self,
        filter_length: int = 800,
        hop_length: int = 200,
        win_length: int | None = None,
        window: str = "hann",
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length or filter_length
        self.register_buffer(
            "window",
            torch.from_numpy(
                get_window(window, self.win_length, fftbins=True).astype(np.float32)
            ),
        )

    def transform(
        self, input_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward STFT.

        Returns:
            magnitude: [B, n_fft//2 + 1, T']
            phase:     [B, n_fft//2 + 1, T']
        """
        spec = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
            return_complex=True,
        )
        return torch.abs(spec), torch.angle(spec)

    def inverse(
        self, magnitude: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        """Inverse STFT from magnitude and phase.

        Returns:
            waveform: [B, 1, T] (unsqueezed to match conv output convention)
        """
        waveform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self.window,
        )
        return waveform.unsqueeze(-2)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Round-trip: STFT then iSTFT (identity under perfect reconstruction)."""
        magnitude, phase = self.transform(input_data)
        return self.inverse(magnitude, phase)


# ---------------------------------------------------------------------------
# Sine generator
# ---------------------------------------------------------------------------


class SineGen(nn.Module):
    """Sine waveform generator from F0 contour.

    Generates sine waves at the fundamental frequency and optional harmonic
    overtones, with proper phase accumulation and voiced/unvoiced handling.

    Phase is accumulated at a lower temporal resolution (downsampled by
    ``upsample_scale``) then interpolated back up, which reduces numerical
    drift in cumulative-sum operations.

    Forward shapes:
        f0:          [B, T, 1] — F0 in Hz (0 for unvoiced frames)
    Returns:
        sine_waves:  [B, T, num_harmonics + 1]
        uv:          [B, T, 1]  — voiced/unvoiced binary mask
        noise:       [B, T, num_harmonics + 1]
    """

    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        flag_for_pulse: bool = False,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.num_harmonics = harmonic_num + 1
        self.sample_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    # -- helpers --------------------------------------------------------------

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        """Binary voiced/unvoiced mask from F0."""
        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values: torch.Tensor) -> torch.Tensor:
        """Convert multi-harmonic F0 values to sine waveforms.

        Args:
            f0_values: [B, T, dim] where dim = num_harmonics
        """
        rad_values = (f0_values / self.sample_rate) % 1

        # Random initial phase (keep fundamental deterministic)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        if not self.flag_for_pulse:
            # --- normal mode: downsample → cumsum → upsample ---
            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)

            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi

            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)

            sines = torch.sin(phase)
        else:
            # --- pulse-train mode: reset phase at voiced onsets ---
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)

        return sines

    # -- forward --------------------------------------------------------------

    def forward(
        self, f0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build harmonic frequencies: f0, 2*f0, …, (H+1)*f0
        harmonics = torch.arange(
            1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype
        ).view(1, 1, -1)
        fn = f0 * harmonics

        sine_waves = self._f02sine(fn) * self.sine_amp

        uv = self._f02uv(f0)

        # Voiced regions get low noise, unvoiced get higher noise
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # Zero out sine in unvoiced, add noise everywhere
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


# ---------------------------------------------------------------------------
# Harmonic-plus-Noise source module
# ---------------------------------------------------------------------------


class SourceModuleHnNSF(nn.Module):
    """Harmonic-plus-Noise Source Module for neural vocoders.

    Generates excitation signals by:
    1. Creating multi-harmonic sine waves from F0 via ``SineGen``
    2. Merging harmonics into a single channel via a learned linear layer
    3. Producing additive noise for the noise branch

    Forward shapes:
        f0:          [B, T, 1]
    Returns:
        sine_merge:  [B, T, 1] — merged harmonic source
        noise:       [B, T, 1] — noise source
        uv:          [B, T, 1] — voiced/unvoiced mask
    """

    def __init__(
        self,
        sample_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std

        self.l_sin_gen = SineGen(
            sample_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(
        self, f0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(f0)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv

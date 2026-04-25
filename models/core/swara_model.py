"""SwaraTTS 2 — Main model class.

Replaces the EDM diffusion style sampler from StyleTTS 2 with an
OT-CFM (Optimal Transport Conditional Flow Matching) sampler, following
the Matcha-TTS formulation (arXiv:2309.03199).

Architecture overview
---------------------
SwaraTTS 2 keeps the full StyleTTS 2 pipeline intact:

    Text Encoder (acoustic)         → h_text       [B, N, 512]
    Text Encoder (prosodic / BERT)  → h_bert        [B, N, 768]
    Style Encoder (acoustic)        → s_a           [B, 128]
    Style Encoder (prosodic)        → s_p           [B, 128]
    Pitch Extractor                 → p_x           [B, T]
    Duration Predictor              → d_pred        [B, N]
    Prosody Predictor               → p_hat, n_hat  [B, T]
    Decoder (iSTFT or HiFi-GAN)     → waveform      [B, 1, T_audio]

The ONE change vs. StyleTTS 2: the style diffusion denoiser (EDM
transformer) is replaced by ``OTCFMSampler``.

    # StyleTTS 2  (EDM):
    style = edm_denoiser.sample(h_bert, speaker_emb, n_steps=5)
    loss_style = edm_denoiser.compute_loss(style_enc(x), h_bert)

    # SwaraTTS 2  (OT-CFM):
    style = flow_sampler.sample(h_bert, speaker_emb, n_timesteps=10)
    loss_style = flow_sampler.compute_loss(style_enc(x), h_bert)

Both paths produce an identical output tensor: (B, style_dim=256).
The style vector is split as:
    s_a = style[:, :style_dim//2]   — acoustic style
    s_p = style[:, style_dim//2:]   — prosodic style

Integration with the training loop
------------------------------------
Pre-training phase (acoustic modules):
    loss = Lmel + Ladv + Lfm + Ls2s + Lmono
    # OTCFMSampler is NOT used here; style comes from the style encoder.

Joint training phase:
    loss_fm  = flow_sampler.compute_loss(s_real, h_bert, speaker_emb)
    loss_total = Lmel + Ladv + Lfm + Ldur + Lf0 + Ln + Ls2s + Lmono
               + loss_fm   # ← replaces Ledm
               + Lslm      # SLM adversarial (unchanged)

Inference (single-speaker):
    style = flow_sampler.sample(h_bert, n_timesteps=10)
    s_a, s_p = style[:, :128], style[:, 128:]
    waveform  = decoder(h_text @ a_pred, p_hat, n_hat, s_a)

Inference (multi-speaker):
    ref_style = style_encoder(x_ref)    # [B, 256]
    style     = flow_sampler.sample(h_bert, speaker_emb=ref_style, n_timesteps=10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .flow_matching import OTCFMSampler


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class SwaraTTS2Config:
    """Hyper-parameters for SwaraTTS 2.

    All dimensions match the original StyleTTS 2 defaults unless noted.
    """

    # Style / latent
    style_dim: int = 256            # total style vector (s_a + s_p concatenated)

    # Flow matching sampler (replaces EDM denoiser)
    fm_text_dim: int = 768          # PL-BERT output dim
    fm_hidden_dim: int = 256        # Transformer hidden dim inside sampler
    fm_n_heads: int = 4             # Attention heads
    fm_n_layers: int = 3            # Transformer layers (StyleTTS 2 used 3)
    fm_time_embed_dim: int = 128    # Sinusoidal time embedding size
    fm_sigma_min: float = 1e-4      # OT-CFM noise floor (Matcha-TTS default)
    fm_dropout: float = 0.1         # Dropout inside the Transformer

    # Inference defaults
    fm_n_timesteps: int = 10        # Euler steps at inference (vs. EDM's 5)
    fm_temperature: float = 1.0     # Initial noise scale (0.667 = Matcha-TTS)

    # Long-form generation
    longform_alpha: float = 0.7     # Style interpolation weight (Algorithm 1 analogue)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class SwaraTTS2(nn.Module):
    """SwaraTTS 2 model with OT-CFM flow-matching style sampler.

    This class owns the ``OTCFMSampler`` and exposes the three methods
    needed by a training loop and inference pipeline:

        ``compute_style_loss``  — returns the OT-CFM loss (replaces Ledm)
        ``sample_style``        — generates style vectors at inference
        ``split_style``         — splits the joint vector into (s_a, s_p)

    The rest of the StyleTTS 2 components (encoders, decoders, predictors,
    discriminators) are architecture-specific and are NOT instantiated here,
    because they depend on your exact encoder/decoder choice (iSTFTNet vs.
    HiFi-GAN) and pre-trained checkpoints.  This class is designed to be
    composed into a larger ``nn.Module`` that owns those components.

    Typical usage::

        class FullModel(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.swara   = SwaraTTS2(cfg)
                self.encoder = AcousticTextEncoder(...)
                self.bert    = ProsodictTextEncoder(...)  # PL-BERT
                self.style_enc_a = StyleEncoder(...)
                self.style_enc_p = StyleEncoder(...)
                self.decoder = ISTFTDecoder(...)
                ...

            def training_step(self, batch):
                ...
                s_real = torch.cat([
                    self.style_enc_a(x),   # [B, 128]
                    self.style_enc_p(x),   # [B, 128]
                ], dim=1)                  # [B, 256]

                # Flow matching loss (replaces Ledm)
                loss_fm = self.swara.compute_style_loss(
                    s_real, h_bert, speaker_emb=speaker_emb, text_mask=mask
                )
                ...
                return loss_fm + ...

            @torch.no_grad()
            def inference(self, h_bert, speaker_emb=None):
                style = self.swara.sample_style(h_bert, speaker_emb)
                s_a, s_p = self.swara.split_style(style)
                waveform = self.decoder(h_text @ a_pred, p_hat, n_hat, s_a)
                return waveform
    """

    def __init__(self, config: Optional[SwaraTTS2Config] = None) -> None:
        super().__init__()
        self.cfg = config or SwaraTTS2Config()

        self.flow_sampler = OTCFMSampler(
            style_dim=self.cfg.style_dim,
            text_dim=self.cfg.fm_text_dim,
            hidden_dim=self.cfg.fm_hidden_dim,
            n_heads=self.cfg.fm_n_heads,
            n_layers=self.cfg.fm_n_layers,
            time_embed_dim=self.cfg.fm_time_embed_dim,
            sigma_min=self.cfg.fm_sigma_min,
            dropout=self.cfg.fm_dropout,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_style_loss(
        self,
        s_real: Tensor,
        h_bert: Tensor,
        speaker_emb: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the OT-CFM style loss.

        Replaces ``Ledm`` in the StyleTTS 2 joint training objective.

        This is added once to the total training loss:
            loss_total = Lmel + Ladv + Lfm + ... + compute_style_loss(...)

        Args:
            s_real:      (B, style_dim)    Real style vectors from the style
                                           encoders: cat([s_a, s_p], dim=1).
            h_bert:      (B, N, 768)       PL-BERT phoneme embeddings.
            speaker_emb: (B, style_dim)    Optional reference speaker vector
                                           for multi-speaker training.
            text_mask:   (B, N) bool       True = padding position.

        Returns:
            loss: scalar MSE tensor, backprop-ready.
        """
        return self.flow_sampler.compute_loss(
            x1=s_real,
            h_text=h_bert,
            speaker_emb=speaker_emb,
            text_mask=text_mask,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_style(
        self,
        h_bert: Tensor,
        speaker_emb: Optional[Tensor] = None,
        n_timesteps: Optional[int] = None,
        temperature: Optional[float] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate a style vector from text (and optionally a reference speaker).

        Drop-in replacement for ``edm_sampler.sample(h_bert, ...)``.

        Args:
            h_bert:      (B, N, 768)       PL-BERT phoneme embeddings.
            speaker_emb: (B, style_dim)    Optional speaker reference.
            n_timesteps: int               Euler ODE steps (default from config).
            temperature: float             Initial noise scale (default from config).
            text_mask:   (B, N) bool       True = padding.

        Returns:
            style: (B, style_dim=256) — same shape as EDM sampler output.
        """
        steps = n_timesteps if n_timesteps is not None else self.cfg.fm_n_timesteps
        temp  = temperature  if temperature  is not None else self.cfg.fm_temperature

        return self.flow_sampler.sample(
            h_text=h_bert,
            speaker_emb=speaker_emb,
            n_timesteps=steps,
            temperature=temp,
            text_mask=text_mask,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def split_style(self, style: Tensor) -> Tuple[Tensor, Tensor]:
        """Split joint style vector into acoustic and prosodic halves.

        Args:
            style: (B, style_dim)   e.g. style_dim = 256

        Returns:
            s_a: (B, style_dim // 2)   acoustic style  → fed to decoder
            s_p: (B, style_dim // 2)   prosodic style  → fed to predictors
        """
        half = self.cfg.style_dim // 2
        return style[:, :half], style[:, half:]

    @torch.no_grad()
    def sample_style_longform(
        self,
        h_bert_list: list[Tensor],
        speaker_emb: Optional[Tensor] = None,
        alpha: Optional[float] = None,
        n_timesteps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> list[Tensor]:
        """Generate consistent style vectors for a sequence of sentences.

        Mirrors StyleTTS 2 Algorithm 1 (Appendix B.3) but uses the flow
        sampler instead of EDM. Each sentence's style is a convex mix of
        the current sample and the previous sentence's style.

            s_curr = alpha * s_curr + (1 - alpha) * s_prev

        Args:
            h_bert_list: List of (1, N_i, 768) tensors, one per sentence.
            speaker_emb: (1, style_dim) optional speaker reference.
            alpha:       Interpolation weight in [0, 1] (default from config).
            n_timesteps: Euler ODE steps (default from config).
            temperature: Noise scale (default from config).

        Returns:
            List of (1, style_dim) style tensors, one per sentence.
        """
        alpha_val = alpha if alpha is not None else self.cfg.longform_alpha
        steps     = n_timesteps if n_timesteps is not None else self.cfg.fm_n_timesteps
        temp      = temperature  if temperature  is not None else self.cfg.fm_temperature

        styles: list[Tensor] = []
        s_prev: Optional[Tensor] = None

        for h_bert in h_bert_list:
            s_curr = self.flow_sampler.sample(
                h_text=h_bert,
                speaker_emb=speaker_emb,
                n_timesteps=steps,
                temperature=temp,
            )

            if s_prev is not None:
                s_curr = alpha_val * s_curr + (1.0 - alpha_val) * s_prev

            styles.append(s_curr)
            s_prev = s_curr

        return styles

    def __repr__(self) -> str:
        cfg = self.cfg
        return (
            f"SwaraTTS2(\n"
            f"  style_dim={cfg.style_dim},\n"
            f"  flow_sampler=OTCFMSampler(\n"
            f"    text_dim={cfg.fm_text_dim}, hidden_dim={cfg.fm_hidden_dim},\n"
            f"    n_heads={cfg.fm_n_heads}, n_layers={cfg.fm_n_layers},\n"
            f"    sigma_min={cfg.fm_sigma_min}, dropout={cfg.fm_dropout}\n"
            f"  ),\n"
            f"  inference: n_timesteps={cfg.fm_n_timesteps}, "
            f"temperature={cfg.fm_temperature}\n"
            f")"
        )

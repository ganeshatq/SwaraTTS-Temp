"""
flow_matching_sampler.py — OT-CFM Style Sampler for SwaraTTS
=============================================================

Implements Optimal Transport Conditional Flow Matching (OT-CFM) as
described in Matcha-TTS (arXiv:2309.03199) to generate style vectors
conditioned on text (PL-BERT phoneme embeddings) and an optional speaker
embedding.

Replaces the EDM diffusion sampler from the original StyleTTS 2.

Integration note (training loop):
----------------------------------
# OLD (StyleTTS 2 diffusion):
# style = edm_sampler.sample(h_bert, speaker_emb)
# loss_edm = edm_sampler.compute_loss(style_encoder(x), h_bert)

# NEW (SwaraTTS flow matching):
# style = flow_sampler.sample(h_bert, speaker_emb, n_timesteps=10)
# loss_fm = flow_sampler.compute_loss(style_encoder(x), h_bert)
# Add loss_fm to total training loss

# Output shape is identical: (B, style_dim=256)
# Split as: s_a = style[:, :128],  s_p = style[:, 128:]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. SinusoidalTimeEmbedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Convert scalar flow-time t ∈ [0, 1] to a sinusoidal embedding.

    Uses the standard transformer sinusoidal position encoding adapted for
    continuous time: half the channels encode sin, half encode cos.

    Args:
        dim: Output embedding dimensionality (must be even).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalTimeEmbedding requires even dim, got {dim}")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Encode a batch of time scalars.

        Args:
            t: Shape (B,), values in [0, 1].

        Returns:
            Tensor of shape (B, dim).
        """
        half = self.dim // 2
        # frequencies: 10000^(2i / dim)
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )  # (half,)
        # outer product: (B, half)
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, dim)
        return embedding

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


# ---------------------------------------------------------------------------
# 2. StyleVectorFieldNet
# ---------------------------------------------------------------------------

class StyleVectorFieldNet(nn.Module):
    """Transformer network that predicts the OT-CFM vector field v_t.

    Given a noisy style vector x_t, a flow time t, and PL-BERT text
    embeddings, outputs the predicted vector field v_t(x_t | text).

    Architecture:
        1. Embed t with SinusoidalTimeEmbedding → project to hidden_dim.
        2. Project x_t to hidden_dim.
        3. Sum time + x_t projections to form a single *query token*.
        4. Optionally add projected speaker embedding to the query token.
        5. Project h_text to hidden_dim.
        6. Prepend the query token to the text sequence → (B, N+1, hidden_dim).
        7. Run through a TransformerEncoder.
        8. Extract the first token (index 0) → project to style_dim.

    Args:
        style_dim:      Dimension of the style vector (default 256).
        text_dim:       Dimension of PL-BERT phoneme embeddings (default 768).
        hidden_dim:     Transformer hidden dimension (default 256).
        n_heads:        Number of attention heads (default 4).
        n_layers:       Number of transformer encoder layers (default 3).
        time_embed_dim: Sinusoidal time embedding dimension (default 128).
        dropout:        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        style_dim: int = 256,
        text_dim: int = 768,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        time_embed_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout

        # Time embedding + projection
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)

        # Style vector projection
        self.style_proj = nn.Linear(style_dim, hidden_dim)

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Optional speaker embedding projection (created lazily to keep
        # the constructor clean; always allocated so parameters are tracked)
        self.speaker_proj = nn.Linear(style_dim, hidden_dim, bias=False)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, style_dim)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        h_text: Tensor,
        speaker_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict the vector field v_t.

        Args:
            x_t:         (B, style_dim)     noisy style vector at time t.
            t:           (B,)               flow time in [0, 1].
            h_text:      (B, N, text_dim)   PL-BERT phoneme embeddings.
            speaker_emb: (B, style_dim)     optional speaker embedding.

        Returns:
            v_t: (B, style_dim) predicted vector field.
        """
        # a) Sinusoidal time embedding + project to hidden_dim
        t_emb = self.time_embed(t)            # (B, time_embed_dim)
        t_hidden = self.time_proj(t_emb)      # (B, hidden_dim)

        # b) Project noisy style vector
        x_hidden = self.style_proj(x_t)       # (B, hidden_dim)

        # c) Query token = sum of time and style projections
        query = t_hidden + x_hidden           # (B, hidden_dim)

        # d) Add projected speaker embedding if provided
        if speaker_emb is not None:
            query = query + self.speaker_proj(speaker_emb)  # (B, hidden_dim)

        query = query.unsqueeze(1)            # (B, 1, hidden_dim)

        # e) Project text tokens
        text_hidden = self.text_proj(h_text)  # (B, N, hidden_dim)

        # f) Prepend query token to text sequence
        seq = torch.cat([query, text_hidden], dim=1)  # (B, N+1, hidden_dim)

        # g) Transformer encoder (batch_first=True)
        out = self.transformer(seq)           # (B, N+1, hidden_dim)

        # h) Take the first token as the style output
        style_token = out[:, 0, :]            # (B, hidden_dim)

        # i) Project back to style_dim
        v_t = self.out_proj(style_token)      # (B, style_dim)
        return v_t

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"style_dim={self.style_dim}, "
            f"text_dim={self.text_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, "
            f"time_embed_dim={self.time_embed_dim}, "
            f"dropout={self.dropout})"
        )


# ---------------------------------------------------------------------------
# 3. OTCFMSampler
# ---------------------------------------------------------------------------

class OTCFMSampler(nn.Module):
    """OT-CFM style sampler for SwaraTTS.

    Top-level module that wraps :class:`StyleVectorFieldNet` and exposes
    training (``compute_loss``) and inference (``sample``) interfaces.

    The OT-CFM interpolation follows Matcha-TTS (arXiv:2309.03199):
        x_t   = (1 − (1 − σ_min) · t) · x0  +  t · x1
        u_t   = x1 − (1 − σ_min) · x0          (target vector field)
        loss  = MSE(v_t, u_t)

    Args:
        style_dim:      Dimension of the style vector (default 256).
        text_dim:       Dimension of PL-BERT phoneme embeddings (default 768).
        hidden_dim:     Transformer hidden dimension (default 256).
        n_heads:        Number of attention heads (default 4).
        n_layers:       Number of transformer encoder layers (default 3).
        time_embed_dim: Sinusoidal time embedding dimension (default 128).
        sigma_min:      Noise floor (default 1e-4, same as Matcha-TTS).
        dropout:        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        style_dim: int = 256,
        text_dim: int = 768,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        time_embed_dim: int = 128,
        sigma_min: float = 1e-4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.sigma_min = sigma_min

        self.net = StyleVectorFieldNet(
            style_dim=style_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x1: Tensor,
        h_text: Tensor,
        speaker_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the OT-CFM training loss.

        Samples random noise x0 and time t, constructs the interpolated
        path x_t, and computes MSE between the predicted and target vector
        field.

        Args:
            x1:          (B, style_dim)    real style vectors from the encoder.
            h_text:      (B, N, text_dim)  PL-BERT phoneme embeddings.
            speaker_emb: (B, style_dim)    optional speaker embedding.

        Returns:
            loss: scalar tensor (MSE between predicted and target vector field).
        """
        B = x1.size(0)

        # 1. Sample source noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 2. Sample time t ~ Uniform(0, 1)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)

        # 3. Interpolate: x_t = (1 - (1 - σ_min)*t)*x0 + t*x1
        t_col = t.unsqueeze(1)  # (B, 1) for broadcasting with (B, style_dim)
        x_t = (1.0 - (1.0 - self.sigma_min) * t_col) * x0 + t_col * x1

        # 4. Target vector field: u_t = x1 - (1 - σ_min)*x0
        u_t = x1 - (1.0 - self.sigma_min) * x0

        # 5. Predict vector field
        v_t = self.net(x_t, t, h_text, speaker_emb)

        # 6. MSE loss
        return F.mse_loss(v_t, u_t)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        h_text: Tensor,
        speaker_emb: Optional[Tensor] = None,
        n_timesteps: int = 10,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate style vectors from text using the Euler ODE solver.

        Args:
            h_text:       (B, N, text_dim)  PL-BERT phoneme embeddings.
            speaker_emb:  (B, style_dim)    optional speaker embedding.
            n_timesteps:  int               number of Euler steps (2–50).
            temperature:  float             scale of the initial noise (0.667
                                            is typical for Matcha-TTS).

        Returns:
            x1_hat: (B, style_dim) generated style vectors.
        """
        B = h_text.size(0)
        device = h_text.device
        dtype = h_text.dtype

        # 1. Start from scaled Gaussian noise
        x = torch.randn(B, self.style_dim, device=device, dtype=dtype) * temperature

        dt = 1.0 / n_timesteps

        # 2. Euler integration
        for t_idx in range(n_timesteps):
            t_val = t_idx / n_timesteps
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.net(x, t, h_text, speaker_emb)
            x = x + v * dt

        return x

    # ------------------------------------------------------------------
    # forward (training alias)
    # ------------------------------------------------------------------

    def forward(
        self,
        x1: Tensor,
        h_text: Tensor,
        speaker_emb: Optional[Tensor] = None,
    ) -> Tensor:
        """Alias for ``compute_loss`` for use in training loops.

        Args:
            x1:          (B, style_dim)    real style vectors from the encoder.
            h_text:      (B, N, text_dim)  PL-BERT phoneme embeddings.
            speaker_emb: (B, style_dim)    optional speaker embedding.

        Returns:
            loss: scalar MSE tensor.
        """
        return self.compute_loss(x1, h_text, speaker_emb)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"style_dim={self.style_dim}, "
            f"sigma_min={self.sigma_min}, "
            f"net={self.net!r})"
        )


# ---------------------------------------------------------------------------
# Self-contained tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running OT-CFM sampler tests …\n")

    torch.manual_seed(0)

    B, N, STYLE_DIM, TEXT_DIM = 4, 20, 256, 768

    x1 = torch.randn(B, STYLE_DIM)
    h_text = torch.randn(B, N, TEXT_DIM)
    sampler = OTCFMSampler()

    # ------------------------------------------------------------------
    # Test 1 — Forward pass / compute_loss
    # ------------------------------------------------------------------
    loss = sampler.compute_loss(x1, h_text)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"[Test 1] Loss: {loss.item():.4f}  ✓")

    # ------------------------------------------------------------------
    # Test 2 — Sampling (inference)
    # ------------------------------------------------------------------
    style = sampler.sample(h_text, n_timesteps=10)
    assert style.shape == (B, STYLE_DIM), f"Expected ({B}, {STYLE_DIM}), got {style.shape}"
    assert not torch.isnan(style).any(), "Style contains NaN"
    print(f"[Test 2] Style mean: {style.mean():.4f}, std: {style.std():.4f}  ✓")

    # ------------------------------------------------------------------
    # Test 3 — Multi-speaker (with speaker embedding)
    # ------------------------------------------------------------------
    speaker_emb = torch.randn(B, STYLE_DIM)
    style_ms = sampler.sample(h_text, speaker_emb=speaker_emb, n_timesteps=5)
    assert style_ms.shape == (B, STYLE_DIM), f"Expected ({B}, {STYLE_DIM}), got {style_ms.shape}"
    assert not torch.isnan(style_ms).any(), "Multi-speaker style contains NaN"
    print("[Test 3] Multi-speaker test passed  ✓")

    # ------------------------------------------------------------------
    # Test 4 — Gradient check (loss must be differentiable)
    # ------------------------------------------------------------------
    # Re-enable grad for this test (sample() uses no_grad)
    loss_grad = sampler.compute_loss(x1, h_text)
    loss_grad.backward()
    total_grad_norm = sum(
        p.grad.norm().item()
        for p in sampler.parameters()
        if p.grad is not None
    )
    assert total_grad_norm > 0, "Gradient norm is zero — no gradients flowed"
    print(f"[Test 4] Gradient norm: {total_grad_norm:.4f}  ✓")

    # ------------------------------------------------------------------
    # Test 5 — Different n_timesteps give different outputs
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    s1 = sampler.sample(h_text, n_timesteps=2)
    s2 = sampler.sample(h_text, n_timesteps=10)
    assert not torch.allclose(s1, s2), "Different timesteps gave identical samples"
    print("[Test 5] Different timesteps give different samples  ✓")

    print("\nALL TESTS PASSED")

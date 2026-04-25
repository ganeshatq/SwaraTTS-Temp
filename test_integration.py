"""
test_integration.py — SwaraTTS 2 Flow Matching Integration Tests
================================================================

Run with:
    cd <project_root>          # the folder that CONTAINS the swara_tts/ package
    python test_integration.py

All tests use CPU and synthetic tensors — no real data or GPU needed.
"""

import sys, os
# Allow running from either the project root or the package directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_HERE, _PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

# -----------------------------------------------------------------
# Import the package
# -----------------------------------------------------------------
try:
    from swara_tts import (
        OTCFMSampler,
        StyleVectorFieldNet,
        SinusoidalTimeEmbedding,
        SwaraTTS2,
        ISTFTDecoder,
        AdaIN,
    )
    from swara_tts.swara_model import SwaraTTS2Config
    print("✓  Package import OK\n")
except ImportError as e:
    print(f"✗  Import failed: {e}")
    raise

DEVICE = torch.device("cpu")
B      = 2          # batch size
N      = 15         # phoneme sequence length
STYLE  = 256        # style dim
TEXT   = 768        # PL-BERT dim


# =================================================================
# TEST 1: OTCFMSampler standalone
# =================================================================
def test_1_sampler_standalone():
    print("=" * 55)
    print("TEST 1 — OTCFMSampler standalone")
    print("=" * 55)

    sampler = OTCFMSampler(style_dim=STYLE, text_dim=TEXT).to(DEVICE)

    x1     = torch.randn(B, STYLE)
    h_bert = torch.randn(B, N, TEXT)

    # 1a. Training loss
    loss = sampler.compute_loss(x1, h_bert)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    print(f"  [1a] Loss = {loss.item():.4f}  ✓")

    # 1b. Backprop
    loss.backward()
    total_norm = sum(
        p.grad.norm().item() for p in sampler.parameters() if p.grad is not None
    )
    assert total_norm > 0, "Zero gradient — backprop broken"
    print(f"  [1b] Gradient norm = {total_norm:.4f}  ✓")

    # 1c. Sample (inference)
    style = sampler.sample(h_bert, n_timesteps=5)
    assert style.shape == (B, STYLE), f"Bad shape: {style.shape}"
    assert not torch.isnan(style).any(), "Style NaN"
    print(f"  [1c] style shape = {tuple(style.shape)}  ✓")

    # 1d. Padding mask
    sampler.zero_grad()
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, -3:] = True   # last 3 tokens of sample 0 are padding
    loss_masked = sampler.compute_loss(x1, h_bert, text_mask=mask)
    loss_masked.backward()
    print(f"  [1d] Padded-mask loss = {loss_masked.item():.4f}  ✓")

    print()


# =================================================================
# TEST 2: SwaraTTS2 model
# =================================================================
def test_2_swara_model():
    print("=" * 55)
    print("TEST 2 — SwaraTTS2 model")
    print("=" * 55)

    cfg   = SwaraTTS2Config(fm_n_layers=2, fm_n_timesteps=5)
    model = SwaraTTS2(cfg).to(DEVICE)
    print(f"  Model:\n  {model}\n")

    h_bert = torch.randn(B, N, TEXT)
    x_real = torch.randn(B, STYLE)

    # 2a. compute_style_loss
    loss = model.compute_style_loss(x_real, h_bert)
    assert not torch.isnan(loss), "Loss NaN"
    loss.backward()
    print(f"  [2a] compute_style_loss = {loss.item():.4f}  ✓")

    # 2b. sample_style
    style = model.sample_style(h_bert)
    assert style.shape == (B, STYLE)
    print(f"  [2b] sample_style shape = {tuple(style.shape)}  ✓")

    # 2c. split_style
    s_a, s_p = model.split_style(style)
    assert s_a.shape == (B, STYLE // 2)
    assert s_p.shape == (B, STYLE // 2)
    print(f"  [2c] split → s_a {tuple(s_a.shape)}, s_p {tuple(s_p.shape)}  ✓")

    # 2d. multi-speaker
    spk = torch.randn(B, STYLE)
    style_ms = model.sample_style(h_bert, speaker_emb=spk)
    assert style_ms.shape == (B, STYLE)
    print(f"  [2d] multi-speaker style = {tuple(style_ms.shape)}  ✓")

    print()


# =================================================================
# TEST 3: Long-form generation
# =================================================================
def test_3_longform():
    print("=" * 55)
    print("TEST 3 — Long-form generation")
    print("=" * 55)

    model = SwaraTTS2(SwaraTTS2Config(fm_n_layers=2, fm_n_timesteps=3)).to(DEVICE)

    # 3 sentences of different lengths
    sentences = [
        torch.randn(1, 10, TEXT),
        torch.randn(1, 18, TEXT),
        torch.randn(1, 7,  TEXT),
    ]

    styles = model.sample_style_longform(sentences, alpha=0.7)
    assert len(styles) == 3
    for i, s in enumerate(styles):
        assert s.shape == (1, STYLE), f"Sentence {i} bad shape {s.shape}"
    print(f"  [3a] {len(styles)} consistent style vectors  ✓")

    # Consecutive styles should differ (interpolation, not identical)
    assert not torch.allclose(styles[0], styles[1]), "Consecutive styles are identical"
    print(f"  [3b] Styles are distinct  ✓")

    print()


# =================================================================
# TEST 4: Training loop simulation (what your actual loop looks like)
# =================================================================
def test_4_training_loop_simulation():
    print("=" * 55)
    print("TEST 4 — Training loop simulation")
    print("=" * 55)

    model = SwaraTTS2(SwaraTTS2Config(fm_n_layers=2)).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step in range(3):
        optim.zero_grad()

        # Simulate one batch
        h_bert  = torch.randn(B, N, TEXT)
        s_real  = torch.randn(B, STYLE)   # from style_enc_a + style_enc_p

        # Padding mask: last 2 tokens are padding in sample 1
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[1, -2:] = True

        # OT-CFM loss (replaces Ledm)
        loss_fm = model.compute_style_loss(s_real, h_bert, text_mask=mask)

        # Simulate other losses (mel, adv, etc.) as random scalars
        loss_mel = torch.tensor(0.5)
        loss_total = loss_mel + loss_fm

        loss_total.backward()
        optim.step()

        print(f"  step {step+1}: loss_fm={loss_fm.item():.4f}, "
              f"loss_total={loss_total.item():.4f}  ✓")

    print()


# =================================================================
# TEST 5: ISTFTDecoder + SwaraTTS2 end-to-end shape check
# =================================================================
def test_5_decoder_integration():
    print("=" * 55)
    print("TEST 5 — ISTFTDecoder + SwaraTTS2 end-to-end shape")
    print("=" * 55)

    STYLE_HALF = STYLE // 2   # 128, matches default decoder style_dim
    T_FRAMES   = 80            # mel frames
    DIM_IN     = 512

    model = SwaraTTS2(SwaraTTS2Config(fm_n_layers=2, fm_n_timesteps=3))

    decoder = ISTFTDecoder(
        dim_in=DIM_IN,
        style_dim=STYLE_HALF,  # decoder only takes s_a (the acoustic half)
        hidden_dim=256,
        asr_res_dim=32,
        generator_dim=128,
        num_decode_blocks=2,
        upsample_rates=(10, 6),
        upsample_initial_channel=128,
        upsample_kernel_sizes=(20, 12),
        resblock_kernel_sizes=(3, 7),
        resblock_dilation_sizes=((1, 3), (1, 3)),
        gen_istft_n_fft=20,
        gen_istft_hop_size=5,
        sample_rate=24000,
    )

    h_bert = torch.randn(B, N, TEXT)
    asr    = torch.randn(B, DIM_IN, T_FRAMES)
    f0     = torch.randn(B, T_FRAMES)
    energy = torch.randn(B, T_FRAMES)

    # Generate style via flow matching
    style = model.sample_style(h_bert, n_timesteps=3)
    s_a, s_p = model.split_style(style)  # s_a → decoder, s_p → predictors

    # Run decoder
    waveform = decoder(asr, f0, energy, s_a)
    print(f"  [5a] waveform shape = {tuple(waveform.shape)}  ✓")
    assert waveform.dim() == 3, "Expected [B, 1, T_audio]"
    assert waveform.shape[0] == B
    assert waveform.shape[1] == 1

    print()


# =================================================================
# Main
# =================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    test_1_sampler_standalone()
    test_2_swara_model()
    test_3_longform()
    test_4_training_loop_simulation()
    test_5_decoder_integration()

    print("=" * 55)
    print("ALL INTEGRATION TESTS PASSED  ✓")
    print("=" * 55)
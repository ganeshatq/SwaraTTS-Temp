import torch
from torch import nn
import torch.nn.functional as F

class InformedPrior(nn.Module):
    def __init__(self, input_dim=256, style_dim=256):
        super().__init__()
        # Map pooled text c [B, 256] to mu_0, sigma_0 [B, 256]
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, style_dim * 2)
        )

    def forward(self, c):
        stats = self.net(c)
        mu_0, log_sigma_0 = stats.chunk(2, dim=-1)
        sigma_0 = torch.exp(log_sigma_0)
        return mu_0, sigma_0

class VelocityField(nn.Module):
    def __init__(self, style_dim=256, hidden_dim=256, n_heads=8, n_layers=6):
        super().__init__()
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim) # Sinusoidal time embedding project
        self.text_proj = nn.Linear(style_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, style_dim)

    def get_time_embedding(self, t, dim):
        # t is [B]
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, s_t, t, c):
        # s_t: [B, 256], t: [B], c: [B, 256]
        t_emb = self.get_time_embedding(t, s_t.size(-1))
        
        # Condition as tokens for Transformer
        s_input = self.style_proj(s_t).unsqueeze(1) # [B, 1, 256]
        t_input = self.time_proj(t_emb).unsqueeze(1) # [B, 1, 256]
        c_input = self.text_proj(c).unsqueeze(1)     # [B, 1, 256]
        
        # Concatenate conditions
        feat = torch.cat([s_input, t_input, c_input], dim=1) # [B, 3, 256]
        
        out = self.transformer(feat)
        v = self.output_proj(out[:, 0, :]) # Take first token corresponding to s_t
        
        return v

class CFMLoss(nn.Module):
    def __init__(self, sigma_min=0.1):
        super().__init__()
        self.sigma_min = sigma_min

    def forward(self, v_theta, s_1, s_0, t, c):
        # s_1: target style [B, 256]
        # s_0: noise/prior style [B, 256]
        # t: time [B]
        # c: text condition [B, 256]
        
        # Optimal transport path: s_t = (1-t)s_0 + t*s_1
        t = t.view(-1, 1)
        s_t = (1 - t) * s_0 + t * s_1
        
        v_target = s_1 - s_0
        v_pred = v_theta(s_t, t.squeeze(), c)
        
        loss = F.mse_loss(v_pred, v_target)
        return loss

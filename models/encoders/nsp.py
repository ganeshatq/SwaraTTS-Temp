import torch
from torch import nn
import torch.nn.functional as F

class NeuralSchedulePredictor(nn.Module):
    def __init__(self, c_dim=256, hidden_dim=128, max_steps=6):
        super().__init__()
        self.max_steps = max_steps
        
        # Predicting relative step sizes delta_tau
        self.mlp = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_steps)
        )
        
        # Classification head for Simple (K=2) vs Complex (K=max_steps)
        self.complexity_head = nn.Linear(c_dim, 2)
        
    def forward(self, c, soft_sort=True):
        """
        Args:
            c: [B, 256]
        Returns:
            tau: [B, K] sorted timesteps in (0, 1)
            is_complex: [B, 2] logits
        """
        # Predict logits for complexity
        is_complex = self.complexity_head(c) # [B, 2]
        
        # Predict relative intervals
        intervals = torch.sigmoid(self.mlp(c)) # [B, K]
        
        # Cumulative sum to get timesteps
        # We want 0 < tau_1 < tau_2 ... < 1
        # Normalize intervals to sum to slightly less than 1
        sum_intervals = intervals.sum(dim=-1, keepdim=True) + 1e-6
        normalized_intervals = (intervals / sum_intervals) * 0.98 # Ensure < 1
        
        # Add a small epsilon to avoid zero steps
        normalized_intervals = normalized_intervals + 0.001
        
        # Cumulative sum starting from ~0
        tau = torch.cumsum(normalized_intervals, dim=-1)
        
        # Differentiable sorting check (using torch.sort as simple proxy)
        # If intervals are positive, cumsum is monotonic.
        
        return tau, is_complex

def soft_sort_timesteps(tau, temperature=0.1):
    # Differentiable sort implementation if needed for more complex delta predictions
    # For cumsum over positive intervals, it's inherently sorted.
    return torch.sort(tau, dim=-1)[0]

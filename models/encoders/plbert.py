import torch
from torch import nn
from transformers import AlbertConfig, AlbertModel

class PLBERTTextEncoder(nn.Module):
    def __init__(self, 
                 vocab_size=30000, 
                 hidden_size=768, 
                 out_size=256,
                 num_layers=12,
                 num_heads=12,
                 intermediate_size=3072):
        super().__init__()
        
        config = AlbertConfig(
            vocab_size=vocab_size,
            embedding_size=128, 
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=512,
        )
        
        self.encoder = AlbertModel(config)
        self.proj = nn.Linear(hidden_size, out_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [B, T_text]
            attention_mask: [B, T_text]
        Returns:
            h_text: [B, 256, T_text]
            c: [B, 256] (pooled text condition)
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B, T, 768]
        
        h_proj = self.proj(h) # [B, T, 256]
        
        # h_text: [B, 256, T_text]
        h_text = h_proj.transpose(1, 2)
        
        # pooled c: [B, 256] (mean pooling over time)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float() # [B, T, 1]
            c = (h_proj * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        else:
            c = h_proj.mean(dim=1)
            
        return h_text, c

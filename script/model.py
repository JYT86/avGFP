# Implemented by JYT on 19 June 2025 – Latest version

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN


class ItPredConfig:
    def __init__(
        self,
        dim=1280,
        n_heads=20,
        n_layers=12,
        intermediate_dim=None,
        dropout=0.1,
        activation="gelu",
        residue_mask=True 
    ):
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim or dim * 4
        self.dropout = dropout
        self.activation = activation
        self.residue_mask = residue_mask


class ItAttention(nn.Module):
    def __init__(self, config: ItPredConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.dim, config.dim)
        self.k_proj = nn.Linear(config.dim, config.dim)
        self.v_proj = nn.Linear(config.dim, config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores += attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(output)


class ItPredBlock(nn.Module):
    def __init__(self, config: ItPredConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = ItAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)

        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.intermediate_dim),
            ACT2FN[config.activation],
            nn.Linear(config.intermediate_dim, config.dim),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ItPredModel(nn.Module):
    def __init__(self, config: ItPredConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ItPredBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].to(dtype=x.dtype, device=x.device)
            
            if self.config.residue_mask:
                attention_mask = attention_mask.expand(attention_mask.size(0), self.config.n_heads, attention_mask.size(-1), attention_mask.size(-1))
                residue_mask = np.load('../data/spatial_distance_attn_mask.npy')
                L = residue_mask.shape[0]
                residue_mask = torch.from_numpy(residue_mask).expand(attention_mask.size(0), self.config.n_heads, L, L).to(dtype=x.dtype, device=x.device)
                temp_mask = attention_mask.clone()
                temp_mask[:, :, :L, :L] = attention_mask[:, :, :L, :L] * residue_mask
                attention_mask = temp_mask
                
            attention_mask = (1.0 - attention_mask) * -10000.0
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
        return self.ln_f(x)
    

class ItPred4Classification(nn.Module):
    def __init__(self, config: ItPredConfig):
        super().__init__()
        self.encoder = ItPredModel(config)
        self.pooler = nn.AdaptiveAvgPool1d(1)  
        self.head = nn.Sequential(
            nn.Linear(config.dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  
        )

    def forward(self, x, attention_mask=None):
        encoded = self.encoder(x, attention_mask)  # (B, T, D)

        x_pooled = encoded.transpose(1, 2)         # (B, D, T)
        x_pooled = self.pooler(x_pooled).squeeze(-1)  # (B, D)

        output = self.head(x_pooled)  # (B, 1)
        return output
    
class ItPred4Regression(nn.Module):
    def __init__(self, config: ItPredConfig):
        super().__init__()
        self.encoder = ItPredModel(config)
        self.pooler = nn.AdaptiveAvgPool1d(1)  
        self.head = nn.Sequential(
            nn.Linear(config.dim, 1),
            nn.Softplus()  
        )

    def forward(self, x, attention_mask=None):
        encoded = self.encoder(x, attention_mask)  # (B, T, D)
        x_pooled = encoded.transpose(1, 2)         # (B, D, T)
        x_pooled = self.pooler(x_pooled).squeeze(-1)  # (B, D)

        output = self.head(x_pooled)  # (B, 1)
        return output
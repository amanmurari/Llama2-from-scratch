import torch
from torch import nn
from config import LLAMA2_CONFIG_7B

def precompute_rope(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    position= torch.arange(context_length)
    angle= position[:,None]*inv_freq[None,:]
    angles= torch.cat([angle,angle],dim=1)
    return torch.cos(angles),torch.sin(angles)

def compute_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]  
    x2 = x[..., head_dim // 2 :]  
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0) 
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)
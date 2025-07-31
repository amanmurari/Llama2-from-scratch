import torch
import numpy as np
from torch import nn
from text_model.attention import MultiHeadAttention

from text_model.layers import SiLU,RMSNorm

class PositionalEmbedding(nn.Module):
    def __init__(self, d_in,max_seqlen,dtype):
        super().__init__()
        self.cls_tk=nn.Parameter(torch.randn(1,1,d_in,dtype=dtype))
        pe=torch.zeros(max_seqlen,d_in,dtype=dtype)
        for pos in range(max_seqlen):
            for i in range(d_in):
                if i%2==0:
                    pe[pos][i]==np.sin(pos/(1000**(i/d_in)))
                else:
                    pe[pos][i]==np.cos(pos/(1000**(i-1/d_in)))
        self.register_buffer("pe",pe.unsqueeze(0))

    def forward(self,x):
        token_batch= self.cls_tk.expand(x.size(0),-1,-1)
        x=torch.cat([token_batch,x],dim=1)
        return x+self.pe

class MultiModelProjector(nn.Module):
    def __init__(self, n_embd,n_imgdim,dtype):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_imgdim,4*n_imgdim,dtype=dtype),
            nn.SiLU(),
            nn.Linear(n_imgdim*4,n_embd,dtype=dtype)
        )
    def forward(self,x):
        return self.net(x)
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["img_embd_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["img_embd_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["img_embd_dim"], dtype=cfg["dtype"], bias=False)
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["img_embd_dim"],
            d_out=cfg["img_embd_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"],is_useimg=True
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["img_embd_dim"])
        self.norm2 = RMSNorm(cfg["img_embd_dim"])
    def forward(self,x):
        sortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=x+sortcut
        sortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        return x+sortcut
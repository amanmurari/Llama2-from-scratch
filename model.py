import torch
from torch import nn
from attention import MultiHeadAttention
from layers import FeedForward,RMSNorm
from config import LLAMA2_CONFIG_7B
from torch.nn import functional as F

class TransformerBlock(nn.Module):
    def __init__(self, cfg: LLAMA2_CONFIG_7B):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"]      
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
    def forward(self,x):
        sortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=x+sortcut
        sortcut=x
        x=self.norm2(x)
        x=self.ff(x)
        return x+sortcut
    

class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) 
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, x):
       
        x = self.tok_emb(x)
        x = self.final_norm(x)
        x = self.out_head(x)
        return x
    

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["context_length"]:]
            logits= self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

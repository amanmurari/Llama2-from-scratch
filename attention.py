import torch
from torch import nn
from config import LLAMA2_CONFIG_7B
from rope import precompute_rope,compute_rope


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):  # ,dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

       
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)  # Linear layer to combine head outputs
        # self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

       
        cos, sin = precompute_rope(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self,x: torch.Tensor):
        b,seq_len,_= x.shape
        keys = self.W_key(x)  
        queries = self.W_query(x)
        values = self.W_value(x)
        keys= keys.view(b,seq_len,self.num_heads,self.head_dim)
        values= values.view(b,seq_len,self.num_heads,self.head_dim)
        queries= queries.view(b,seq_len,self.num_heads,self.head_dim)
        keys=keys.transpose(1,2)
        queries=queries.transpose(1,2)
        values=values.transpose(1,2)
        keys=compute_rope(keys,self.cos,self.sin)
        queries=compute_rope(queries,self.cos,self.sin)
        atten_sccr= queries @keys.transpose(2,3)
        mask_bool= self.mask.bool()[:seq_len,:seq_len]
        atten_sccr.masked_fill_(mask_bool,-torch.inf)
        atten_weg= torch.softmax(atten_sccr/keys.shape[-1]**.5,dim=-1)
        contxt_vec= (atten_weg@values).transpose(1,2)
        contxt_vec=contxt_vec.reshape(b,seq_len,self.d_out)
        return self.out_proj(contxt_vec)



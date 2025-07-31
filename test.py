
from model import Llama2Model
from vision_model.vit import VIT
import numpy as np
import torch
from config import LLAMA2_CONFIG_7B
img= torch.randn(1,3,64,64,dtype=torch.bfloat16)
batch_size, seq_len = 1, 32
test_input = torch.randint(0, LLAMA2_CONFIG_7B["vocab_size"], (batch_size, seq_len))

vi=VIT(LLAMA2_CONFIG_7B["img_size"],LLAMA2_CONFIG_7B["patch_size"],LLAMA2_CONFIG_7B["img_embd_dim"],LLAMA2_CONFIG_7B["n_layers"],dtype=LLAMA2_CONFIG_7B["dtype"])
llmv=Llama2Model(LLAMA2_CONFIG_7B,True)
print(llmv(test_input,img).shape)



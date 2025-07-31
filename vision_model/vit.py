import torch
from torch import nn
from text_model.layers import FeedForward
from vision_model.layer import TransformerBlock
from config import LLAMA2_CONFIG_7B
from vision_model.layer import PositionalEmbedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size,hid_dim,patch_size,dtype):
        super().__init__()
        self.num_patch= (img_size//patch_size)**2
        self.conv= nn.Conv2d(3,hid_dim,patch_size,patch_size,dtype=dtype)

    def forward(self,x):
        x=self.conv(x)
        x= x.flatten(2)
        x=x.transpose(1,2)
        return x
    


class VIT(nn.Module):
    def __init__(self, img_size,patch_size,hid_dim,n_layer,dtype):
        super().__init__()

        self.patchs= PatchEmbedding(img_size,hid_dim,patch_size,dtype=dtype)

        self.pos=PositionalEmbedding(hid_dim,self.patchs.num_patch+1,dtype=dtype)
        self.blocks=nn.ModuleList([TransformerBlock(LLAMA2_CONFIG_7B) for _ in range(n_layer)])
    def forward(self,x):
        x=self.patchs(x)
        x=self.pos(x)
        for block in self.blocks:
            x=block(x)
        return x[:,0]




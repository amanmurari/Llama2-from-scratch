import torch
from torch import nn 
from text_model.model import Llama2TextModel
from vision_model.vit import VIT

class Llama2Model(nn.Module):
    def __init__(self, cfg,use_img=False):
        super().__init__()
        self.use_img=use_img
        self.texter= Llama2TextModel(cfg,use_img)
        self.visioner= VIT(cfg["img_size"],cfg["patch_size"],cfg["img_embd_dim"],cfg["n_layers"],dtype=cfg["dtype"])

    def forward(self,x,img=None):
        if self.use_img:
            img_embd=self.visioner(img)
            if img_embd.nelement()==0 or img_embd.shape[1]==0:
                raise ValueError("sonething went wrong")
            return self.texter(x,img_embd)
        return self.texter(x)
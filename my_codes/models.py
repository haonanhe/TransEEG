from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F

import math 
from einops.layers.torch import Rearrange
from einops import rearrange

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import DepthwiseConv2d, SeparableConv2d, Conv2dNormWeight

from attention import Attention, ConvAttention, MultiHeadAttention
from utils import _get_activation_fn

# positional emcoding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        dive_term = (torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * dive_term)
        pe[:, 1::2] = torch.cos(position * dive_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe + x

# position-wise feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = expansion * d_model

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.feedforward(x)

# MBconv
class MBConv(nn.Module):
    def __init__(self, dim, dropout=0.5, activation='gelu'):
        super().__init__()
        self.conv_expand = nn.Conv2d(dim, 10*dim, kernel_size=1, stride=1, bias=False)
        self.conv_depthwise = DepthwiseConv2d(10*dim, 1, kernel_size=(3, 3), padding='same', bias=False) 
        self.conv_reduce = nn.Conv2d(10*dim, dim, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(10*dim)
        self.bn2 = nn.BatchNorm2d(dim)

        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def forward(self, x):
        if len(x.shape) < 4:
            x = rearrange(x, 'b h w -> b 1 h w')
        x = self.dropout(self.activation(self.bn1(self.conv_expand(x))))
        x = self.conv_depthwise(x)
        x = self.dropout(self.activation(self.bn2(self.conv_reduce(x))))
        return x

# pre normalization
class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Conv Transformer
class Convformer(nn.Module):
    def __init__(self, d_model, dim, depth, n_head, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, ConvAttention(dim, d_model, n_head, dropout)),
                PreNorm(d_model, MBConv(dim, dropout=dropout)),
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# original Transformer
class Transformer(nn.Module):
    def __init__(self, d_model, dim, depth, n_head, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, MultiHeadAttention(d_model, n_head, dropout)),
                PreNorm(d_model, PositionwiseFeedForward(d_model, dropout=dropout)),
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
        
# our modified model
class ConvViT(nn.Module):
    def __init__(self, n_timepoints=1000, n_channels=22, dim=1, emb_size=16, n_classes=4, n_head=1, n_layers=1, dropout=0.5):
        super().__init__()
        # pre conv feature extract & patch emb
        self.patch_emb = nn.Sequential(
            nn.Conv2d(1, emb_size, (25, 1), padding='same',bias=False),
            nn.BatchNorm2d(emb_size),
            DepthwiseConv2d(emb_size, 1, (1, n_channels), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.Dropout(0.3),
            Rearrange('b c h w -> b w h c')
        ) 

        # downsampling
        self.down1 = nn.AvgPool2d((5, 1), stride=(5, 1))
        self.down2 = nn.AvgPool2d((5, 1), stride=(5, 1))
        self.down3 = nn.AvgPool2d((5, 1), stride=(5, 1))
        # self.down1 = nn.MaxPool2d((5, 1), stride=(5, 1)) # maxpool may cause severe overfit
        # self.down2 = nn.MaxPool2d((5, 1), stride=(5, 1))
        # self.down3 = nn.MaxPool2d((5, 1), stride=(5, 1))

        # transformer
        self.transformer1 = Convformer(emb_size, dim, 1, n_head, dropout)
        self.transformer2 = Convformer(emb_size, dim, 1, n_head, dropout)
        self.transformer3 = Convformer(emb_size, dim, 1, n_head, dropout)
        
        # classifier
        self.cls = nn.Sequential(
            Rearrange('b c h w -> b w h c'),
            nn.Conv2d(emb_size, n_classes, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.LogSoftmax(dim=1)
        )

        # positional encoding
        self.pos_embedding = PositionalEmbedding(emb_size, n_timepoints)
        self.pos_drop =  nn.Dropout(0.1)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.patch_emb(x)
        
        x = self.pos_drop(self.pos_embedding(x))

        x = self.down1(x)
        x = self.transformer1(x)

        x = self.down2(x)
        x = self.transformer2(x)

        x = self.down3(x)
        x = self.transformer3(x)

        x = self.cls(x)
        x = x[:, :, 0, 0]

        return x
        
if __name__ == '__main__' :
    x = torch.ones(16, 1, 1000, 22)
    model = ConvViT(n_timepoints=1000, n_channels=22, dim=1, emb_size=16, n_classes=4, n_head=1, n_layers=1, dropout=0.5)
    x = model(x)
    print(x.shape)
    print(sum(x.numel() for x in model.parameters()))

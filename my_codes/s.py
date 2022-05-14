import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log
from common.torchutils import DepthwiseConv2d
from common.torchutils import Conv2dNormWeight

from attention import ConvAttention, Attention
from attention import MultiHeadAttention
from common.torchutils import SeparableConv2d
from utils import _get_activation_fn

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

# positional embedding
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

# positionwise feedforward
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
        self.conv_expand = nn.Conv2d(dim, 4*dim, kernel_size=1, stride=1, bias=False)
        self.conv_depthwise = DepthwiseConv2d(4*dim, 1, kernel_size=(1, 10), padding='same', bias=False) # kernel_size很重要，不能是(3, 3)
        self.conv_reduce = nn.Conv2d(4*dim, dim, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(4*dim)
        self.bn2 = nn.BatchNorm2d(dim)

        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def forward(self, x):
        if len(x.shape) < 4:
            x = rearrange(x, 'b h w -> b 1 h w')
        x = self.dropout(self.activation(self.bn1(self.conv_expand(x))))
        x = self.conv_depthwise(x)
        x = self.dropout(self.activation(self.bn2(self.conv_reduce(x))))
        # x = rearrange(x, 'b 1 h w -> b h w')
        return x

# residual add
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_size, dropout, kernel_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size, stride=kernel_size),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(emb_size),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, n_head, dropout = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, n_head, dropout)),
                PreNorm(dim, PositionwiseFeedForward(dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class LiteViT(nn.Module):
    def __init__(self, dim, emb_size, n_classes, n_head=1, n_layers=1, dropout=0.5):
        super().__init__()
        self.conv_t = nn.Sequential(
            nn.Conv2d(1, dim, (25, 1), padding='same', bias=False),
            nn.BatchNorm2d(dim),
            nn.Dropout(dropout),
            # nn.AvgPool2d((10, 1), (10, 1)),
            # nn.Conv2d(dim, emb_size, (5, 22), stride=(5, 1), bias=False),
            # Rearrange('b c h w -> b w w c'),
            # nn.LayerNorm(emb_size)
        ) 

        # self.conv_pre = nn.Sequential(
        #     nn.Conv2d(1, 4, (25, 1), padding=(25//2, 0), bias=False),
        #     nn.BatchNorm2d(4),
        #     Conv2dNormWeight(
        #         4, emb_size, (1, 22), stride=(1, 1),
        #         max_norm=1, groups=1, bias=False
        #     ),
        #     nn.BatchNorm2d(emb_size),
        #     nn.ELU(),
        #     nn.AvgPool2d((5, 1), stride=(5, 1)),
        #     nn.Dropout(dropout),
        #     Rearrange('b c h w -> b (h w) c')
        # ) 

        # self.transformer_t = nn.Sequential(
        #     PreNorm(22, ConvAttention(dim, 22, 2, dropout)),
        #     # PreNorm(22, PositionwiseFeedForward(22, 4, dropout=dropout))
        #     PreNorm(22, MBConv(dim, dropout=dropout))
        # )

        self.transformer_s = nn.Sequential(
            # PreNorm(emb_size, PositionwiseFeedForward(emb_size, 4, dropout=dropout))
            PreNorm(1000, ConvAttention(dim, 1000, n_head, dropout)),
            PreNorm(1000, MBConv(dim, dropout=dropout))
        )

        # self.ff = PreNorm(22, MBConv(2, dropout=dropout))

        self.conv_post = nn.Sequential(
            # DepthwiseConv2d(dim, 1, (1, 22), padding='same', bias=False),
            # nn.BatchNorm2d(dim),
            # # Expression(cov),
            # # Expression(safe_log),
            # nn.Dropout(dropout),

            SeparableConv2d(dim, n_classes, (5, 22), padding='same', bias=False),
            nn.BatchNorm2d(n_classes),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.cls = nn.Sequential(
            # Reduce('b n e -> b e', reduction='mean'),
            # nn.LayerNorm(emb_size),
            # nn.Linear(emb_size, n_classes),
            # nn.LogSoftmax(dim=1)

            # Reduce('b n e -> b e', reduction='mean'),
            # nn.LayerNorm(44),
            # nn.Linear(44, n_classes),
            # nn.LogSoftmax(dim=1)

            # Reduce('b c n e -> b e', reduction='mean'),
            # nn.LayerNorm(22),
            # nn.Linear(22, n_classes),
            # nn.LogSoftmax(dim=1)

            nn.AdaptiveAvgPool2d(1),
            nn.LogSoftmax(dim=1)
        )

        self.pos_embedding_s = PositionalEmbedding(1000, 22)
        self.pos_embedding_t = PositionalEmbedding(22, 1000)

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
        # pre
        # x = self.conv_pre(x)
        x = self.conv_t(x)
        # spatial
        s = rearrange(x, 'b c h w -> b c w h')
        s = self.pos_drop(self.pos_embedding_s(s))
        s = self.transformer_s(s)
        s = rearrange(s, 'b c h w -> b c w h')
        print(s.shape)
        # temporal
        # t = self.pos_drop(self.pos_embedding_t(x))
        # t = self.transformer_t(t)
        x = s #+ t
        # merge
        # x = s + t
        # classification
        # x = self.conv_post(x)
        out = self.cls(x)

        return out[:, :, 0, 0]

x = torch.ones(16, 1, 1000, 22)
model = LiteViT(4, 10, 4, n_head=10, n_layers=1)
x = model(x)
print(x.size())
print(sum(x.numel() for x in model.parameters()))


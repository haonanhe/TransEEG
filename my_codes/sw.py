import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from attention import Attention
# from csplit import PositionalEmbedding

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

import torch
import torch.nn as nn
import math

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log
from common.torchutils import DepthwiseConv2d

from attention import ConvAttention, Attention
from attention import MultiHeadAttention
from EEG_Transformer.Trans import channel_attention
from common.torchutils import SeparableConv2d
from utils import _get_activation_fn
# from model import PositionalEmbedding

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from attention import Attention

from common.torchutils import Conv2dNormWeight

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
        self.conv_depthwise = DepthwiseConv2d(4*dim, 1, kernel_size=(3, 3), padding='same', bias=False)
        self.conv_reduce = nn.Conv2d(4*dim, 1, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(4*dim)
        self.bn2 = nn.BatchNorm2d(1)

        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def forward(self, x):
        if len(x.shape) < 4:
            x = rearrange(x, 'b h w -> b 1 h w')
        x = self.dropout(self.activation(self.bn1(self.conv_expand(x))))
        x = self.conv_depthwise(x)
        x = self.dropout(self.activation(self.bn2(self.conv_reduce(x))))
        x = rearrange(x, 'b 1 h w -> b h w')
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

# patch merging
class PatchMerging(nn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.resolution = resolution
        self.reduction = nn.Linear(resolution*dim, resolution*dim // 4, bias=False)
        self.norm = nn.LayerNorm(resolution*dim)

    def forward(self, x):
        tmp = x[:, 0::self.resolution, :]
        for i in range(1, self.resolution):
            tmp = torch.cat([tmp, x[:, i::self.resolution, :]], -1)

        tmp = self.norm(tmp)
        tmp = self.reduction(tmp)

        return tmp

class PreNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x), **kwargs)

from einops import repeat
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
    def __init__(self, emb_size, n_classes, n_head=1, n_layers=1, dropout=0.5):
        super().__init__()
        # self.conv_t = nn.Sequential(
        #     nn.Conv2d(1, 1, (25, 1), padding='same'),
        #     nn.BatchNorm2d(1),
        #     nn.Dropout(0.1),
        # ) 

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

        # self.embed = nn.Linear(220, 22)
        # # self.transformer_t = Transformer(22, n_layers, n_head, dropout)
        # self.transformer_t = nn.Sequential(
        #     PreNorm(22, MultiHeadAttention(22, 2, dropout)),
        #     PreNorm(22, PositionwiseFeedForward(22, dropout=dropout))
        # )
        self.transformer_t = nn.Sequential(
            PreNorm(22, ConvAttention(22, 2, dropout)),
            # PreNorm(22, PositionwiseFeedForward(22, 4, dropout=dropout))
            # PreNorm(emb_size, ConvAttention(emb_size, 2, dropout)),
            # PreNorm(emb_size, MultiHeadAttention(emb_size, 2, dropout)),
            # PreNorm(emb_size, PositionwiseFeedForward(emb_size, 4, dropout=dropout))
            PreNorm(22, MBConv(1, dropout=dropout))
        )
        # self.pooling_t = nn.AvgPool2d((10, 1), (10, 1))

        # self.patch_embedding = PatchEmbedding(1, emb_size, dropout, (10, 1))
        # self.transformer = Transformer(emb_size, n_layers, n_head, dropout)
        self.transformer_s = nn.Sequential(
            # PreNorm(emb_size, ConvAttention(emb_size, n_head, dropout)),
            # # PreNorm(emb_size, PositionwiseFeedForward(emb_size, 4, dropout=dropout))
            # # PreNorm(emb_size, MBConv(1, emb_size, dropout=dropout))

            PreNorm(1000, ConvAttention(1000, n_head, dropout)),
            PreNorm(1000, MBConv(1, dropout=dropout))
        )

        self.ff = PreNorm(22, MBConv(2, dropout=dropout))

        # self.patch_merging = PatchMerging(4, emb_size)
        # self.window_attn_2 = ResidualAdd(
        #     PreNorm(emb_size, MultiHeadAttention(emb_size, n_head, window_size, dropout))
        # )
        # self.ff_2 = ResidualAdd(
        #     PreNorm(emb_size, PositionwiseFeedForward(emb_size, dropout=dropout))
        # )
        
        # self.transformer_post = Transformer(emb_size, 1, n_head, dropout)

        # self.conv_post = nn.Sequential(
        #     # Rearrange('b h w -> b 1 h w'),
        #     # SeparableConv2d(1, 1, (10, emb_size), padding='same', bias=False, depth_multiplier=4),
        #     # nn.BatchNorm2d(1),
        #     # nn.ELU(),
        #     # nn.Dropout(dropout),
        #     # Rearrange('b c h w -> b (c h) w'),

        #     # Rearrange('b h w -> b 1 h w'),
        #     SeparableConv2d(1, 1, (1, 22), padding='same', bias=False, depth_multiplier=4),
        #     nn.BatchNorm2d(1),
        #     nn.ELU(),
        #     nn.Dropout(dropout),
        #     Rearrange('b c h w -> b (c h) w'),
        # )

        # self.reduce = Reduce('b n e -> b e', reduction='mean')

        # self.conv_reduce = nn.Sequential(
        #     # Rearrange('b h w -> b 1 w h'),
        #     DepthwiseConv2d(100, 1, (22, 1), bias=False),
        #     nn.BatchNorm2d(100),
        #     nn.Dropout(dropout),
        #     # nn.AvgPool2d((1, 10), (1, 10)),
        #     Reduce('b c h w -> b c w', reduction='mean'),
        # )

        # self.conv_trans = nn.Sequential(
        #     # Rearrange('b h w -> b 1 w h'),
        #     # DepthwiseConv2d(100, 1, (1, 10), bias=False),
        #     # nn.BatchNorm2d(100),
        #     # nn.Dropout(dropout),
        #     nn.Conv2d(1, emb_size, (10, 22), stride=(10, 1)),
        #     Rearrange('b e (h) (w) -> b (h w) e')
        # )

        self.cls = nn.Sequential(
            # Reduce('b n e -> b e', reduction='mean'),
            # nn.LayerNorm(emb_size),
            # nn.Linear(emb_size, n_classes),
            # nn.LogSoftmax(dim=1)

            # Reduce('b n e -> b e', reduction='mean'),
            # nn.LayerNorm(44),
            # nn.Linear(44, n_classes),
            # nn.LogSoftmax(dim=1)

            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(22),
            nn.Linear(22, n_classes),
            nn.LogSoftmax(dim=1)

            # Reduce('b n e -> b e', reduction='mean'),
            # nn.LayerNorm(32),
            # nn.Linear(32, n_classes),
            # nn.LogSoftmax(dim=1)
        )

        # self.pos_embedding_s = nn.Parameter(torch.randn(1, 22, 1000))
        self.pos_embedding_s = PositionalEmbedding(1000, 22)
        self.pos_embedding_t = PositionalEmbedding(22, 1000)
        # self.pos_embedding_t = PositionalEmbedding(10, 200)
        # self.pos_embedding_t = nn.Parameter(torch.randn(1, 101, emb_size))
        self.pos_embedding_post = nn.Parameter(torch.randn(1, 101, emb_size))
        self.pos_drop =  nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.cls_token_post = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.cls_token_t = nn.Parameter(torch.zeros(1, 1, emb_size))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x = self.conv_pre(x)
        # x = self.conv_t(x)
        # x = rearrange(x, 'b c h w -> (b c) 1 h w')
        ####################

        # s = self.patch_embedding(x)
        # s = rearrange(s, 'b c h w -> (b c) h w')
        # s = self.pos_drop(s + self.pos_embedding[:, :22])
        # s = self.transformer_s(s)
        # s = rearrange(s, '(b h) w c -> b h w c', h=100)
        # s = self.conv_reduce(s)

        s = rearrange(x, 'b c h w -> (b c) w h')
        s = self.pos_drop(self.pos_embedding_s(s))
        s = self.transformer_s(s)
        s = rearrange(s, 'b h w -> b w h')

        # t = self.conv_trans(x)
        # print(t.shape)

        t = rearrange(x, 'b c h w -> b (c h) w')
        t = self.pos_drop(self.pos_embedding_t(t))
        t = self.transformer_t(t)

        # t = self.pooling_t(t)
        # s = rearrange(s, 'b h w -> b 1 h w')
        # t = rearrange(t, 'b h w -> b 1 h w')
        # x = torch.cat((s, t), 1)
        # x = self.ff(x)
        x = s + t
        # x = s
        # print(x.shape)

        
        # x = s + t
        # x = self.pos_drop(self.pos_embedding_t(x))
        # x = self.transformer_t(x)

        # x = self.reduce(x)


        # s = self.reduce(s)
        # print(s.shape)
        
        # t = rearrange(x, 'b c h w -> b (c h) w')
        # # t = self.embed(t)
        # t = self.pos_drop(self.pos_embedding_t(t))
        # t = self.transformer_t(t)
        # t = rearrange(t, 'b h w -> b 1 h w')
        # t = self.patch_embedding(t)

        
        
        # print(s.shape)
        # print(t.shape)
        # x = s + t
        # x = rearrange(x, 'b c h w -> b c (h w)')
        # # x = self.reduce(x)
        # x = self.embed(x)
        # x = self.ff(x)

        # x = self.reduce(x)

        # s = self.embed(s)
        # print(s.shape)

        # t = rearrange(x, 'b c h w -> b c (h w)')
        # t = self.embed(t)
        # # cls_token_t = self.cls_token_t.repeat(t.shape[0], 1, 1)
        # # t = torch.cat((cls_token_t, t), 1)
        # t = self.pos_drop(t + self.pos_embedding_t[:, :100])
        # t = self.transformer_t(t)
        # t = self.reduce(t)

        # cls_token_t = t[:, 0]

        # cls_token = x[:, 0]
        # cls_token = rearrange(cls_token, '(b h) w c -> b h w c', h=100)
        # cls_token_post = self.reduce(cls_token)

        # cls_token_post = self.conv_reduce(cls_token)

        # cls_token_post = self.cls_token_post.repeat(cls_token.shape[0], 1, 1) 
        # cls_token = torch.cat((cls_token_post, cls_token), 1)

        # cls_token = self.pos_drop(cls_token + self.pos_embedding_post[:, :100])
        # cls_token = self.transformer_post(cls_token)

        # cls_token_post = cls_token[:, 0]
        ################
        # out = self.cls(cls_token_post)

        # cls_token = self.conv_post(cls_token)
        # out = self.cls(cls_token)
        # x = self.conv_post(x)
        
        # x = torch.concat((s, t), -1)
        # s = self.conv_reduce(s)
        out = self.cls(x)

        # cls_token = torch.cat((cls_token_t, cls_token_post), -1)
        # out = self.cls(cls_token)

        return out

x = torch.ones(16, 1, 1000, 22)
model = LiteViT(10, 4, n_head=2, n_layers=1)
x = model(x)
print(x.size())
print(sum(x.numel() for x in model.parameters()))


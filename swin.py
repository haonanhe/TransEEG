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

class WindowAttention(nn.Module):
    def __init__(self, d_model, n_head, window_size, dropout=0.5):
        super(WindowAttention, self).__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.window_size = window_size
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), n_head))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        n_channels = x.size(1)
        
        query = self.w_q(x).view(batch_size, -1, self.n_head*n_channels, self.d_head).transpose(1, 2)
        key = self.w_k(x).view(batch_size, -1, self.n_head*n_channels, self.d_head).transpose(1, 2)
        value = self.w_v(x).view(batch_size, -1, self.n_head*n_channels, self.d_head).transpose(1, 2)
        
        # query = self.dropout(query)
        # key = self.dropout(key)
        # value = self.dropout(value)

        scale = math.sqrt(query.size(-1))
        query = query * scale
        attn = torch.matmul(query, key.transpose(-2, -1))
        # print(attn.size())

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)

        attn = self.dropout(attn)

        x = torch.matmul(attn, value)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        x = self.w_out(x)
        x = x.view(batch_size, n_channels, -1, self.d_model)
        
        return x

import torch
import torch.nn as nn
import math

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log
from common.torchutils import DepthwiseConv2d

from attention import ConvMultiHeadAttention, Attention
from attention import MultiHeadAttention
from EEG_Transformer.Trans import channel_attention
from common.torchutils import SeparableConv2d
from utils import _get_activation_fn
from model import PositionalEmbedding

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from attention import Attention

from common.torchutils import Conv2dNormWeight
from swin import *

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

class MBConv(nn.Module):
    def __init__(self, dim, dropout=0.5, activation='gelu'):
        super().__init__()
        self.conv_expand = nn.Conv2d(dim, 4*dim, kernel_size=1, stride=1, bias=False)
        self.conv_depthwise = DepthwiseConv2d(4*dim, 1, kernel_size=(3, 3), padding='same', bias=False)
        self.conv_reduce = nn.Conv2d(4*dim, dim, kernel_size=1, stride=1, bias=False)

        self.bn1 = nn.BatchNorm2d(4*dim)
        self.bn2 = nn.BatchNorm2d(dim)

        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
    def forward(self, x):
        x = self.dropout(self.activation(self.bn1(self.conv_expand(x))))
        x = self.conv_depthwise(x)
        x = self.dropout(self.activation(self.bn2(self.conv_reduce(x))))

        return x

# positionwise feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = 4 * d_model

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.feedforward(x)

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
        self.reduction = nn.Linear(resolution*dim, resolution*dim // 2, bias=False)
        self.norm = nn.LayerNorm(resolution*dim)

    def forward(self, x):
        tmp = x[:, 0::self.resolution, :, :]
        for i in range(1, self.resolution):
            tmp = torch.cat([tmp, x[:, i::self.resolution, :, :]], -1)
        
        # tmp = rearrange(tmp, 'b c h w -> b h w c')
        tmp = self.norm(tmp)
        tmp = self.reduction(tmp)
        # tmp = rearrange(tmp, 'b h w c -> b c h w')

        return tmp

# patch add
class PatchAdd(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def forward(self, x):
        tmp = x[:, :, 0::self.resolution, :]
        for i in range(1, self.resolution):
            tmp = torch.add(tmp, x[:, :, i::self.resolution, :])

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
            # nn.Conv2d(in_channels, emb_size, kernel_size, stride=kernel_size),
            nn.Conv2d(in_channels, emb_size, kernel_size, stride=kernel_size),
            # nn.Conv2d(1, emb_size, (25, 1), padding='same'),# stride=(1, 1)),
            # nn.BatchNorm2d(emb_size),
            # nn.Dropout(0.1),
            # DepthwiseConv2d(emb_size, 1, (20, 1), stride=(20, 1)),
            # nn.AvgPool2d((20, 1), (20, 1)),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

# # conv multi-head attention
# class ConvAttention(nn.Module):
#     def __init__(self, dim, n_head, dropout=0.5, kernel_size=(100, 1)):
#         super(ConvAttention, self).__init__()
#         # extract temporal features from input
#         self.conv_in = nn.Conv2d(dim, 3 * n_head, kernel_size, padding='same',bias=False)
#         # self-attention
#         self.attention = Attention()
#         # point-wise conv
#         self.conv_out = nn.Conv2d(n_head, dim, kernel_size=1, stride=1, bias=False)
#         # self.w_out = nn.Linear(n_head*d_model, d_model)
        
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, x, mask=None):
        
#         x = rearrange(x, 'b c h w -> b c w h')
#         conv = self.conv_in(x)

#         x = rearrange(conv, 'b (a c) h w -> a b c h w', a=3)

#         query, key, value = x[0], x[1], x[2]
        
#         if mask is not None:
#             mask = mask.unsqueeze(1) 

#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

#         # x = torch.cat((x, conv), 1)
#         # x = rearrange(x, 'b c h w -> b h (c w)')
#         # x = self.w_out(x)
#         # x = x.unsqueeze(1)

#         # x = x + conv
#         # x = torch.cat((x, conv), 1)

#         x = self.conv_out(x)

#         x = rearrange(x, 'b c h w -> b c w h')

#         return x
        
class LiteViT(nn.Module):
    def __init__(self, emb_size, n_classes, window_size, n_head=1, n_layers=1, dropout=0.5):
        super().__init__()

        self.window_size = window_size

        self.conv_t = nn.Sequential(
            nn.Conv2d(1, 11, (25, 1), padding='same'),
            nn.BatchNorm2d(11),
            nn.Dropout(dropout),
        ) 

        self.pooling = nn.AvgPool2d((4, 1), (4, 1))

        self.patch_embedding = PatchEmbedding(11, emb_size, dropout, (10, 1))
        self.window_attn = ResidualAdd(
            PreNorm(emb_size, WindowAttention(emb_size, n_head, window_size, dropout))
        )
        self.ff = ResidualAdd(
            PreNorm(emb_size, PositionwiseFeedForward(emb_size, dropout=dropout))
        )
        
        self.conv = nn.Sequential(
            DepthwiseConv2d(100, 1, (1, emb_size)),
            nn.BatchNorm2d(100),
            nn.Dropout(dropout),
            Rearrange('b c h w -> b c (h w)'),
            nn.Linear(22, emb_size)
            # nn.AvgPool1d(2)

        )
        self.window_attn_t = ResidualAdd(
            PreNorm(emb_size, WindowAttention(emb_size, n_head, window_size, dropout))
        )
        self.ff_t = ResidualAdd(
            PreNorm(emb_size, PositionwiseFeedForward(emb_size, dropout=dropout))
        )

        # self.patch_merging = PatchMerging(2, emb_size)
        # # emb_size = emb_size * 1
        # self.window_attn_2 = ResidualAdd(
        #     PreNorm(emb_size, WindowAttention(emb_size, n_head, window_size, dropout))
        # )
        # self.ff_2 = ResidualAdd(
        #     PreNorm(emb_size, PositionwiseFeedForward(emb_size, dropout=dropout))
        # )

        # self.patch_merging_2 = PatchMerging(2, emb_size)
        # # emb_size = emb_size * 1
        # self.window_attn_3 = ResidualAdd(
        #     PreNorm(emb_size, WindowAttention(emb_size, 2, window_size, dropout))
        # )
        # self.ff_3 = ResidualAdd(
        #     PreNorm(emb_size, PositionwiseFeedForward(emb_size, dropout=dropout))
        # )

        # self.conv = DepthwiseConv2d(1, 1, (10, 1), padding='same')

        self.conv_s = nn.Sequential(
            # Rearrange('b h w c -> b c h w'),
            # DepthwiseConv2d(emb_size, 1, (10, 1), padding='same'),
            DepthwiseConv2d(1, 1, (5, emb_size), padding='same', bias=False),
            nn.BatchNorm2d(1),
            nn.Dropout(dropout),
            # Expression(cov),
            # Expression(safe_log),
            # Rearrange('b c h w -> b (h w) c'),
        )

        # self.conv_post = nn.Sequential(
        #     Rearrange('b h w c -> b c h w'),
        #     SeparableConv2d(emb_size, emb_size, (5, 22), bias=False),
        #     nn.BatchNorm2d(emb_size),
        #     # nn.ELU(),
        #     # nn.AvgPool2d((pool_size_time_2, 1), stride=(pool_stride_time_2, 1)),
        #     nn.Dropout(dropout),
        #     # Expression(cov),
        #     # Expression(safe_log),
        #     Rearrange('b c h w -> b (h w) c'),
        # )

        #     SeparableConv2d(
        #         emb_size, emb_size, (10, 1), 
        #         padding=(10//2, 0), bias=False
        #     ),
        #     nn.BatchNorm2d(emb_size),
        #     nn.ELU(),
        #     # nn.AvgPool2d((pool_size_time_2, 1), stride=(pool_stride_time_2, 1)),
        #     nn.Dropout(dropout),
        #     Rearrange('b c h w -> b (h w) c'),
        # )

        self.classifier = nn.Sequential(
            # Reduce('b n e -> b e', reduction='mean'),
            Reduce('b c n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
            nn.LogSoftmax(dim=1)
        )

        # # self.pos_embedding = nn.Parameter(torch.randn(window_size[0], window_size[1], 1))
        # self.pos_embedding = nn.Parameter(torch.randn(1, 20, 22))
        self.pos_embedding = PositionalEmbedding(emb_size, 22)
        self.pos_drop =  nn.Dropout(0.1)

        self.pos_embedding_t = PositionalEmbedding(emb_size, 100)

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
        x = self.conv_t(x)

        # x = self.pooling(x)

        x = self.patch_embedding(x)

        # x = window_partition(x, self.window_size)
        x = self.pos_drop(self.pos_embedding(x))
        x = self.window_attn(x)
        x = self.ff(x)
        # x = window_reverse(x, self.window_size, 100, 22)
        # print(x.size())
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = self.patch_embedding_t(x)
        x = self.conv(x)
        # print(x.size())
        # print(x.size())
        x = x.unsqueeze(1)
        # x = rearrange(x, 'b h w c -> b w h c')
        x = self.pos_drop(self.pos_embedding_t(x))
        x = self.window_attn_t(x)
        x = self.ff_t(x)

        # print(x.size())
        
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = self.pos_drop(self.pos_embedding_t(x))
        # x = self.window_attn_t(x)
        # x = self.ff_t(x)
        # x = rearrange(x, 'b c h w -> b h w c')

        # x = self.patch_merging(x)
        # # x = window_partition(x, self.window_size)
        # x = self.window_attn_2(x)
        # x = self.ff_2(x)
        # x = window_reverse(x, self.window_size, 50, 22)
        
        # x = self.patch_merging_2(x)
        # # x = window_partition(x, self.window_size)
        # x = self.window_attn_3(x)
        # x = self.ff_3(x)
        # x = window_reverse(x, self.window_size, 25, 22)
        # print(x.size())
        x = self.conv_s(x)
        # print(x.size())
        # x = self.conv_post(x)

        out = self.classifier(x) 
        return out

x = torch.ones(16, 1, 1000, 22)
model = LiteViT(16, 4, (1, 22), n_head=8)
x = model(x)
print(x.size())
print(sum(x.numel() for x in model.parameters()))


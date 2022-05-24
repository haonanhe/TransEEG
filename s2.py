import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log, square
from common.torchutils import DepthwiseConv2d
from common.torchutils import Conv2dNormWeight

from attention import Attention#, ConvAttention
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
        self.conv_expand = nn.Conv2d(dim, 10*dim, kernel_size=1, stride=1, bias=False)
        self.conv_depthwise = DepthwiseConv2d(10*dim, 1, kernel_size=(3, 3), padding='same', bias=False) # kernel_size很重要，不能是(3, 3)
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
        # x = self.dropout(self.activation(self.conv_expand(x)))
        # x = self.conv_depthwise(x)
        # x = self.dropout(self.activation(self.conv_reduce(x)))
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
    def __init__(self, d_model, dim, depth, n_head, dropout = 0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, ConvAttention(dim, d_model, n_head, dropout)),
                PreNorm(d_model, MBConv(dim, dropout=dropout)),
                # PreNorm(d_model, MultiHeadAttention(d_model, n_head, dropout)),
                # PreNorm(d_model, PositionwiseFeedForward(d_model, dropout=dropout)),
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# class LiteViT(nn.Module):
#     def __init__(self, dim, emb_size, n_classes, n_head=1, n_layers=1, dropout=0.5):
#         super().__init__()
#         self.conv_pre = nn.Sequential(
#             nn.Conv2d(1, 8, (25, 1), padding='same',bias=False),
#             nn.BatchNorm2d(8),
#             nn.Conv2d(8, emb_size, (1, 22), bias=False), 
#             nn.BatchNorm2d(emb_size),
#             nn.Dropout(dropout),

#             Rearrange('b c h w -> b w h c')
#         ) 

#         # self.down1 = nn.Conv2d(1, 1, (5, 1), stride=(5, 1))
#         # self.down2 = nn.Conv2d(1, 1, (5, 1), stride=(5, 1))
#         # self.down3 = nn.Conv2d(1, 1, (5, 1), stride=(5, 1))
#         self.down1 = nn.AvgPool2d((5, 1), stride=(5, 1))
#         self.down2 = nn.AvgPool2d((5, 1), stride=(5, 1))
#         self.down3 = nn.AvgPool2d((5, 1), stride=(5, 1))
#         self.transformer1 = Transformer(emb_size, dim, 1, n_head, dropout)
#         self.transformer2 = Transformer(emb_size, dim, 1, n_head, dropout)
#         self.transformer3 = Transformer(emb_size, dim, 1, n_head, dropout)

#         self.cls = nn.Sequential(
#             # Reduce('b c n e -> b e', reduction='mean'),
#             # nn.LayerNorm(emb_size),
#             # nn.Linear(emb_size, n_classes),
#             # nn.LogSoftmax(dim=1),

#             # Rearrange('b c h w -> b w h c'),
#             # Expression(square),
#             # Expression(safe_log),
#             # nn.Dropout(dropout), 
#             # Conv2dNormWeight(emb_size, n_classes, (5, 1), max_norm=0.5),
#             # nn.LogSoftmax(dim=1)

#             Rearrange('b c h w -> b w h c'),
#             Conv2dNormWeight(emb_size, n_classes, (8, 1), max_norm=0.5),
#             nn.AdaptiveAvgPool2d(1),
#             nn.LogSoftmax(dim=1)
#         )

#         self.pos_embedding = PositionalEmbedding(emb_size, 1000)
#         self.pos_drop =  nn.Dropout(0.1)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight, gain=1)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)

#     def forward(self, x):
#         x = self.conv_pre(x)

#         x = self.pos_drop(self.pos_embedding(x))
#         # x = rearrange(x, 'b c h w -> (b c) h w')
#         x = self.down1(x)
#         x = self.transformer1(x)

#         x = self.down2(x)
#         x = self.transformer2(x)

#         x = self.down3(x)
#         x = self.transformer3(x)

#         x = self.cls(x)

#         x = x[:, :, 0, 0]

#         return x

# conv attention
class ConvAttention(nn.Module):
    def __init__(self, dim, d_model, n_head, dropout=0.5):
        super(ConvAttention, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dim = dim

        # self.w_q = nn.Linear(d_model, d_model)
        # temporal filtering
        self.conv_in = nn.Sequential(
            nn.Conv2d(dim, 3*n_head*dim, (1, self.n_head), (1, self.n_head), groups=dim, bias=False),
            nn.BatchNorm2d(3*n_head*dim),
            nn.Dropout(0.5),
        )    

        self.conv_out = nn.Sequential(
            SeparableConv2d(
                n_head*dim, n_head*dim, (1, 1),  
                padding='same', bias=False
            ),
        )

        # self.w_out = nn.Linear(d_model, d_model)
        
        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):

        batch_size = x.size(0)
        
        # query = self.w_q(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        x = self.conv_in(x)
        x = rearrange(x, 'b (n c) h w -> n b c h w', n=3)
        query, key, value = x[0], x[1], x[2]
        # key, value = x[0], x[1]
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = self.conv_out(x)
        # x = x.squeeze(1)
        x = rearrange(x, 'b (c n) h w -> b c h (n w)', n=self.n_head)
        # x = self.w_out(x)
        return x

class LiteViT(nn.Module):
    def __init__(self, dim, emb_size, n_classes, n_head=1, n_layers=1, dropout=0.5):
        super().__init__()
        self.conv_pre = nn.Sequential(
            nn.Conv2d(1, 8, (25, 1), padding='same',bias=False),
            nn.BatchNorm2d(8),
            # nn.Conv2d(8, 8*emb_size, (1, 22), bias=False), 
            nn.Conv2d(8, 8*emb_size, (1, 22), groups=8, bias=False),
            nn.BatchNorm2d(8*emb_size),
            nn.Dropout(dropout),

            # Rearrange('b (c n) h w -> b c h (w n)', n=emb_size)
            # Rearrange('b (c n) h w -> (b c) h (w n)', n=emb_size)
            Rearrange('b (c e) h w -> (b c) w h e', e=emb_size)
        ) 

        # self.conv_pre = nn.Sequential(
        #     nn.Conv2d(1, emb_size, (1, 22), bias=False), 
        #     nn.BatchNorm2d(emb_size),
        #     nn.Dropout(dropout),

        #     Rearrange('b c h w -> b w h c')
        # ) 

        # self.down1 = nn.AvgPool2d((5, 1), stride=(5, 1))
        # self.down2 = nn.AvgPool2d((5, 1), stride=(5, 1))
        # self.down3 = nn.AvgPool2d((5, 1), stride=(5, 1))
        self.down1 = nn.MaxPool2d((5, 1), stride=(5, 1))
        self.down2 = nn.MaxPool2d((5, 1), stride=(5, 1))
        self.down3 = nn.MaxPool2d((5, 1), stride=(5, 1))
        self.transformer1 = Transformer(emb_size, 1, 1, n_head, dropout)
        self.transformer2 = Transformer(emb_size, 1, 1, n_head, dropout)
        self.transformer3 = Transformer(emb_size, 1, 1, n_head, dropout)

        self.cls = nn.Sequential(
            # Reduce('b c n e -> b e', reduction='mean'),
            # nn.LayerNorm(emb_size),
            # nn.Linear(emb_size, n_classes),
            # nn.LogSoftmax(dim=1),

            # Rearrange('b c h w -> b w h c'),
            # Expression(square),
            # Expression(safe_log),
            # nn.Dropout(dropout), 
            # Conv2dNormWeight(emb_size, n_classes, (5, 1), max_norm=0.5),
            # nn.LogSoftmax(dim=1)

            # Rearrange('b c h w -> b w h c'),
            # Rearrange('(b c) h w -> b c h w', c=8),
            Rearrange('(b c) w h e -> b c (w h) e', c=8),
            Conv2dNormWeight(8, n_classes, (8, emb_size), max_norm=0.5),
            nn.AdaptiveAvgPool2d(1),
            nn.LogSoftmax(dim=1)
        )

        self.pos_embedding = PositionalEmbedding(emb_size, 1000)
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
        x = self.conv_pre(x)

        x = self.pos_drop(self.pos_embedding(x))
        # x = rearrange(x, 'b c h w -> (b c) h w')
        x = self.down1(x)
        x = self.transformer1(x)

        x = self.down2(x)
        x = self.transformer2(x)

        x = self.down3(x)
        x = self.transformer3(x)

        x = self.cls(x)

        x = x[:, :, 0, 0]

        return x

x = torch.ones(16, 1, 1000, 22)
model = LiteViT(1, 16, 4, n_head=8, n_layers=1)
x = model(x)
print(x.size())
print(sum(x.numel() for x in model.parameters()))

class LiteCSPNet(nn.Module):
    """
    ConvNet model mimics CSP
    """
    def __init__(
        self, n_timepoints, n_channels, n_classes, dropout = 0.5,
        n_filters_t = 20, filter_size_t = 25,
        n_filters_s = 2, filter_size_s = -1,
        pool_size_1 = 75, pool_stride_1 = 15,
        n_filters_f = 16, filter_size_f = 16,
        pool_size_2 = 8, pool_stride_2 = 8,
    ):
        super().__init__()
        assert filter_size_t <= n_timepoints, "Temporal filter size error"
        if filter_size_s <= 0: filter_size_s = n_channels

        self.features = nn.Sequential(
            # temporal filtering
            nn.Conv2d(1, n_filters_t, (filter_size_t, 1), padding=(filter_size_t//2, 0), bias=False),
            nn.BatchNorm2d(n_filters_t),
            # spatial filtering
            nn.Conv2d(
                n_filters_t, n_filters_t*n_filters_s, (1, filter_size_s), 
                groups=n_filters_t, bias=False
            ),
            nn.BatchNorm2d(n_filters_t*n_filters_s),
            nn.Dropout(dropout),

            Rearrange('b c h w -> b w h c'),
            PositionalEmbedding(n_filters_t*n_filters_s, 1000),
            nn.Dropout(0.1),
            nn.AvgPool2d((5, 1), stride=(5, 1)),
            Transformer(n_filters_t*n_filters_s, 1, 1, 8, dropout),
            nn.AvgPool2d((5, 1), stride=(5, 1)),
            Transformer(n_filters_t*n_filters_s, 1, 1, 8, dropout),
            nn.AvgPool2d((5, 1), stride=(5, 1)),
            Transformer(n_filters_t*n_filters_s, 1, 1, 8, dropout),
            Rearrange('b c h w -> b w h c'),

            # Expression(square),
            # nn.AvgPool2d((pool_size_1, 1), stride=(pool_stride_1, 1)),
            # Expression(safe_log),
            # nn.Dropout(dropout), 
            
        )

        n_features = (n_timepoints - pool_size_1) // pool_stride_1 + 1
        n_filters_out = n_filters_t * n_filters_s
        self.classifier = nn.Sequential(
            Conv2dNormWeight(n_filters_out, n_classes, (8, 1), max_norm=0.5),
            nn.AdaptiveAvgPool2d(1),
            nn.LogSoftmax(dim=1)
            
            # Reduce('b c n e -> b e', reduction='mean'),
            # nn.LayerNorm(n_filters_t*n_filters_s),
            # nn.Linear(n_filters_t*n_filters_s, n_classes),
            # nn.LogSoftmax(dim=1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x[:, :, 0, 0]
        return x

# x = torch.ones(16, 1, 1000, 22)
# model = LiteCSPNet(n_timepoints=1000, n_channels=22, n_classes=4, dropout = 0.5)
# x = model(x)
# print(x.size())
# print(sum(x.numel() for x in model.parameters()))
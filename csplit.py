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
        return self.pe #+ x

class CSPProjection(nn.Module):
    def __init__(self, emb_size, filter_size_time, filter_size_spatial, pool_size_time=4, pool_stride_time=4, dropout=0.3):
        super(CSPProjection, self).__init__()
        self.projection = nn.Sequential(
            # temporal
            nn.Conv2d(1, emb_size, (filter_size_time, 1), padding=(filter_size_time//2, 0), bias=False),
            nn.BatchNorm2d(emb_size),
            # depthwise spatial 
            # DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), bias=False),
            Conv2dNormWeight(
                emb_size, emb_size, (1, filter_size_spatial), 
                max_norm=1, groups=emb_size, bias=False
            ),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            nn.Dropout(dropout),
            # nn.AvgPool2d((n_head, 1))
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        return self.projection(x)

class CSPTrans(nn.Module):
    def __init__(self, emb_size, n_head, filter_size_time, filter_size_spatial, pool_size_time=4, pool_stride_time=4, dropout=0.5):
        super(CSPTrans, self).__init__()
        self.projection = nn.Sequential(
            # PatchMerging(10, 1),

            # nn.Linear(22, 10),

            nn.Conv2d(1, 11, (100, 1), padding='same'),
            nn.BatchNorm2d(11),
            # nn.Dropout(0.1),
            
            PatchEmbedding(10),

            # ResidualAdd(
            #     PreNorm(22, ConvAttention(d_model=22, n_head=1, dropout=0.1, kernel_size=(100, 1)))
            # ),

            # ResidualAdd(
            #     PreNorm(22, MultiHeadAttention(22, n_head=2, dropout=dropout))
            # ),
            # ResidualAdd(
            #     PreNorm(22, PositionwiseFeedForward(22, dropout=dropout))
            # ),

            # nn.Conv2d(1, 10, (5, 22), stride=(5, 1)),
            # Rearrange('b (n c) h w -> b n (w h) c', n=1),

            # ResidualAdd(nn.Sequential(
            #     Rearrange('b c h w -> b c w h'),
            #     nn.LayerNorm(1000),
            #     channel_attention(1000, 30),
            #     nn.Dropout(0.1),
            #     Rearrange('b c h w -> b c w h'),
            # )),
            
            
            # Rearrange('b c h w -> b c w h'),
            # ResidualAdd(
            #     PreNorm(5, WindowAttention(5, 1, (22, 5)))
            # ),
            # ResidualAdd(
            #     PreNorm(5, PositionwiseFeedForward(5, dropout=dropout))
            # ),
            

            # nn.Linear(1000, 22),
            # ResidualAdd(
            #     PreNorm(22, MultiHeadAttention(22, n_head=1, dropout=dropout))
            # ),
            # ResidualAdd(
            #     PreNorm(22, PositionwiseFeedForward(22, dropout=dropout))
            # ),

            # PositionalEmbedding(1000, 22),
            # ResidualAdd(
            #     PreNorm(1000, ConvAttention(dim=11, d_model=1000, n_head=1, dropout=0.1, kernel_size=(100, 1)))
            # ),
            # ResidualAdd(
            #     PreNorm(1000, MBConv(11, dropout=dropout))
            # ),

            # PositionwiseFeedForward(22, dropout=dropout),

            DepthwiseConv2d(11, 1, (1, 22), bias=False),
            nn.BatchNorm2d(11),
            Expression(cov),
            Expression(safe_log),
            nn.Dropout(0.1),
            Rearrange('b c h w -> b h w c'),

            # PositionwiseFeedForward(11, dropout=dropout)

        )

    def forward(self, x):
        return self.projection(x)

# conv multi-head attention
class ConvAttention(nn.Module):
    def __init__(self, dim, d_model, n_head, dropout=0.5, kernel_size=(100, 1)):
        super(ConvAttention, self).__init__()
        # extract temporal features from input
        self.conv_in = nn.Conv2d(dim, 3 * n_head, kernel_size, padding='same', bias=False)
        # self-attention
        self.attention = Attention()
        # point-wise conv
        self.conv_out = nn.Conv2d(n_head, dim, kernel_size=1, stride=1, bias=False)
        # self.w_out = nn.Linear(n_head*d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        conv = self.conv_in(x)

        x = rearrange(conv, 'b (a c) h w -> a b c h w', a=3)

        query, key, value = x[0], x[1], x[2]
        
        if mask is not None:
            mask = mask.unsqueeze(1) 

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # x = torch.cat((x, conv), 1)
        # x = rearrange(x, 'b c h w -> b h (c w)')
        # x = self.w_out(x)
        # x = x.unsqueeze(1)

        # x = x + conv
        # x = torch.cat((x, conv), 1)

        x = self.conv_out(x)

        return x

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
        self.reduction = nn.Linear(resolution*dim, resolution*dim//2, bias=False)
        self.norm = nn.LayerNorm(resolution*dim)

    def forward(self, x):
        tmp = x[:, :, 0::self.resolution, :]
        for i in range(1, self.resolution):
            tmp = torch.cat([tmp, x[:, :, i::self.resolution, :]], 1)
        
        tmp = rearrange(tmp, 'b c h w -> b h w c')
        tmp = self.norm(tmp)
        tmp = self.reduction(tmp)

        tmp = rearrange(tmp, 'b h w c -> b c h w')

        return tmp

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.n_head = n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

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

        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        x = self.w_out(x)
        x = x.view(batch_size, n_channels, -1, self.d_model)
        
        return x

# Lite multi-head attention
class LiteAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.5, kernel_size=(4, 1), resolution=4):
        super(LiteAttention, self).__init__()
        # extract temporal features from input
        self.conv_in = DepthwiseConv2d(resolution//2, 1, kernel_size, padding='same', bias=False)
        # multi-head attention
        self.multi_attn = MultiHeadAttention(d_model, n_head)
        # point-wise conv
        self.conv_out = nn.Conv2d(resolution * n_head, resolution, kernel_size=1, stride=1, bias=False)
        self.w_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        # n_channels = x.size(1)

        # conv = self.conv_in(x[:, :n_channels//2, :, :])
        # conv = self.dropout(conv)
        
        # attn = self.multi_attn(x[:, n_channels//2:, :, :])

        # x = torch.cat((conv, attn), 1)

        x = self.multi_attn(x)
 
        # x = self.conv_out(x)
        # x = self.dropout(x)
        # x = self.w_out(x)

        return x

# Lite multi-head attention
class convAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.5, kernel_size=(4, 1), resolution=4):
        super(convAttention, self).__init__()
        # extract temporal features from input
        self.conv_in = DepthwiseConv2d(1, 3, kernel_size, padding='same', bias=False)
        # multi-head attention
        self.multi_attn = MultiHeadAttention(d_model, n_head)
        # point-wise conv
        self.conv_out = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
        self.w_out = nn.Linear(d_model, d_model)

        # self-attention
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        x = self.conv_in(x)
        x = rearrange(x, 'b c h w -> c b h w')
        query, key, value = x[0], x[1], x[2]
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        if mask is not None:
            mask = mask.unsqueeze(1) 

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # n_channels = x.size(1)

        # conv = self.conv_in(x[:, :n_channels//2, :, :])
        # conv = self.dropout(conv)
        
        # attn = self.multi_attn(x[:, n_channels//2:, :, :])

        # x = torch.cat((conv, attn), 1)

        # x = self.multi_attn(x)
 
        x = self.conv_out(x)
        # x = self.dropout(x)
        # x = self.w_out(x)

        return x

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
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, emb_size, (10, 22), stride=(10, 1)),
            # nn.BatchNorm2d(emb_size),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(emb_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x

class LiteViT(nn.Module):
    def __init__(self, n_timepoints, n_channels, d_model, n_classes, n_head=1, n_layers=1, dropout=0.5, 
                 kernel_size_t=(4, 1), kernel_size_s=(4, 1), resolution_t=10, resolution_s=10):
        super().__init__()

        # self.patch_merging_t = PatchMerging(resolution_t, 1)
        # self.patch_merging_s = PatchMerging(resolution_s, resolution_t//2)

        # # self.pos_embedding = PositionalEmbedding(d_model=n_channels, max_len=n_timepoints//resolution_t)
        # self.pos_embedding = nn.Parameter(torch.randn(1, n_timepoints//resolution_t, n_channels))
        # self.pos_drop =  nn.Dropout(0.1)

        # self.attn_t = ResidualAdd(
        #     PreNorm(n_channels, LiteAttention(n_channels, n_head, dropout, kernel_size_t, resolution_t//2))
        # )

        # self.attn_s = ResidualAdd(
        #     PreNorm(d_model, LiteAttention(d_model, n_head, dropout, kernel_size_s, resolution_s*resolution_t//4))
        # )

        # self.ff = ResidualAdd(
        #     # PreNorm(d_model, MBConv(resolution_s*resolution_t//4, dropout = dropout))
        #     PreNorm(d_model, PositionwiseFeedForward(d_model, dropout = dropout))
        # )

        self.classifier = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            # Reduce('b c n e -> b e', reduction='mean'),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=1)

            # Conv2dNormWeight(11, n_classes, (1, 1), max_norm=0.5),
            # nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

        # self.patch_embedding = PatchEmbedding(10)

        # self.csp_projection = CSPProjection(10, 25, 22, 4, 4, dropout=dropout)

        self.csp_trans = CSPTrans(10, 1, 25, 10, 50, 50, dropout=dropout)

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
        # x = self.patch_merging_t(x)

        # pe = self.pos_embedding(x)[:, :x.size(-2), :x.size(-1)]
        # pe = self.pos_embedding[:, :x.size(-2), :x.size(-1)]
        # x = self.pos_drop(x + pe)
        # x = self.attn_t(x)

        # x = self.patch_merging_s(x)
        # x = rearrange(x, 'b c h w -> b c w h')

        # x = self.patch_embedding(x)
        # x = x.unsqueeze(2)

        # x = self.attn_s(x)

        x = window_partition(x, (5, 22))

        x = self.csp_trans(x)

        x = window_reverse(x, (5, 22))

        # x = x.unsqueeze(1)
        # x = self.attn_s(x)
        # x = rearrange(x, 'b c h w -> b (c h) w')

        x = rearrange(x, 'b c h w -> b (c h) w')
        out = self.classifier(x) 
        return out

x = torch.ones(16, 1, 1000, 22)
model = LiteViT(1000, 22, 11, 4)
x = model(x)
print(x.size())

# from motorimagery.convnet import CSPNet, EEGNet
# from motorimagery.transformer import EEGTransformer
# model = EEGTransformer(1000, 22, 4)

# from EEG_Transformer.Trans import ViT 
# model = ViT(emb_size=10, depth=1, n_classes=4)
# x = torch.ones(16, 1, 16, 1000)
# x = model(x)
# print(x.size())

print(sum(x.numel() for x in model.parameters()))

# for name, layer in model._modules.items():
#     print(name)
#     print(layer)
import torch
import torch.nn as nn

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, safe_log, square
from common.torchutils import DepthwiseConv2d

from attention import ConvMultiHeadAttention, Attention
from attention import MultiHeadAttention

from common.torchutils import SeparableConv2d
from utils import _get_activation_fn
from model import PositionalEmbedding

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from attention import Attention

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

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

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

# conv positionwise feedforward
class ConvPositionwiseFeedForward(nn.Module):
    def __init__(self, resolution, dropout=0.5, activation='gelu'):
        super(ConvPositionwiseFeedForward, self).__init__()
        self.conv = SeparableConv2d(resolution, resolution, (3, 3), padding='same', bias=False, depth_multiplier=4)
        self.conv_1 = nn.Conv2d(resolution, resolution, kernel_size=(3, 3), padding='same', bias=False)
        self.conv_2 = nn.Conv2d(resolution, resolution, kernel_size=(3, 3), padding='same', bias=False)
        self.norm = nn.BatchNorm2d(resolution)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        # x = self.dropout(self.activation(self.norm(self.conv_1(x))))
        # x = self.dropout(self.conv_2(x))
        x = self.dropout(self.activation(self.conv(x)))
        return x

# positionwise feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
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

# conv multi-head attention
class ConvAttention(nn.Module):
    def __init__(self, n_head, dropout=0.5, kernel_size=(4, 1), resolution=4):
        super(ConvAttention, self).__init__()
        # extract temporal features from input
        self.conv_in = DepthwiseConv2d(resolution, 3 * n_head, kernel_size, padding='same', bias=False)
        # self-attention
        self.attention = Attention()
        # point-wise conv
        self.conv_out = nn.Conv2d(resolution * n_head, resolution, kernel_size=1, stride=1, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        conv = self.conv_in(x)

        x = rearrange(conv, 'b (a c) h w -> a b c h w', a=3)

        query, key, value = x[0], x[1], x[2]
        
        if mask is not None:
            mask = mask.unsqueeze(1) 

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # x = torch.cat((x, conv), 1)

        x = self.conv_out(x)

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

class ConvTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_head, dropout=0.5, kernel_size=(4, 1), resolution=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, ConvAttention(n_head, dropout, kernel_size, resolution)),
                PreNorm(d_model, ConvPositionwiseFeedForward(resolution, dropout))
                # PreNorm(d_model, PositionwiseFeedForward(d_model, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ConvViT(nn.Module):
    def __init__(self, n_timepoints, n_channels, d_model, n_classes, n_head=4, n_layers=1, dropout=0.5, 
                 kernel_size_t=(3, 3), kernel_size_s=(3, 3), resolution_t=10, resolution_s=10):
        super().__init__()

        self.patch_merging_t = PatchMerging(resolution_t, 1)
        self.patch_merging_s = PatchMerging(resolution_s, resolution_t//2)

        self.pos_embedding = PositionalEmbedding(d_model=n_channels, max_len=n_timepoints//resolution_t)
        self.pos_drop =  nn.Dropout(0.1)

        self.attn_t = ResidualAdd(
            PreNorm(n_channels, ConvAttention(n_head, dropout, kernel_size_t, resolution_t//2))
        )

        self.attn_s = ResidualAdd(
            PreNorm(d_model, ConvAttention(n_head, dropout, kernel_size_s, resolution_s*resolution_t//4))
        )
        # self.transformer = ConvTransformer(d_model, n_layers, n_head, dropout, kernel_size_s, resolution_s*resolution_t)

        self.ff = ResidualAdd(
            # PreNorm(d_model, ConvPositionwiseFeedForward(resolution_s*resolution_t//4, dropout))
            # PreNorm(d_model, PositionwiseFeedForward(d_model, dropout = dropout))

            PreNorm(d_model, MBConv(resolution_s*resolution_t//4, dropout = dropout))
        )

        self.classifier = nn.Sequential(
            Reduce('b c n e -> b e', reduction='mean'),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_merging_t(x)

        pe = self.pos_embedding(x)[:, :x.size(-2), :x.size(-1)]
        x = self.pos_drop(x + pe)
        
        # x = self.attn_t(x)

        x = self.patch_merging_s(x)
        x = rearrange(x, 'b c h w -> b c w h')
        # x = self.attn_s(x)

        x = self.ff(x)

        out = self.classifier(x) 

        return out

# x = torch.ones(16, 1, 1000, 22)
# model = ConvViT(1000, 22, 10, 4)
# x = model(x)
# print(x.size())
# print(sum(x.numel() for x in model.parameters()))
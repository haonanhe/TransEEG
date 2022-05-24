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
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # position
        # x += self.positions
        return x

class LiteViT(nn.Module):
    def __init__(self, n_timepoints, n_channels, d_model, n_classes, n_head=1, n_layers=1, dropout=0.5, 
                 kernel_size_t=(4, 1), kernel_size_s=(4, 1), resolution_t=10, resolution_s=10):
        super().__init__()

        self.patch_merging_t = PatchMerging(resolution_t, 1)
        self.patch_merging_s = PatchMerging(resolution_s, resolution_t//2)

        # self.pos_embedding = PositionalEmbedding(d_model=n_channels, max_len=n_timepoints//resolution_t)
        self.pos_embedding = nn.Parameter(torch.randn(1, n_timepoints//resolution_t, n_channels))
        self.pos_drop =  nn.Dropout(0.1)

        self.attn_t = ResidualAdd(
            PreNorm(n_channels, LiteAttention(n_channels, n_head, dropout, kernel_size_t, resolution_t//2))
        )

        self.attn_s = ResidualAdd(
            PreNorm(d_model, LiteAttention(d_model, n_head, dropout, kernel_size_s, resolution_s*resolution_t//4))
        )

        self.ff = ResidualAdd(
            # PreNorm(d_model, MBConv(resolution_s*resolution_t//4, dropout = dropout))
            PreNorm(d_model, PositionwiseFeedForward(d_model, dropout = dropout))
        )

        self.classifier = nn.Sequential(
            Reduce('b c n e -> b e', reduction='mean'),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=1)
        )

        self._reset_parameters()

        self.patch_embedding = PatchEmbedding(10)

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
        x = rearrange(x, 'b c h w -> b c w h')

        x = self.patch_embedding(x)
        x = x.unsqueeze(2)

        # x = self.attn_s(x)

        x = self.ff(x)

        out = self.classifier(x) 

        return out

x = torch.ones(16, 1, 1000, 22)
model = LiteViT(1000, 22, 10, 4)
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

# print(sum(x.numel() for x in model.parameters()))

# for name, layer in model._modules.items():
#     print(name)
#     print(layer)
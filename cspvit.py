import torch
import torch.nn as nn

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, safe_log, square
from common.torchutils import DepthwiseConv2d

from attention import ConvMultiHeadAttention, Attention

from common.torchutils import SeparableConv2d
from utils import _get_activation_fn
from model import PositionalEmbedding

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from attention import Attention

class PatchMerging(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    def forward(self, x):
        tmp = x[:, :, 0::self.resolution, :]
        for i in range(1, self.resolution):
            tmp = torch.cat([tmp, x[:, :, i::self.resolution, :]], 1)

        return tmp

# conv positionwise feedforward
class ConvPositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, resolution, dropout=0.1, activation='gelu'):
        super(ConvPositionwiseFeedForward, self).__init__()
        d_ff = 4 * d_model
        # self.conv_1 = nn.Conv2d(resolution, resolution*4, kernel_size=1, stride=1, bias=False)
        # self.conv_2 = nn.Conv2d(resolution*4, resolution, kernel_size=1, stride=1, bias=False)
        self.conv_1 = nn.Conv2d(resolution, 4*resolution, kernel_size=(3, 3), padding='same')
        self.conv_2 = nn.Conv2d(4*resolution, resolution, kernel_size=(3, 3), padding='same')
        # self.conv = SeparableConv2d(
        #         1, 1, (3, 1), 
        #         padding=(3//2, 0), bias=False, depth_multiplier=4
        #         )
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv_2(self.dropout(self.activation(self.conv_1(x))))
        # x = self.dropout(self.activation(self.conv(x)))
        return x#.squeeze(1)

from common.torchutils import Conv2dNormWeight
class CSPProjection(nn.Module):
    def __init__(self, n_head, filter_size_time, filter_size_spatial, pool_size_time=4, pool_stride_time=4, dropout=0.3):
        super(CSPProjection, self).__init__()
        self.projection = nn.Sequential(
            # temporal
            nn.Conv2d(1, n_head, (filter_size_time, 1), padding=(filter_size_time//2, 0), bias=False),
            nn.BatchNorm2d(n_head),
            # depthwise spatial 
            # DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), bias=False),
            Conv2dNormWeight(
                n_head, n_head, (1, filter_size_spatial), 
                max_norm=1, groups=n_head, bias=False
            ),
            nn.BatchNorm2d(n_head),
            nn.ELU(),
            nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            nn.Dropout(dropout),
            # nn.AvgPool2d((n_head, 1))
        )

    def forward(self, x):
        return self.projection(x)



class PatchEmbedding(nn.Module):
    def __init__(self, patch_height, patch_width):
        super(PatchEmbedding, self).__init__()

        self.embedding = nn.Sequential(
            Rearrange('b h (n1 p1) (n2 p2)-> b h (n1 n2) (p1 p2)', p1 = patch_height, p2 = patch_width),
            # nn.Linear(patch_height*patch_width, d_head)
        )

    def forward(self, x):
        return self.embedding(x)

class CSPAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        # self.csp_q = nn.Sequential(
        #     CSPProjection(n_head, filter_size_time, filter_size_spatial),
        #     PatchEmbedding(patch_height, patch_width),
        # ) 
        # self.csp_k = nn.Sequential(
        #     CSPProjection(n_head, filter_size_time, filter_size_spatial),
        #     PatchEmbedding(patch_height, patch_width),
        # ) 
        # self.csp_v = nn.Sequential(
        #     CSPProjection(n_head, filter_size_time, filter_size_spatial),
        #     PatchEmbedding(patch_height, patch_width),
        # ) 
        self.n_head = n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention = Attention()
        self.w_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # x = rearrange(x, 'b c h w -> b c w h')
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        # print(query.size())
        # query = rearrange(query, 'b c (h w) -> b h c w', h=self.n_head)
        # key = rearrange(key, 'b c (h w) -> b h c w', h=self.n_head)
        # value = rearrange(value, 'b c (h w) -> b h c w', h=self.n_head)
        query = rearrange(query, 'b n c (h w) -> b (n h) c w', h=self.n_head)
        key = rearrange(key, 'b n c (h w) -> b (n h) c w', h=self.n_head)
        value = rearrange(value, 'b n c (h w) -> b (n h) c w', h=self.n_head)
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = rearrange(x, 'b (c h) n p -> b c n (h p)', h=self.n_head)
        x = self.w_out(x)
        # print(x.size())

        return x

# conv multi-head attention
class ConvAttention(nn.Module):
    def __init__(self, n_head, dropout=0.1, kernel_size=(25, 1), resolution=4):
        super(ConvAttention, self).__init__()
        # extract temporal features from input
        self.conv_in = DepthwiseConv2d(resolution, 3 * n_head, kernel_size, padding='same', bias=False)
        # self-attention
        self.attention = Attention()
        # point-wise conv
        self.conv_out = nn.Conv2d(resolution * n_head, resolution, kernel_size=1, stride=1, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        x = self.conv_in(x)

        x = rearrange(x, 'b (a c) h w -> a b c h w', a=3)

        query, key, value = x[0], x[1], x[2]
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
 
        x = self.conv_out(x)

        return x

# channel multi-head attention
class ChannelAttention(nn.Module):
    def __init__(self, n_head, dropout=0.1, kernel_size=(25, 1), resolution=4):
        super(ChannelAttention, self).__init__()

        self.pre_norm = nn.Sequential(
            Rearrange('b c h w -> b c w h'),
            nn.LayerNorm(250),
            Rearrange('b c h w -> b c w h')
        )
        # extract temporal features from input
        self.conv_in = DepthwiseConv2d(resolution, 3 * n_head, kernel_size, padding='same', bias=False)
        # self-attention
        self.attention = Attention()
        # point-wise conv
        self.conv_out = nn.Conv2d(resolution * n_head, resolution, kernel_size=1, stride=1, bias=False)

        self.norm_t = nn.LayerNorm(250)
        self.norm = nn.LayerNorm(22)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, eeg_epoch, mask=None):
        # x = self.pre_norm(eeg_epoch)
        x = eeg_epoch
        x = self.conv_in(x)

        x = rearrange(x, 'b (a c) h w -> a b c h w', a=3)

        query, key, value = self.norm(x[0]), self.norm(x[1]), self.norm(x[2])
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
 
        x = self.conv_out(x)

        return eeg_epoch + x

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

from utils import _get_activation_fn

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

from attention import MultiHeadAttention
class ConvTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_head, dropout = 0.1, kernel_size=(25, 1), resolution=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, ConvAttention(n_head, dropout, kernel_size, resolution)),
                # PreNorm(d_model, PositionwiseFeedForward(d_model, dropout = dropout))
                PreNorm(d_model, ConvPositionwiseFeedForward(d_model, resolution, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# # CSP-ViT              
# class CSPViT(nn.Module):
#     def __init__(self, n_timepoints, n_channels, d_model, n_classes, n_head = 4, n_layers = 2, dropout = 0.3, 
#                  filter_size_time=25, filter_size_spatial=22):
#         super().__init__()

#         d_head = d_model // n_head
#         patch_height, patch_width = d_head, 1
#         img_height = n_timepoints
#         img_width = n_channels // filter_size_spatial
#         n_patches = (img_height//patch_height) * (img_width//patch_width)

#         self.csp = CSPProjection(n_head, filter_size_time, filter_size_spatial, dropout=dropout)
#         self.patch_embedding = PatchEmbedding(patch_height, patch_width, d_head)

#         # self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=n_patches)
#         self.pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
#         self.pos_drop =  nn.Dropout(0.1)

#         self.transformer_s = CSPTransformer(n_channels, n_layers, 1, dropout)
#         self.transformer_t = CSPTransformer(d_model, n_layers, n_head, dropout)
        
#         self.classifier = nn.Sequential(
#             # nn.Conv2d(1, n_classes, kernel_size=1),
#             # nn.LogSoftmax(dim=1)
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, n_classes),
#             nn.LogSoftmax(dim=1)
#         )

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight, gain=1)
#                 # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         # x = rearrange(x, 'b c h w -> b c w h')
#         # x = rearrange(x, 'b c h w -> b (c h) w')
#         # x = self.transformer_s(x)
#         # x = x.unsqueeze(1)
#         x = self.csp(x) # (b_s, n_h, n_t, 1)
#         print(x.size())
#         x = self.patch_embedding(x) # (b_s, n_h, -1, d_h)
#         print(x.size())
#         x = rearrange(x, 'b c h w -> b h (c w)') # (b_s, -1, d_model)
#         pe = self.pos_embedding[:, :x.size(-2), :x.size(-1)]
#         x = self.pos_drop(x + pe)
#         x = self.transformer_t(x)
#         # x = x.unsqueeze(1).permute(0, 1, 3, 2) 
#         # print(x.size())
#         x = self.classifier(x) 
#         out = x
#         # out = x[:, :, 0, 0] # (bs, n_classes)
#         return out

class CSPViT(nn.Module):
    def __init__(self, n_timepoints, n_channels, d_model, n_classes, n_head = 4, n_layers = 1, dropout = 0.5, 
                 kernel_size_t=(4, 1), kernel_size_s=(4, 1), resolution_t=10, resolution_s=10):
        super().__init__()

        # d_head = d_model // n_head
        patch_height, patch_width = d_model, 1
        img_height = n_timepoints
        img_width = n_channels // kernel_size_s[0]
        n_patches = (img_height//patch_height) * (img_width//patch_width)

        self.norm = nn.LayerNorm(n_timepoints//resolution_t)

        self.patch_merging_t = PatchMerging(resolution_t)
        self.patch_merging_s = PatchMerging(resolution_s)

        self.conv_inter = nn.Sequential(
            nn.Conv2d(resolution_t*resolution_s, 1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1), 
            # nn.AvgPool2d((10, 1), stride=(10, 1)),
            nn.Dropout(dropout)
        ) 

        self.pos_embedding_t = PositionalEmbedding(d_model=n_channels, max_len=n_timepoints//resolution_t)
        # self.pos_embedding_s = PositionalEmbedding(d_model=d_model+1, max_len=n_patches+1)
        # self.pos_embedding_t = nn.Parameter(torch.randn(1, n_timepoints//resolution_t, n_channels))
        self.pos_drop =  nn.Dropout(0.1)

        self.channel_attn = ChannelAttention(n_head, dropout, kernel_size_t, resolution_t)
        # self.transformer_t = ConvTransformer(n_channels, n_layers, n_head, dropout, kernel_size_t, resolution_t)
        self.transformer_s = ConvTransformer(d_model, n_layers, n_head, dropout, kernel_size_s, resolution_s*resolution_t)

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
        pe = self.pos_embedding_t(x)[:, :x.size(-2), :x.size(-1)]
        # pe = self.pos_embedding_t[:, :x.size(-2), :x.size(-1)]
        x = self.pos_drop(x + pe)

        # x = self.transformer_t(x)
        x = self.channel_attn(x)
        # x = self.conv_inter(x)

        x = self.patch_merging_s(x)
        x = rearrange(x, 'b c h w -> b c w h')
        x = self.transformer_s(x)

        x = self.classifier(x) 
        out = x
        # out = x[:, :, 0, 0] # (bs, n_classes)
        return out

x = torch.ones(16, 1, 1000, 22)
model = CSPViT(1000, 22, 10, 4)
x = model(x)
print(x.size())
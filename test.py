# import torch
# import torch.nn as nn

# import sys
# sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
# from common.torchutils import Expression, safe_log, square
# from common.torchutils import DepthwiseConv2d

# from attention import ConvMultiHeadAttention, Attention

# x = torch.ones(16, 1, 22, 1000)
# n_filters_time = 16
# filter_size_time = 25


# # model = ConvMultiHeadAttention(max_len, d_model, n_head)
# # x, _ = model(x, x, x)
# # print(x.size())
# # x: torch.Size([16, 22, 1000])

# # conv multi-head attention
# class ConvMultiHeadAttention(nn.Module):
#     def __init__(self, max_len, d_model, n_head, emb_size, dropout=0.1, 
#                  filter_size_time=25, filter_size_spatial=22):#, pool_size_time=10, pool_stride_time=10):
#         super(ConvMultiHeadAttention, self).__init__()
#         assert d_model % n_head == 0
        
#         self.d_model = d_model #// pool_size_time
#         self.d_head = self.d_model // n_head
#         self.n_head = n_head
#         self.max_len = max_len

#         # get spatial temporal tokens
#         self.conv_q = nn.Sequential(
#             # temporal
#             nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
#             nn.BatchNorm2d(n_head),
#             # depthwise spatial 
#             DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), padding = 'same', bias=False),
#             nn.BatchNorm2d(n_head),
#             # nn.Dropout(dropout),
#             # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
#             nn.AvgPool2d((n_head, 1))
#         ) 
#         self.conv_k = nn.Sequential(
#             nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
#             nn.BatchNorm2d(n_head),
#             DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), padding = 'same', bias=False),
#             nn.BatchNorm2d(n_head),
#             # nn.Dropout(dropout),
#             # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
#             nn.AvgPool2d((n_head, 1))
#         ) 
#         self.conv_v = nn.Sequential(
#             nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
#             nn.BatchNorm2d(n_head),
#             DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), padding = 'same', bias=False),
#             nn.BatchNorm2d(n_head),
#             # nn.Dropout(dropout),
#             # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
#             nn.AvgPool2d((n_head, 1))
#         ) 
#         self.w_out = nn.Linear(d_model, d_model)
#         self.conv_out = nn.Conv2d(1, 1, kernel_size=1)

#         self.attention = Attention()
        
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, query, key, value, mask=None):
#         # query.size() = (batch_size, n_filters*n_channels, d_model)
#         batch_size = query.size(0)
#         query = query.view(batch_size, -1, self.d_model, self.max_len)
#         key = key.view(batch_size, -1, self.d_model, self.max_len)
#         value = value.view(batch_size, -1, self.d_model, self.max_len)

#         query = self.conv_q(query).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         key = self.conv_k(key).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         value = self.conv_v(value).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

#         # conv: torch.Size([16, 4, 1000, 22])
#         # pooling: ([16, 4, 250, 22])
#         # query: torch.Size([16, 22, 1000])
#         # query = self.pooling(self.conv_q(query)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
#         # key = self.pooling(self.conv_k(key)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         # value = self.pooling(self.conv_v(value)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

#         if mask is not None:
#             mask = mask.unsqueeze(1) 
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
#         x = x.unsqueeze(1)
#         x = self.conv_out(x)
#         x = x.view(batch_size, -1, self.n_head*self.d_head)
#         # torch.Size([16, 22, 1000])
 
#         # x = self.w_out(x)

#         return x, attn

# from utils import _get_activation_fn
# # conv positionwise feedforward
# class ConvPositionwiseFeedForward(nn.Module):
#     def __init__(self, d_ff, dropout=0.1, activation='gelu'):
#         super(ConvPositionwiseFeedForward, self).__init__()
#         self.conv_1 = nn.Conv2d(1, d_ff, kernel_size=1)
#         self.conv_2 = nn.Conv2d(d_ff, 1, kernel_size=1)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = _get_activation_fn(activation)
    
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv_2(self.dropout(self.activation(self.conv_1(x))))
#         return x.squeeze(1)

# # max_len = 22
# # d_model = 1000
# # n_head = 4
# # emb_size = 50
# # x = torch.ones(16, 22, 1000)
# # model = ConvMultiHeadAttention(max_len, d_model, n_head, emb_size)
# # # model = nn.Conv2d(1, 1, (25, 1), padding='same')
# # x, _ = model(x, x, x)
# # print(x.size())


# class CSPProjection(nn.Module):
#     def __init__(self, n_head, filter_size_time, filter_size_spatial):
#         super(CSPProjection, self).__init__()
#         self.projection = nn.Sequential(
#             # temporal
#             nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
#             nn.BatchNorm2d(n_head),
#             # depthwise spatial 
#             DepthwiseConv2d(n_head, n_head, (1, filter_size_spatial), bias=False),
#             nn.BatchNorm2d(n_head),
#             # nn.Dropout(dropout),
#             # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
#             nn.ELU(),
#             # nn.AvgPool2d((n_head, 1))
#         )

#     def forward(self, x):
#         return self.projection(x)

# from einops.layers.torch import Rearrange
# from einops import rearrange

# class PatchEmbedding(nn.Module):
#     def __init__(self, n_head, patchsize, d_model):
#         super(PatchEmbedding, self).__init__()
#         patch_height, patch_width = patchsize[0], patchsize[1]
#         patch_dim = n_head * patch_height * patch_width

#         self.embedding = nn.Sequential(
#             Rearrange('b c (n1 p1) (n2 p2)-> b (n1 n2) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, d_model),
#         )

#     def forward(self, x):
#         return self.embedding(x)

# class CSPAttention(nn.Module):
#     def __init__(self, n_head, d_model, filter_size_time=25, filter_size_spatial=22, patchsize=(25, 1)):
#         super().__init__()

#         patch_height, patch_width = patchsize[0], patchsize[1]

#         self.csp_q = nn.Sequential(
#             CSPProjection(n_head, filter_size_time, filter_size_spatial),
#             Rearrange('b h (n1 p1) (n2 p2)-> b h (n1 n2) (p1 p2)', p1 = patch_height, p2 = patch_width)
#         ) 
#         self.csp_k = nn.Sequential(
#             CSPProjection(n_head, filter_size_time, filter_size_spatial),
#             Rearrange('b h (n1 p1) (n2 p2)-> b h (n1 n2) (p1 p2)', p1 = patch_height, p2 = patch_width)
#         ) 
#         self.csp_v = nn.Sequential(
#             CSPProjection(n_head, filter_size_time, filter_size_spatial),
#             Rearrange('b h (n1 p1) (n2 p2)-> b h (n1 n2) (p1 p2)', p1 = patch_height, p2 = patch_width)
#         ) 

#         self.w_out = nn.Linear(n_head*patch_height, d_model)
    
#     def forward(self, x, mask=None):
#         query = self.csp_q(x)
#         key = self.csp_k(x)
#         value = self.csp_v(x)

#         if mask is not None:
#             mask = mask.unsqueeze(1) 
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

#         x = rearrange(x, 'b h n p -> b n (h p)')
#         x = self.w_out(x)

#         return x

# from model import *
# # CSP-ViT              
# class CSPViT(nn.Module):
#     def __init__(self, n_timepoints, n_channels, n_classes, 
#                  dropout = 0.1, n_head = 4, num_layers = 2, 
#                  filter_size_time=25, filter_size_spatial=22,
#                  patchsize=(25, 1)):
#         super().__init__()

#         patch_height, patch_width = patchsize[0], patchsize[1]

#         # # CSP projection
#         # self.csp = CSPProjection(n_head, filter_size_time, filter_size_spatial)
#         # self.n_timepoints = n_timepoints
#         # self.n_channels = 1

#         # # patch embedding
#         # self.patch_embedding = PatchEmbedding(n_head, patchsize, patch_height)
  
#         # positional encoding
#         n_patches = (self.n_timepoints//patch_height) * (self.n_channels//patch_width)
#         self.pos_embedding = PositionalEmbedding(d_model=patch_height+1, max_len=n_patches)
#         self.pos_drop = nn.Dropout(0.1)
#         # transformer
#         self.transformer = Transformer(
#             max_len = n_patches,
#             d_model = patch_height,
#             n_head = n_head,
#             num_layers = num_layers,
#             dropout=dropout
#         )
#         # classfier
#         self.classifier = nn.Sequential(
#             nn.Conv2d(1, n_classes, kernel_size=1),
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
#         # eeg_epoch: (bs, 1, n_timepoints, n_channels)
#         # temporal: max_len = n_channels, d_model = n_timepoints
#         # x = x.permute(0, 1, 3, 2) # (bs, 1, max_len, d_model)
#         # raw_shape = x.shape
#         x = self.csp(x)
#         x = self.patch_embedding(x)
#         x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, max_len, d_model)
#         pe = self.pos_embedding(x)[:, :, :x.size(-1)]
#         x = self.pos_drop(x + pe)
#         x = self.transformer(x) # (bs, max_len, d_model)
#         # x = x.view(raw_shape) # (bs, 1, max_len, d_model)
#         x = x.unsqueeze(1).permute(0, 1, 3, 2) # (bs, 1, d_model, max_len)
#         x = self.classifier(x) # (bs, n_classes, d_model, max_len)
#         out = x[:, :, 0, 0] # (bs, n_classes)
#         return out

# # model = CSPProjection(25, 22, 4, 2)
# # x = torch.ones(16, 1, 1000, 22)
# # x = model(x)
# # print(x.size())

# # imgsize = x.shape
# # patchsize = (25, 1)
# # model = PatchEmbedding(imgsize, patchsize, 25)
# # x = model(x)
# # print(x.size())

# # x = torch.ones(16, 1, 1000, 22)
# # model = CSPViT(1000, 22, 4)
# # x = model(x)
# # print(x.size())

# import sys
# sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
# from common.torchutils import DepthwiseConv2d
# import torch

# x = torch.ones(16, 4, 1000, 22)
# model = DepthwiseConv2d(4, 3, 1)
# x = model(x)
# print(x.size())

import torch.nn as nn
import torch

# x = torch.ones((16, 1, 1000, 22))
# model = nn.Conv2d(1, 1, (1, 1), dilation=10, stride=(10, 1))
# x = model(x)
# print(x.size())
# x = torch.ones((16, 23, 16))
# a = x[:, :22]
# print(a.shape)
import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from motorimagery.convnet import CSPNet, EEGNet
import matplotlib.pyplot as plt
model = EEGNet(n_timepoints=1000, n_channels=22, n_classes=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-09, weight_decay=0)

from scheduler import WarmupCosineLR, WarmupExponenLR, WarmupNoamLR
warm_up = 10
n_epochs = 1000
# scheduler = WarmupCosineLR(optimizer, 1e-9, 1e-2, warm_up, n_epochs, 1)
scheduler = WarmupExponenLR(optimizer, 1e-9, 1e-2, warm_up, gamma=0.98, start_ratio=1, last_epoch=-1)
# scheduler = WarmupNoamLR(optimizer, model_size=512, factor=1, warm_up=5, last_epoch=-1, verbose=False)

# from torch.optim.lr_scheduler import ExponentialLR
# steplr = ExponentialLR(optimizer, gamma=0.999, last_epoch=-1)
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=steplr)


# torch_lr_scheduler = ExponentialLR(optimizer, gamma=0.98)
# scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
#                                             warmup_start_value=0.0,
#                                             warmup_end_value=0.1,
#                                             warmup_duration=3)

lrs=[]
for epoch in range(n_epochs):
    lrs.append(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()
figpath = '/home/scutbci/public/hhn/Trans_EEG/training_results/figs/'
plt.plot(lrs)
plt.savefig(figpath + 'scheduler.png')

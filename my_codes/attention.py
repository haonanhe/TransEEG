import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import DepthwiseConv2d, SeparableConv2d, Conv2dNormWeight

# attention
class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        # size: (batch_size, n_head, -1, d_head)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask==0, 1e-9)
            
        p_attn = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn

# multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        
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
        
        query = self.w_q(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
        key = self.w_k(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.w_v(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)

        return self.w_out(x), attn

# conv attention
class ConvAttention(nn.Module):
    def __init__(self, dim, d_model, n_head, dropout=0.5):
        super(ConvAttention, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dim = dim

        self.conv_in = nn.Sequential(
            nn.Conv2d(dim, 3*n_head*dim, (5, 1), padding='same', groups=dim, bias=False), # temporal filter
            DepthwiseConv2d(3*n_head*dim, 1, (1, self.n_head), (1, self.n_head), bias=False), # split into heads
            nn.Dropout(0.3),
            # nn.Linear(self.d_head, self.d_head)
        )     

        self.w_out = nn.Linear(d_model, d_model)

        # may replace w_out with 1*1 conv
        # self.conv_out =SeparableConv2d(n_head*dim, n_head*dim, (1, 1),  padding='same', bias=False),
        
        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):

        x = self.conv_in(x)
        x = rearrange(x, 'b (n c) h w -> n b c h w', n=3)
        query, key, value = x[0], x[1], x[2]
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # x = self.conv_out(x)
        x = rearrange(x, 'b (c n) h w -> b c h (n w)', n=self.n_head)
        x = self.w_out(x)

        return x



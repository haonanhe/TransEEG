import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

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
        
        self.w_q = nn.Linear(d_model, d_model) # d_model = n_head * d_head 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.w_out = nn.Linear(d_model, d_model)
        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.w_q(query).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
        key = self.w_k(key).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        
        return self.w_out(x), attn

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import DepthwiseConv2d

# # conv multi-head attention
# class ConvMultiHeadAttention(nn.Module):
#     def __init__(self, max_len, d_model, n_head, dropout=0.1, filter_size_time=25, n_filters=1):
#         super(ConvMultiHeadAttention, self).__init__()
#         assert d_model % n_head == 0
        
#         self.d_model = d_model
#         self.d_head = d_model // n_head
#         self.n_head = n_head
#         self.max_len = max_len

#         # extract temporal features from input
#         self.conv_q = DepthwiseConv2d(n_filters, n_head, (filter_size_time, 1), padding='same', bias=False)
#         self.conv_k = DepthwiseConv2d(n_filters, n_head, (filter_size_time, 1), padding='same', bias=False)
#         self.conv_v = DepthwiseConv2d(n_filters, n_head, (filter_size_time, 1), padding='same', bias=False)
#         self.pooling = nn.AvgPool2d((n_head, 1))

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

#         # conv: torch.Size([16, 4, 1000, 22])
#         # pooling: ([16, 4, 250, 22])
#         # query: torch.Size([16, 22, 1000])
#         query = self.pooling(self.conv_q(query)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
#         key = self.pooling(self.conv_k(key)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         value = self.pooling(self.conv_v(value)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

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

# conv multi-head attention
class ConvMultiHeadAttention(nn.Module):
    def __init__(self, max_len, d_model, n_head, emb_size, dropout=0.1, filter_size_time=25, n_filters=1):
        super(ConvMultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.max_len = max_len

        # extract temporal features from input
        self.conv_q = DepthwiseConv2d(n_filters, n_head * emb_size, (filter_size_time, 1), bias=False)
        self.conv_k = DepthwiseConv2d(n_filters, n_head * emb_size, (filter_size_time, 1), bias=False)
        self.conv_v = DepthwiseConv2d(n_filters, n_head * emb_size, (filter_size_time, 1), bias=False)
        self.pooling = nn.AvgPool2d((n_head, 1))

        self.w_out = nn.Linear(d_model, d_model)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=1)

        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # query.size() = (batch_size, n_filters*n_channels, d_model)
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.d_model, self.max_len)
        key = key.view(batch_size, -1, self.d_model, self.max_len)
        value = value.view(batch_size, -1, self.d_model, self.max_len)

        # conv: torch.Size([16, 4, 1000, 22])
        # pooling: ([16, 4, 250, 22])
        # query: torch.Size([16, 22, 1000])
        query = self.pooling(self.conv_q(query)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
        key = self.pooling(self.conv_k(key)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.pooling(self.conv_v(value)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        x = x.unsqueeze(1)
        x = self.conv_out(x)
        x = x.view(batch_size, -1, self.n_head*self.d_head)
        # torch.Size([16, 22, 1000])
 
        # x = self.w_out(x)

        return x, attn

# channel attention
class channel_attention(nn.Module):
    def __init__(self, channels=22, sequence_num=1000, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(self.sequence_num / self.inter)  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3)
        )
        self.key = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.Dropout(0.3)
        )
        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.Dropout(0.3),
        )

        self.dropout = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        query = self.query(x).transpose(2, 3)
        key = self.key(x).transpose(2, 3)

        channel_query = self.pooling(query)
        channel_key = self.pooling(key)

        scale = math.sqrt(self.extract_sequence) 

        channel_attn = torch.matmul(channel_query, channel_key.transpose(-2, -1)) / scale

        channel_attn_score = F.softmax(channel_attn, dim=-1)
        channel_attn_score = self.dropout(channel_attn_score)
        out = torch.matmul(channel_attn_score, x.transpose(-2, -1))
        '''
        projections after or before multiplying with attention score are almost the same.
        '''
        out = out.permute(0, 1, 3, 2)
        out = self.projection(out)
        return out
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
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        query = self.w_q(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
        key = self.w_k(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.w_v(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)

        return self.w_out(x)#, attn

# # conv attention
# class ConvAttention(nn.Module):
#     def __init__(self, dim, d_model, n_head, dropout=0.1):
#         super(ConvAttention, self).__init__()
#         assert d_model % n_head == 0
        
#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_head = d_model // n_head

#         self.w_q = nn.Linear(d_model, d_model)
#         self.conv_in = nn.Sequential(
#             DepthwiseConv2d(dim, 2*n_head, (1, self.n_head), stride=(1, self.n_head), bias=False)
#         )
#         # self.conv_out = SeparableConv2d(n_head, 1, (1, d_model), padding='same', bias=False)
#         self.w_out = nn.Linear(d_model, d_model)
# #         self.conv_out = nn.Sequential(
# #             SeparableConv2d(dim*n_head, dim*n_head, (1, 1), padding='same', bias=False),
# # #             # nn.BatchNorm2d(dim),
# # #             # nn.ELU(),
# # #             # nn.Dropout(0.3),
# #         )
        
#         self.attention = Attention()
        
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, x, mask=None):
#         batch_size = x.size(0)

#         query = self.w_q(x).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
#         # query = x.view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

#         # x = rearrange(x, 'b h w -> b 1 h w')
#         x = self.conv_in(x)
#         x = rearrange(x, 'b (n c) h w -> n b c h w', n=2)
#         key, value = x[0], x[1]
        
#         if mask is not None:
#             mask = mask.unsqueeze(1) 
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
#         x = rearrange(x, 'b c h w -> b h (c w)')
#         x = self.w_out(x)
#         # x = rearrange(x, 'b 1 h w -> b h w')

#         # x = self.conv_out(x)
#         # x = rearrange(x, 'b (n c) h w -> b c h (n w)', n=self.n_head)

#         return x

# # conv attention
# class ConvAttention(nn.Module):
#     def __init__(self, dim, d_model, n_head, dropout=0.1):
#         super(ConvAttention, self).__init__()
        
#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_head = d_model // n_head
#         self.dim = dim

#         self.conv_in = DepthwiseConv2d(dim, 3*n_head, (1, n_head), padding='same', bias=False)
#         # self.conv_out = SeparableConv2d(dim*n_head, dim*n_head, 1, padding='same', bias=False)
#         # self.w_out = nn.Linear(d_model, d_model)
#         # self.conv_out = nn.Conv2d(dim*n_head, dim, (3, 3), padding='same', bias=False)
#         self.conv_out = nn.Sequential(
#             SeparableConv2d(dim*n_head, dim, (1, 1), padding='same', bias=False),
#             # nn.BatchNorm2d(dim),
#             # nn.ELU(),
#             # nn.Dropout(0.3),
#         )
        
#         self.attention = Attention()
        
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, x, mask=None):
#         batch_size = x.size(0)

#         x = self.conv_in(x)
#         x = rearrange(x, 'b (n c) h w -> n b c h w', n=3)
#         query, key, value = x[0], x[1], x[2]
        
#         if mask is not None:
#             mask = mask.unsqueeze(1) 
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

#         x = self.conv_out(x)
#         # x = rearrange(x, 'b (d n) h w -> b d h (n w)', n=self.n_head)

#         return x


# # conv attention
# class ConvAttention(nn.Module):
#     def __init__(self, dim, d_model, n_head, dropout=0.1):
#         super(ConvAttention, self).__init__()
        
#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_head = d_model // n_head
#         self.dim = dim

#         # temporal filtering
#         self.conv_in = nn.Sequential(
#             nn.Conv2d(dim, 3*n_head*dim, (1, n_head), padding='same', bias=False),
#             # nn.BatchNorm2d(3*n_head*dim),
#         )    
#         self.conv_s = nn.Sequential(
#             # Conv2dNormWeight(
#             #     3*n_head*dim, 3*n_head*dim, (4, 1), padding='same',
#             #     max_norm=1, groups=3*n_head*dim, bias=False
#             # ),
#             nn.Conv2d(
#                 3*n_head*dim, 3*n_head*dim, (4, 1), padding='same',
#                 groups=3*n_head*dim, bias=False
#             ),
#             nn.BatchNorm2d(3*n_head*dim),
#             # nn.GELU(),
#             nn.GELU(),
#             # nn.AvgPool2d((pool_size_time_1, 1), stride=(pool_stride_time_1, 1)),
#             nn.Dropout(dropout),
#         )

#         self.conv_out = nn.Sequential(
#             SeparableConv2d(
#                 n_head*dim, dim, (1, n_head),  
#                 padding='same', bias=False
#             ),
#             # nn.BatchNorm2d(dim),
#             # nn.ELU(),
#             # nn.AvgPool2d((pool_size_time_2, 1), stride=(pool_stride_time_2, 1)),
#             # nn.Dropout(0.3),
#         )
        
#         self.attention = Attention()
        
#         self.dropout = nn.Dropout(p=dropout)
        
#     def forward(self, x, mask=None):
#         batch_size = x.size(0)
#         x = x.unsqueeze(1)

#         x = self.conv_in(x)
#         # s = self.conv_s(x)
#         x = rearrange(x, 'b (n c) h w -> n b c h w', n=3)
#         query, key, value = x[0], x[1], x[2]
        
#         if mask is not None:
#             mask = mask.unsqueeze(1) 
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

#         # x = torch.concat((x, s), 1)
#         x = self.conv_out(x)
#         x = x.squeeze(1)
#         # x = rearrange(x, 'b (d n) h w -> b d h (n w)', n=self.n_head)
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
            nn.Conv2d(dim, 3*n_head*dim, (25, self.n_head), (1, self.n_head), padding=(25//2, 0), bias=False),
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

# # conv attention
# class ConvAttention(nn.Module):
#     def __init__(self, dim, d_model, n_head, dropout=0.1):
#         super(ConvAttention, self).__init__()
#         assert d_model % n_head == 0
        
#         self.d_model = d_model
#         self.n_head = n_head
#         self.d_head = d_model // n_head
#         self.dim = dim

#         self.conv_in = DepthwiseConv2d(dim, 3*n_head, (1, n_head), stride=(1, n_head), bias=False)
#         self.conv_out = nn.Conv2d(dim*n_head, dim*n_head, (3, 3), padding='same', bias=False)
#         # self.w_out = nn.Linear(d_model, d_model)
        
#         self.attention = Attention()
        
#         self.dropout = nn.Dropout(p=dropout)

#         self.dropout2 = nn.Dropout(0.2)
        
#     def forward(self, x, mask=None):
#         batch_size = x.size(0)

#         x = self.conv_in(x)
#         x = self.dropout2(x)
#         x = rearrange(x, 'b (n c) h w -> n b c h w', n=3)
#         query, key, value = x[0], x[1], x[2]
        
#         if mask is not None:
#             mask = mask.unsqueeze(1) 
#         x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
#         # x = rearrange(x, 'b c h w -> b h (c w)')
#         # # x = self.w_out(x)
#         # x = rearrange(x, 'b h w -> b 1 h w')

#         x = self.conv_out(x)
#         x = self.dropout2(x)
#         x = rearrange(x, 'b (d n) h w -> b d h (n w)', n=self.n_head)

#         return x

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import DepthwiseConv2d, SeparableConv2d, Conv2dNormWeight

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

class CSPProjection(nn.Module):
    def __init__(self, filter_size_time, filter_size_spatial, n_filter_time, n_filter_spatial):
        super(CSPProjection, self).__init__()
        self.projection = nn.Sequential(
            # temporal
            nn.Conv2d(1, n_filter_time, (filter_size_time, 1), padding='same', bias=False),
            nn.BatchNorm2d(n_filter_time),
            # depthwise spatial 
            DepthwiseConv2d(n_filter_time, n_filter_spatial, (1, filter_size_spatial), bias=False),
            nn.BatchNorm2d(n_filter_time*n_filter_spatial),
            # nn.Dropout(dropout),
            # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            nn.ELU(),
            # nn.AvgPool2d((n_head, 1))
        )

    def forward(self, x):
        return self.projection(x)
        
# conv multi-head attention
class ConvMultiHeadAttention(nn.Module):
    def __init__(self, max_len, d_model, n_head, dropout=0.1, 
                 filter_size_time=25, filter_size_spatial=22):#, pool_size_time=10, pool_stride_time=10):
        super(ConvMultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        
        self.d_model = d_model #// pool_size_time
        self.d_head = self.d_model // n_head
        self.n_head = n_head
        self.max_len = max_len

        # get spatial temporal tokens
        self.conv_q = nn.Sequential(
            # temporal
            nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
            nn.BatchNorm2d(n_head),
            # depthwise spatial 
            DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), padding = 'same', bias=False),
            nn.BatchNorm2d(n_head),
            # nn.Dropout(dropout),
            # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            nn.ELU(),
            nn.AvgPool2d((n_head, 1))
        ) 
        self.conv_k = nn.Sequential(
            nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
            nn.BatchNorm2d(n_head),
            DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), padding = 'same', bias=False),
            nn.BatchNorm2d(n_head),
            # nn.Dropout(dropout),
            # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            nn.ELU(),
            nn.AvgPool2d((n_head, 1))
        ) 
        self.conv_v = nn.Sequential(
            nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False),
            nn.BatchNorm2d(n_head),
            DepthwiseConv2d(n_head, 1, (1, filter_size_spatial), padding = 'same', bias=False),
            nn.BatchNorm2d(n_head),
            # nn.Dropout(dropout),
            # nn.AvgPool2d((pool_size_time, 1), stride=(pool_stride_time, 1)),
            nn.ELU(),
            nn.AvgPool2d((n_head, 1))
        ) 
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

        query = self.conv_q(query).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        key = self.conv_k(key).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.conv_v(value).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

        # conv: torch.Size([16, 4, 1000, 22])
        # pooling: ([16, 4, 250, 22])
        # query: torch.Size([16, 22, 1000])
        # query = self.pooling(self.conv_q(query)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
        # key = self.pooling(self.conv_k(key)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        # value = self.pooling(self.conv_v(value)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)

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
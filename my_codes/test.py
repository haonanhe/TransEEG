import torch
import torch.nn as nn

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, safe_log, square
from common.torchutils import DepthwiseConv2d

from attention import ConvMultiHeadAttention, Attention

x = torch.ones(16, 1, 22, 1000)
n_filters_time = 16
filter_size_time = 25


# model = ConvMultiHeadAttention(max_len, d_model, n_head)
# x, _ = model(x, x, x)
# print(x.size())
# x: torch.Size([16, 22, 1000])

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

        query = self.conv_q(query)
        print(query.size())
        # query = self.pooling(query)
        # print(query.size())
        query = query.view(batch_size, -1, emb_size, self.max_len)
        print(query.size())
        query = query.transpose(2, 2)
        print(query.size())

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

max_len = 22
d_model = 1000
n_head = 4
emb_size = 50
x = torch.ones(16, 22, 1000)
model = ConvMultiHeadAttention(max_len, d_model, n_head, emb_size)
# model = nn.Conv2d(1, 1, (25, 1), padding='same')
x, _ = model(x, x, x)
print(x.size())
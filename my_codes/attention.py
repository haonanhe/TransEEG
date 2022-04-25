import torch
import torch.nn as nn
import torch.nn.functional as F
# others
import math

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
    
# conv multi-head attention
class ConvMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, filter_size_time=25):
        super(ConvMultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        
        self.d_head = d_model // n_head
        self.n_head = n_head
        
        self.conv_q = nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False) 
        self.conv_k = nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False)
        self.conv_v = nn.Conv2d(1, n_head, (filter_size_time, 1), padding='same', bias=False)
        self.pooling = nn.MaxPool2d((1, n_head))
        self.w_out = nn.Linear(d_model, d_model)
        self.attention = Attention()
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        # query.size() = (batch_size, n_channels, d_model)
        batch_size = query.size(0)
        query = query.unsqueeze(1) # query.size() = (batch_size, 1, n_channels, d_model)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        # conv: (batch_size, n_head, n_channels, d_model)
        # pooling: (batch_size, n_head, n_channels, d_head)
        query = self.pooling(self.conv_q(query)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2) # transpose for attention
        key = self.pooling(self.conv_k(key)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.pooling(self.conv_v(value)).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1) 
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        
        return self.w_out(x), attn

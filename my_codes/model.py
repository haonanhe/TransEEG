from turtle import forward
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# others
import math
# self-defined
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log

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
        
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_head)
        
        return self.w_out(x)

# positionwise feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# choose activation functions
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.") 

# positional embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, maxlen).float().unsqueeze(1)
        dive_term = (torch.arange(0, d_model, 2).float * (-math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe[:, :x.size(1)]
        
# Transformer block   
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, normalize_before=False):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask = None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask = None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.feed_forward(src2)
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask)
        return self.forward_post(src, src_mask)

# Transformer network
class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, n_head: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5):
        super(Transformer, self).__init__()
        #
        d_ff = 4 * d_model
        
        self.d_model = d_model
        self.n_head = n_head

        encoder_layer = TransformerEncoderLayer(d_model, n_head, d_ff, dropout,
                                                activation, layer_norm_eps)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self._reset_parameters()

    def forward(self, src):
        output = self.encoder(src)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# EEGTransformer               
class EEGTransformer(nn.Module):
    def __init__(self, n_timepoints, n_channels, n_classes, dropout = 0.1,
                 n_filters_time = 16, filter_size_time = 25, n_filters_spatial = -1,
                 n_head = 8, num_layers = 2, ffn_dim = 512,):
        super().__init__()
        assert filter_size_time <= n_timepoints, "Temporal filter size error"
        if n_filters_spatial <= 0: n_filters_spatial = n_channels
        d_model = n_timepoints
        # temporal filtering
        self.feature_pre = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (filter_size_time, 1), padding='same', bias=False),
            nn.BatchNorm2d(n_filters_time),
        )
        n_patches = n_filters_time * n_channels
        # positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, d_model))
        self.pos_drop = nn.Dropout(dropout)
        # transformer
        self.transformer = Transformer(
            d_model = d_model,
            n_head = n_head,
            num_layers = num_layers,
            d_ff = ffn_dim,
        )
        # spatial filtering & calculate covariance
        self.feature_post = nn.Sequential(
            DepthwiseConv2d(n_filters_time, n_filters_spatial, (1, n_channels), bias=False),
            nn.BatchNorm2d(n_filters_time*n_filters_spatial),
            Expression(cov),
            Expression(safe_log),
            nn.Dropout(dropout),
        )
        n_features = n_filters_time * n_filters_spatial
        # classfier
        self.classifier = nn.Sequential(
            nn.Conv2d(n_features, n_classes, kernel_size=1),
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

    def forward(self, eeg_epoch):
        x = self.feature_pre(eeg_epoch) # (bs, filters, n_timepoints, n_channels)
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_channels, n_timepoints)
        raw_shape = x.shape
        x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, n_patches, n_timepoints)
        x = self.pos_drop(x + self.pos_embedding)
        x = self.transformer(x)
        x = x.view(raw_shape) # (bs, filters, n_channels, n_timepoints)
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_timepoints, n_channels)
        x = self.feature_post(x)
        x = self.classifier(x)
        out = x[:, :, 0, 0]
        return out
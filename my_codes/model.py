from turtle import forward
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# others
import math
# self-defined
import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log
from utils import _get_activation_fn
from attention import *
from conv_model import CSPFeedForward

# positionwise feedforward
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1, activation='gelu'):
        super(PositionwiseFeedForward, self).__init__()
        d_ff = 4 * d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

# positional embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        dive_term = (torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).exp()
        
        pe[:, 0::2] = torch.sin(position * dive_term)
        pe[:, 1::2] = torch.cos(position * dive_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.pe#[:, :x.size(1)]
        
# Transformer block   
class TransformerEncoderLayer(nn.Module):
    def __init__(self, max_len, d_model, n_head, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, normalize_before=True, conv=False,
                 n_filters_time=4, filter_size_time=25, 
                 n_filters_spatial=4, filter_size_spatial=22):
        super(TransformerEncoderLayer, self).__init__()
        
        if conv:
            self.self_attn = ConvMultiHeadAttention(max_len, d_model, n_head, dropout=dropout)
            self.feed_forward = CSPFeedForward(n_filters_time, filter_size_time, 
                                               n_filters_spatial, filter_size_spatial, 
                                               dropout=dropout)
        else:
            self.self_attn = MultiHeadAttention(d_model, n_head, dropout=dropout)
            self.feed_forward = PositionwiseFeedForward(d_model=d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward_post(self, src, src_mask = None):
        src2 = self.self_attn(src, src, src, mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask = None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, mask=src_mask)[0]
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
    def __init__(self, max_len: int = 22, d_model: int = 1000, n_head: int = 8, num_layers: int = 1,
                 dropout: float = 0.1, activation: str = "relu", layer_norm_eps: float = 1e-5, conv: bool = False):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head

        encoder_layer = TransformerEncoderLayer(max_len=max_len, d_model=d_model, n_head=n_head, dropout=dropout,
                                                activation=activation, layer_norm_eps=layer_norm_eps, conv=conv)
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

# NaiveTransformer               
class NaiveTransformer(nn.Module):
    def __init__(self, max_len, d_model, n_classes, dropout = 0.1,
                 n_head = 8, num_layers = 2, conv=False):
        super().__init__()

        # temporal
        # max_len = n_channels
        # d_model = n_timepoints
  
        # positional encoding
        self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.pos_drop = nn.Dropout(0.1)
        # transformer
        self.transformer = Transformer(
            max_len=max_len,
            d_model = d_model,
            n_head = n_head,
            num_layers = num_layers,
            dropout=dropout,
            conv = conv
        )
        # classfier
        self.classifier = nn.Sequential(
            nn.Conv2d(1, n_classes, kernel_size=1),
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
        # eeg_epoch: (bs, 1, n_timepoints, n_channels)
        # temporal: max_len = n_channels, d_model = n_timepoints
        x = eeg_epoch.permute(0, 1, 3, 2) # (bs, 1, max_len, d_model)
        raw_shape = x.shape
        x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, max_len, d_model)
        pe = self.pos_embedding(x)
        x = self.pos_drop(x + pe)
        x = self.transformer(x) # (bs, max_len, d_model)
        x = x.view(raw_shape) # (bs, 1, max_len, d_model)
        x = x.permute(0, 1, 3, 2) # (bs, 1, d_model, max_len)
        x = self.classifier(x) # (bs, n_classes, d_model, max_len)
        out = x[:, :, 0, 0] # (bs, n_classes)
        return out

# EEGTransformer               
class EEGTransformer(nn.Module):
    def __init__(self, n_timepoints, n_channels, n_classes, dropout = 0.1,
                 n_filters_time = 16, filter_size_time = 25, n_filters_spatial = 1,
                 n_head = 8, num_layers = 2, conv = False):
        super().__init__()
        assert filter_size_time <= n_timepoints, "Temporal filter size error"
        if n_filters_spatial <= 0: n_filters_spatial = n_channels
        iner = 20
        d_model = n_timepoints // iner
        max_len = n_channels
        # temporal filtering
        self.feature_pre = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (filter_size_time, 1), padding='same', bias=False),
            nn.AvgPool2d(kernel_size=(20, 1), stride=(20, 1)),
            nn.BatchNorm2d(n_filters_time),
            # nn.Dropout(dropout),
        )

        # positional encoding
        n_patches = n_filters_time * n_channels
        # self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, d_model))
        self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=n_patches)
        self.pos_drop = nn.Dropout(0.1)
        # transformer
        self.transformer = Transformer(
            max_len = max_len, 
            d_model = d_model,
            n_head = n_head,
            num_layers = num_layers,
            dropout = dropout,
            conv = conv
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
        # x = eeg_epoch
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_channels, n_timepoints)
        raw_shape = x.shape
        x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, n_patches, n_timepoints)
        x = self.pos_drop(x + self.pos_embedding(x))
        x = self.transformer(x)
        x = x.view(raw_shape) # (bs, filters, n_channels, n_timepoints)
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_timepoints, n_channels)
        # x = self.feature_post(x)
        x = self.classifier(x)
        out = x[:, :, 0, 0]
        return out

from einops.layers.torch import Rearrange
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (16, 5), stride=(1, 5)),
            # Rearrange('b e (h) (w) -> b (h w) e'),
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

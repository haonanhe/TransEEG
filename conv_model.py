from turtle import forward
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# others
import math
# self-defined
from common.torchutils import Expression, DepthwiseConv2d, cov, safe_log
from model import *
from utils import _get_activation_fn
from common.torchutils import Expression, safe_log, square

class CSPFeedForward(nn.Module):
    def __init__(self, n_filters_time = 4, filter_size_time = 25, 
                 n_filters_spatial = 4, filter_size_spatial = 22,
                 dropout = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (filter_size_time, 1), padding='same', bias=False),
            nn.BatchNorm2d(n_filters_time),
            nn.Conv2d(
                n_filters_time, n_filters_time*n_filters_spatial, (1, filter_size_spatial), padding = 'same',
                groups=n_filters_time, bias=False
            ),
            nn.BatchNorm2d(n_filters_time*n_filters_spatial),
            Expression(square),
            Expression(safe_log),
            nn.Dropout(dropout)
        )
        n_filters_out = n_filters_time * n_filters_spatial
        self.classifier = nn.Sequential(
            nn.Conv2d(n_filters_out, 1, kernel_size=1),
        )

        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: torch.Size([16, 22, 1000])
        x = x.unsqueeze(1).permute(0, 1, 3, 2) # (16, 1, 22, 1000)
        x = self.features(x)
        x = self.classifier(x) #(16, 1, 1000, 22)
        x = x.squeeze(1).transpose(1, 2) #(16, 22, 1000)
        return x

# conv positionwise feedforward
class ConvPositionwiseFeedForward(nn.Module):
    def __init__(self, d_ff, dropout=0.1, activation='gelu'):
        super(ConvPositionwiseFeedForward, self).__init__()
        self.conv_1 = nn.Conv2d(1, d_ff, kernel_size=1)
        self.conv_2 = nn.Conv2d(d_ff, 1, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_2(self.dropout(self.activation(self.conv_1(x))))
        return x.squeeze(1)
        
# ConvNaiveTransformer               
class ConvNaiveTransformer(nn.Module):
    def __init__(self, n_timepoints, n_channels, n_classes, dropout = 0.1,
                 n_filters_time = 1, filter_size_time = 25, n_filters_spatial = 1,
                 n_head = 8, num_layers = 2, conv = True):
        super().__init__()

        # # spatial
        # d_model = n_channels
        # max_len = n_timepoints

        # temporal
        d_model = n_timepoints
        max_len = n_channels

        # positional encoding
        self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.pos_drop = nn.Dropout(dropout)
        # transformer
        self.transformer = Transformer(
            max_len = max_len,
            d_model = d_model,
            n_head = n_head,
            num_layers = num_layers,
            dropout = dropout,
            conv = conv
        )
        n_features = n_filters_time #* n_filters_spatial
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
        # (bs, filters, n_timepoints, n_channels)
        x = eeg_epoch.permute(0, 1, 3, 2) # (bs, filters, n_channels, n_timepoints)
        # x = eeg_epoch
        raw_shape = x.shape
        x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, n_patches, n_timepoints)
        x = self.pos_drop(x + self.pos_embedding(x))
        x = self.transformer(x)
        x = x.view(raw_shape) # (bs, filters, n_channels, n_timepoints)
        x = x.permute(0, 1, 3, 2) # (bs, filters, n_timepoints, n_channels)
        x = self.classifier(x)
        out = x[:, :, 0, 0]
        return out

# # ConvEEGTransformer               
# class ConvEEGTransformer(nn.Module):
#     def __init__(self, n_timepoints, n_channels, n_classes, dropout = 0.1,
#                  n_filters_time = 16, filter_size_time = 25, n_filters_spatial = 1,
#                  n_head = 8, num_layers = 2, ffn_dim = 512,):
#         super().__init__()
#         assert filter_size_time <= n_timepoints, "Temporal filter size error"
#         if n_filters_spatial <= 0: n_filters_spatial = n_channels
#         d_model = n_timepoints
#         # temporal filtering
#         self.feature_pre = nn.Sequential(
#             nn.Conv2d(1, n_filters_time, (filter_size_time, 1), padding='same', bias=False),
#             nn.BatchNorm2d(n_filters_time),
#             # nn.Dropout(dropout),
#         )

#         # positional encoding
#         n_patches = n_filters_time * n_channels
#         self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=n_patches)
#         self.pos_drop = nn.Dropout(0.1)
#         # transformer
#         self.transformer = ConvTransformer(
#             d_model = d_model,
#             n_head = n_head,
#             num_layers = num_layers,
#             d_ff = ffn_dim,
#             filter_size_time=filter_size_time,
#             n_filters = n_filters_time
#         )
#         # spatial filtering & calculate covariance
#         self.feature_post = nn.Sequential(
#             DepthwiseConv2d(n_filters_time, n_filters_spatial, (1, n_channels), bias=False),
#             nn.BatchNorm2d(n_filters_time*n_filters_spatial),
#             Expression(cov),
#             Expression(safe_log),
#             nn.Dropout(dropout),
#         )
#         n_features = n_filters_time * n_filters_spatial
#         # classfier
#         self.classifier = nn.Sequential(
#             nn.Conv2d(n_features, n_classes, kernel_size=1),
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

#     def forward(self, eeg_epoch):
#         x = self.feature_pre(eeg_epoch) # (bs, filters, n_timepoints, n_channels)
#         x = x.permute(0, 1, 3, 2) # (bs, filters, n_channels, n_timepoints)
#         raw_shape = x.shape
#         x = x.reshape(x.size(0), -1, x.size(-1)) # (bs, n_patches, n_timepoints)
#         x = self.pos_drop(x + self.pos_embedding(x))
#         x = self.transformer(x)
#         x = x.view(raw_shape) # (bs, filters, n_channels, n_timepoints)
#         x = x.permute(0, 1, 3, 2) # (bs, filters, n_timepoints, n_channels)
#         x = self.feature_post(x)
#         x = self.classifier(x)
#         out = x[:, :, 0, 0]
#         return out
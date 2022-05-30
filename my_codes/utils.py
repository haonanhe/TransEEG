import os
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

def plot_figs(records, type, figpath, subject):
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.plot(list(range(0, len(records.index))), records[type])
    plt.xlabel('Epochs')
    if type[-4:] == 'loss':
        plt.ylabel('Loss')
        plt.savefig(figpath + subject + type + '.png', dpi=500, bbox_inches = 'tight')
    else:
        plt.ylabel('Accuracy [%]')
        plt.savefig(figpath + subject + type + '.png', dpi=500, bbox_inches = 'tight')
    plt.show()
    plt.close()
    print('Done')

def plot_figs_both(records, figpath, subject):
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    plt.plot(list(range(0, len(records.index))), records['train_loss'])
    plt.plot(list(range(0, len(records.index))), records['valid_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'val loss'], loc='upper right')
    plt.savefig(figpath + subject + 'loss.png', dpi=500, bbox_inches = 'tight')
    plt.close()
    
    plt.plot(list(range(0, len(records.index))), records['train_accu'])
    plt.plot(list(range(0, len(records.index))), records['valid_accu'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper right')
    plt.savefig(figpath + subject + 'accuracy.png', dpi=500, bbox_inches = 'tight')
    plt.close()
    
    print('Done')

def plot_bar(test_acc, subjects, figpath):
    x = np.arange(len(test_acc))
    plt.bar(x, test_acc)
    plt.title('Averaged accuracy for all subjects')
    plt.xticks(x, subjects)
    plt.ylabel('Accuracy [%]')
    plt.grid(which='both', axis='both')
    plt.show()
    plt.savefig(figpath + 'bars.png', dpi=500, bbox_inches = 'tight')
    plt.close()
    print('Done')

# choose activation functions
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.") 

# the way downsampling in swin transformer
class PatchMerging(nn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.resolution = resolution
        self.reduction = nn.Linear(resolution*dim, resolution*dim//2, bias=False)
        self.norm = nn.LayerNorm(resolution*dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        tmp = x[:, :, 0::self.resolution, :]
        for i in range(1, self.resolution):
            tmp = torch.cat([tmp, x[:, :, i::self.resolution, :]], 1)
        
        tmp = rearrange(tmp, 'b c h w -> b h w c')
        tmp = self.norm(tmp)
        tmp = self.reduction(tmp)
        x = self.dropout(x)
        tmp = rearrange(tmp, 'b h w c -> b c h w')

        return tmp

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

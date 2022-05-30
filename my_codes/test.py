import torch.nn as nn
import torch
from sklearn.preprocessing import robust_scale

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.transforms import *
    
import numpy as np
np.random.seed(0)
import seaborn as sns
import matplotlib.pyplot as plt

# data = np.load('/home/scutbci/public/hhn/Trans_EEG/training_results/attn.npz')
# # print(data)
# attn, label = data['arr_0'], data['arr_1']
# print(attn.shape)

datapath = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/'
subject = 'A09'
print('Load raw features for subject ' + subject + '...')
data = np.load(datapath+'rawfeatures/'+subject+'.npz')
dataTrain, labelTrain = data['dataTrain'], data['labelTrain']
dataTest, labelTest = data['dataTest'], data['labelTest']

data = torch.from_numpy(dataTrain)
data = data.unsqueeze(1)
from codes.my_codes.models import LiteViT
model = LiteViT(1, 16, 4, n_head=8, n_layers=1, dropout=0.5)
model.load_state_dict(torch.load('/home/scutbci/public/hhn/Trans_EEG/training_results/model_params/LiteViT/A09.pth'))
# model = torch.load('/home/scutbci/public/hhn/Trans_EEG/training_results/model_params/LiteViT/A09.pth')
model.eval()
y, attn = model(data.float())

sns.set_theme()
attn = attn.cpu().detach().numpy()
print(attn.shape)
x = attn[-2]
print(x.shape)
x = np.mean(x, 0)
label = labelTrain[-2]
print(x.shape)
print(label)
f = plt.figure(figsize=(8, 8))
ax = sns.heatmap(x, vmax=0.01, vmin=0, square=True, robust=True, cbar=True)
# plt.xticks([])
# plt.yticks([])
plt.savefig('/home/scutbci/public/hhn/Trans_EEG/training_results/figs/attn.png', dpi=500, bbox_inches = 'tight')

# from sklearn.manifold import TSNE
# import time
# from visualization import fashion_scatter
# # data = np.concatenate((all_data[:500], aug_data[:1000]), 0)
# # label1 = np.zeros((500, 1))
# # label2 = np.ones((1000, 1))
# # labels = np.concatenate((label1, label2), 0)
# # temp = rearrange(data, 'b h w -> b (h w)') 

# fashion_tsne = TSNE(random_state=3).fit_transform(dataTrain)
# fashion_scatter(fashion_tsne, labelTrain)
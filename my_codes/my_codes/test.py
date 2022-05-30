import numpy as np

# a = [0] * 4
# b = [[0] * 4]
# c = [0 for _ in range(4)]
# print(a)
# print(b)
# print(c)

import torch

# images = torch.ones(1, 1, 3, 3)
# print(images)
# n, c, w, h = images.shape
# padding = 1
# images = images.clone()
# images = torch.cat((torch.zeros(n, c, padding, h), images), 2)
# print(images)
# images = torch.cat((images, torch.zeros(n, c, padding, h)), 2)
# print(images)
# images = torch.cat((torch.zeros(n, c, w+2*padding, padding), images), 3)
# print(images)
# imasges = torch.cat((images, torch.zeros(n, c, w+2*padding, padding)), 3)
# print(images)

# a = [0, 1, 2]
# b = [[1, 2, 0], [3, 4, 5]]
# c = [set(_) for _ in b]
# print(set(a))
# # print()
# print(set(a) in [set(_) for _ in b])

# x = torch.ones(10, 3, 22, 100)
# # pooling = torch.nn.MaxPool2d((1,2))
# # x = pooling(x)
# conv = torch.nn.Conv2d(3, 3, (3, 3), padding='same', groups=3)
# x = conv(x)
# print(x.size())

import pandas as pd

# df = pd.DataFrame(
#     [
#         [24.3, 75.7, "high"],
#         [31, 87.8, "high"],
#         [22, 71.6, "medium"],
#         [35, 95, "medium"],
#     ],
#     columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
#     index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
# )

# import matplotlib.pyplot as plt
# x = list(range(0, len(df.index)))
# print(x)
# plt.plot(x, x)

# df1 = pd.DataFrame(np.random.randn(500, 4))
# print(len(df1.index))
# data = np.load('/home/scutbci/public/hhn/Trans_EEG/training_results/logs/EEGTransformer.npy' ,  allow_pickle=True)
# print(data)
 
# a = {}
# b = {}
# a['a0'] = df
# b['a1'] = df

# np.save('my_codes/file.npy', a)
# new_dict = np.load('my_codes/file.npy', allow_pickle='TRUE')
# print(new_dict)

# np.save('my_codes/file.npy', b)
# new_dict = np.load('my_codes/file.npy', allow_pickle='TRUE')
# print(new_dict)

# # print(df['temp_celsius'].to_list())
# print(df.index)

# a = 'val_loss'
# print(a[-4:])

# test1 = [0.8, 0.9, 0.6]
# test2 = [0.7, 0.3, 0.5]

# with open('D:\Trans_EEG\codes\my_codes\\accs.txt', 'a+') as f:
#     f.write('test1: ')
#     for i in test1:
#         f.write(str(i) + ' ')
#     f.write('\n')

# with open('D:\Trans_EEG\codes\my_codes\\accs.txt', 'a+') as f:
#     f.write('test2: ')
#     for i in test2:
#         f.write(str(i) + ' ')
#     f.write('\n')

# from model import NaiveTransformer
# model = NaiveTransformer(n_timepoints=1000, n_channels=22, n_classes=4, n_head=8, num_layers=2)
# x = torch.ones(16, 1, 1000, 22)
# y = model(x)

from data_processing import *
from common.transforms import *
datapath = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/'
subject = 'A01'
tf_tensor = ToTensor()
trainset, validset, testset = load_dataset(datapath, subject, tf_tensor)
print(trainset.type)

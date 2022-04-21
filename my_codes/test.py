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

a = [0, 1, 2]
b = [[1, 2, 0], [3, 4, 5]]
c = [set(_) for _ in b]
print(set(a))
# print()
print(set(a) in [set(_) for _ in b])
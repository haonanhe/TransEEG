import torch
import torch.nn as nn

def Conv2d(images, in_channels, out_channels, kernel_size, stride, padding, weights=None, bias=None):
    # weight & bias
    if weights is None:
        weights = torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1])
    if bias is None:
        bias = torch.zeros(out_channels)
    n, c, w, h = images.shape
    # padding
    images = images.clone()
    images = torch.cat((torch.zeros(n, c, padding, h), images), 2)
    images = torch.cat((images, torch.zeros(n, c, padding, h)), 2)
    images = torch.cat((torch.zeros(n, c, w+2*padding, padding), images), 3)
    images = torch.cat((images, torch.zeros(n, c, w+2*padding, padding)), 3)
    n, c, w, h = images.shape
    output = []
    for i, img in enumerate(images):
        img_out = []
        for j in range(out_channels):
            feature_map = []
            row = 0
            while row + kernel_size[0] <= h:
                row_feature_map = []
                column = 0
                while column + kernel_size[1] <= w:
                    channels = [0 for _ in range(c)]
                    for k in range(c):
                        for y in range(kernel_size[0]):
                            for x in range(kernel_size[1]):
                                channels[k] += img[k][row+y][column+x] * weights[j][k][y][x]
                    point = sum(channels) + bias[j]
                    row_feature_map.append(point)
                    column += stride[1]
                feature_map.append(row_feature_map)
                row += stride[0]
            img_out.append(feature_map)
        output.append(img_out)
    return torch.Tensor(output)            
                

        
import numpy as np
import random


def data_augmentation(dataTrain, labelTrain, segment=4, dilation=4, n_classes=4): 
    random.seed(3)
    
    dataTrain_new, labelTrain_new = [[] for _ in range(n_classes)], [[] for _ in range(n_classes)]
    for label in range(n_classes):
        index = [i for (i, v) in list(enumerate(labelTrain)) if v == (label+1)]
        for idx in index:
            dataTrain_new[label].append(dataTrain[idx])
            labelTrain_new[label].append(label + 1)

    dataTrain_seg = [[[] for x in range(segment)] for _ in range(n_classes)]
    length = dataTrain[0].shape[0] // segment
    for label in range(n_classes):
        for data in dataTrain_new[label]:
            for i in range(segment):
                dataTrain_seg[label][i].append(data[i*length:(i+1)*length, :])
    
    dataTrain_aug, labelTrain_aug, tmp = [], [], []
    for label in range(n_classes):
        for i in range(len(dataTrain_new[label])*dilation):
            for j in range(segment):
                tmp.append(random.choice(dataTrain_seg[label][j])) 
            dataTrain_aug.append(np.vstack(tmp))
            labelTrain_aug.append(label+1)
            tmp.clear()
    
    dataTrain_aug = np.concatenate((np.array(dataTrain_aug), dataTrain), 0)
    labelTrain_aug = np.concatenate((np.array(labelTrain_aug), labelTrain), 0)

    return dataTrain_aug, labelTrain_aug


def add_Gauss(x, mu=0, sigma=1e-2):
    gaussian_noise_matrix = np.random.normal(loc=mu, scale=sigma, size=x.shape)
    x = x + gaussian_noise_matrix

    return x

def cropping(x, y, seg_len, dilation, fs):
    num = x.shape[0]

    x_aug_list = []
    y_aug_list = []
    x_test_list = []
    y_test_list = []
    tmp = []

    for i in range(num):
        for j in range(dilation):
            r = random.randint(0, (4-seg_len)*fs)
            tmp.append(x[i][r : r+fs*seg_len, :])
            x_aug_list.append(x[i][r : r+fs*seg_len, :])
            y_aug_list.append(y[i])
        x_test_list.append(np.mean(tmp, 0))
        tmp = []
        y_test_list.append(y[i])
    
    x_aug_list = np.array(x_aug_list)
    y_aug_list = np.array(y_aug_list)
    x_test_list = np.array(x_test_list)
    y_test_list = np.array(y_test_list)
    
    return x_aug_list, y_aug_list, x_test_list, y_test_list



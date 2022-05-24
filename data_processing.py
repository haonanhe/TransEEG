# -*- coding: utf-8 -*-

import os
from einops import rearrange
import torch
from torch.utils.data import Dataset
import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.datawrapper import read_matdata, read_gdfdata
from common.signalproc import *


class EEGDataset(Dataset):
    def __init__(self, epochs, labels, transforms=None):
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.labels = torch.Tensor(labels - 1).long() # label {1, 2} to {0, 1}

    def __getitem__(self, idx):
        return self.epochs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

_available_dataset = [
    'bcicomp2008IIa',     # 4 class (L, R, F, T)
    ]

def load_eegdata(setname, datapath, subject):
    assert setname in _available_dataset, 'Unknown dataset name ' + setname
    if setname in ['bcicomp2008IIa', 'bcicomp2008IIa_2c']:
        fdatatrain = datapath+subject+'T.gdf'
        flabeltrain = datapath+'true_labels/'+subject+'T.mat'
        fdatatest = datapath+subject+'E.gdf'
        flabeltest = datapath+'true_labels/'+subject+'E.mat'
        dataTrain, targetTrain, dataTest, targetTest = \
            load_eegdata_bcicomp2008IIa(fdatatrain, flabeltrain, fdatatest, flabeltest)
    return dataTrain, targetTrain, dataTest, targetTest

def load_eegdata_bcicomp2008IIa(fdatatrain, flabeltrain, fdatatest, flabeltest):
    startcode = 768 # trial start event code
    s, events, clabs = read_gdfdata(fdatatrain)
    s = 1e6 * s # convert to millvolt for numerical stability
    pos = events['pos']
    code = events['type']
    indices = np.argwhere(code == startcode) # 768

    num_train = len(indices)
    fs = 250
    timewin = [-0.5, 4.0] # 4.5s data including 0.5s pre-task
    sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
    num_samples = sampleseg[1] - sampleseg[0]
    num_channels = s.shape[1]
    dataTrain = np.zeros([num_train, num_samples, num_channels])
    for i in range(num_train):
        begin = pos[indices[i, 0]] + sampleseg[0] + 2*fs # 2 seconds prepare
        end = begin + num_samples
        dataTrain[i,:,:] = s[begin:end,:]
    labeldata = read_matdata(flabeltrain, ['classlabel'])
    targetTrain = np.array(np.squeeze(labeldata['classlabel']), dtype=np.int)

    s, events, clabs = read_gdfdata(fdatatest)
    s = 1e6 * s # convert to millvolt for numerical stability
    pos = events['pos']
    code = events['type']
    indices = np.argwhere(code == startcode) # 768

    num_test = len(indices)
    dataTest = np.zeros([num_test, num_samples, num_channels])
    for i in range(num_test):
        begin = pos[indices[i, 0]] + sampleseg[0] + 2*fs # 2 seconds prepare
        end = begin + num_samples
        dataTest[i,:,:] = s[begin:end,:]
    labeldata = read_matdata(flabeltest, ['classlabel'])
    targetTest = np.array(np.squeeze(labeldata['classlabel']), dtype=np.int)
    
    return dataTrain, targetTrain, dataTest, targetTest

def extract_rawfeature(data, target, sampleseg, chanset, filter=None, standardize=False):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_samples_used = sample_end - sample_begin
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    features = np.zeros([num_trials, num_samples_used, num_channel_used])
    for i in range(num_trials):
        signal_filtered = data[i]
        if filter is not None:
            signal_filtered = signal.lfilter(filter[0], filter[1], signal_filtered, axis=0)
            # signal_filtered = signal.filtfilt(filter[0], filter[1], signal_epoch, axis=0)
        if standardize:
            # init_block_size = 1000 this param setting has a big impact on the result
            signal_filtered = exponential_running_standardize(signal_filtered, init_block_size=1000)
        features[i] = signal_filtered[sample_begin:sample_end, chanset]

    return features, labels

def extract_variance(data, target, sampleseg, chanset, filter=None):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_channel_used = len(chanset)

    # show_filtering_result(filter[0], filter[1], data[0,:,0])

    labels = target
    Rs = np.zeros([num_trials, num_channel_used, num_channel_used])
    for i in range(num_trials):
        signal_filtered = data[i]
        if filter is not None:
            signal_filtered = signal.lfilter(filter[0], filter[1], signal_filtered, axis=0)
            # signal_filtered = signal.filtfilt(filter[0], filter[1], signal_epoch, axis=0)
        signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
        cov_tmp = np.dot(signal_filtered.T, signal_filtered)
        Rs[i] = cov_tmp/np.trace(cov_tmp)

    return Rs, labels

def extract_variance_multiband(data, target, bands, sampleseg, chanset):
    num_trials, num_samples, num_channels = data.shape
    sample_begin = sampleseg[0]
    sample_end = sampleseg[1]
    num_channel_used = len(chanset)

    fs = 100
    order = 3
    num_bands = len(bands)
    Rss = []
    for k in range(num_bands):
        f1, f2 = bands[k]
        fb, fa = signal.butter(order, [2*f1/fs, 2*f2/fs], btype='bandpass')
        # fb, fa = signal.cheby2(order, 40, [2*f1/fs, 2*f2/fs], btype='bandpass')
        Rs = np.zeros([num_trials, num_channel_used, num_channel_used])
        for i in range(num_trials):
            signal_filtered = data[i]
            signal_filtered = signal.lfilter(filter[0], filter[1], signal_filtered, axis=0)
            # signal_filtered = signal.filtfilt(filter[0], filter[1], signal_epoch, axis=0)
            signal_filtered = signal_filtered[sample_begin:sample_end, chanset]
            cov_tmp = np.dot(signal_filtered.T, signal_filtered)
            Rs[i] = cov_tmp/np.trace(cov_tmp)
        Rss.append(Rs)

    labels = target

    return Rss, labels

def load_dataset_preprocessed(datapath, subject):
    f = np.load(datapath+'processed/'+subject+'.npz')
    return f['dataTrain'], f['targetTrain'], f['dataTest'], f['targetTest']

import os
from torch.utils.data.dataset import Subset
from EEG_Transformer.common_spatial_pattern import csp
from data_augmentation import data_augmentation, add_Gauss, cropping
import matplotlib.pyplot as plt
import scipy
def load_dataset(datapath, subject, tf_tensor, aug=True, dilation=4):
    ## data processing
    # print('Load EEG epochs for subject ' + subject)
    # dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    # print('Extract raw features from epochs for subject ' + subject)
    # featTrain, labelTrain = extract_rawfeature(dataTrain, targetTrain, sampleseg, chanset, [fb, fa])
    # featTest, labelTest = extract_rawfeature(dataTest, targetTest, sampleseg, chanset, [fb, fa])
    # if not os.path.isdir(datapath + 'rawfeatures\\'):
    #     os.mkdir(datapath + 'rawfeatures\\')
    # np.savez(datapath+'rawfeatures\\'+subject+'.npz',
    #              dataTrain=featTrain, labelTrain=labelTrain, dataTest=featTest, labelTest=labelTest)
    
    ### load data
    print('Load raw features for subject ' + subject + '...')
    data = np.load(datapath+'rawfeatures/'+subject+'.npz')
    dataTrain, labelTrain = data['dataTrain'], data['labelTrain']
    dataTest, labelTest = data['dataTest'], data['labelTest']

    all_data = np.concatenate((dataTrain, dataTest), 0)
    all_label = np.concatenate((labelTrain, labelTest), 0)
    all_shuff_num = np.random.permutation(len(all_data))
    all_data = all_data[all_shuff_num]
    all_label = all_label[all_shuff_num]

    # dataTrain, labelTrain, dataTest, labelTest = cropping(all_data, all_label, seg_len=3, dilation=10, fs=250)
    
    dataTrain = all_data[:516]
    labelTrain = all_label[:516]
    dataTest = all_data[516:]
    labelTest = all_label[516:]

    # standardize
    target_mean = np.mean(dataTrain)
    target_std = np.std(dataTrain)
    dataTrain = (dataTrain - target_mean) / target_std
    dataTest = (dataTest - target_mean) / target_std
    
    # data = dataTrain[100]
    # x = np.arange(1000)
    # for i in range(22):
    #     y = data[:, i]
    #     plt.plot(x, y)
    # plt.savefig('/home/scutbci/public/hhn/Trans_EEG/training_results/figs/eeg2.png')

    if aug:
        # dataTrain, labelTrain = data_augmentation(dataTrain, labelTrain, dilation=dilation)
        # target_mean = np.mean(dataTrain)
        # target_std = np.std(dataTrain)
        # dataTrain = (dataTrain - target_mean) / target_std
        dataTrain = add_Gauss(dataTrain, mu=0, sigma=1e-2)

    # dataTrain = dataTrain.transpose((0, 2, 1)) 
    # dataTest = dataTest.transpose((0, 2, 1)) 

    trainset_full = EEGDataset(dataTrain, labelTrain, tf_tensor)
    testset = EEGDataset(dataTest, labelTest, tf_tensor)
    ### dataset split
    valid_set_fraction = 0.2
    valid_set_size = int(len(trainset_full) * valid_set_fraction)
    train_set_size = len(trainset_full) - valid_set_size
    # trainset, validset = random_split(trainset_full, [train_set_size, valid_set_size])
    trainset = Subset(trainset_full, list(range(0, train_set_size)))
    validset = Subset(trainset_full, list(range(train_set_size, len(trainset_full))))


    # ###################################
    # root = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/processed_mat/'

    # # to get the data of target subject
    # total_data = scipy.io.loadmat(root + subject + 'T.mat')
    # train_data = total_data['data']
    # train_label = total_data['label']

    # train_data = np.transpose(train_data, (2, 1, 0))
    # train_data = np.expand_dims(train_data, axis=1)
    # train_label = np.transpose(train_label)

    # allData = train_data
    # allLabel = train_label[0]

    # # test data
    # # to get the data of target subject
    # test_tmp = scipy.io.loadmat(root + subject + 'E.mat')
    # test_data = test_tmp['data']
    # test_label = test_tmp['label']

    # train_data = train_data[250:1000, :, :]
    # test_data = np.transpose(test_data, (2, 1, 0))
    # test_data = np.expand_dims(test_data, axis=1)
    # test_label = np.transpose(test_label)

    # testData = test_data
    # testLabel = test_label[0]

    # all_data = np.concatenate((allData, testData), 0)
    # all_label = np.concatenate((allLabel, testLabel), 0)
    # all_shuff_num = np.random.permutation(len(all_data))
    # all_data = all_data[all_shuff_num]
    # all_label = all_label[all_shuff_num]

    # allData = all_data[:516]
    # allLabel = all_label[:516]
    # testData = all_data[516:]
    # testLabel = all_label[516:]

    # # standardize
    # target_mean = np.mean(allData)
    # target_std = np.std(allData)
    # allData = (allData - target_mean) / target_std
    # testData = (testData - target_mean) / target_std

    # # tmp_alldata = np.transpose(np.squeeze(allData), (0, 2, 1))
    # # Wb = csp(tmp_alldata, allLabel-1)  # common spatial pattern
    # # allData = np.einsum('abcd, ce -> abed', allData, Wb)
    # # testData = np.einsum('abcd, ce -> abed', testData, Wb)

    # allData = allData.squeeze(1)
    # testData = testData.squeeze(1)

    # allData = rearrange(allData, 'b h w -> b w h')
    # testData = rearrange(testData, 'b h w -> b w h')

    # all_shuff_num = np.random.permutation(len(allData))
    # allData = allData[all_shuff_num]
    # allLabel = allLabel[all_shuff_num]
    # ### dataset split
    # valid_set_fraction = 0.2
    # valid_set_size = int(allData.shape[0] * valid_set_fraction)
    # valData = allData[:valid_set_size]
    # valLabel = allLabel[:valid_set_size]
    # trainData = allData[valid_set_size:]
    # trainLabel = allLabel[valid_set_size:]
    # # augmentation
    # if aug:
    #     trainData, trainLabel = data_augmentation(trainData, trainLabel, dilation=dilation)
    # # trainset, validset = random_split(trainset_full, [train_set_size, valid_set_size])
    # trainset_full = EEGDataset(allData, allLabel, tf_tensor)
    # trainset = EEGDataset(trainData, trainLabel, tf_tensor)
    # validset = EEGDataset(valData, valLabel, tf_tensor)
    # testset = EEGDataset(testData, testLabel, tf_tensor)

    # # trainset = Subset(trainset_full, list(range(0, train_set_size)))
    # # validset = Subset(trainset_full, list(range(train_set_size, len(trainset_full))))

    # # return self.allData, self.allLabel, self.testData, self.testLabel

    return trainset, validset, testset, trainset_full


def load_dataset_full(datapath, subject, tf_tensor, aug=True, dilation=4):
    root = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/processed_mat/'

    subject = 'A01'
    total_data = scipy.io.loadmat(root + subject + 'T.mat')
    train_data = total_data['data']
    train_label = total_data['label']

    train_data = np.transpose(train_data, (2, 1, 0))
    train_data = np.expand_dims(train_data, axis=1)
    train_label = np.transpose(train_label)

    test_tmp = scipy.io.loadmat(root + subject + 'E.mat')
    test_data = test_tmp['data']
    test_label = test_tmp['label']

    test_data = np.transpose(test_data, (2, 1, 0))
    test_data = np.expand_dims(test_data, axis=1)
    test_label = np.transpose(test_label)

    allData = np.concatenate((train_data, test_data), 0)
    allLabel = np.concatenate((train_label[0], test_label[0]), 0)

    for i in range(1, 7):
        subject = 'A0' + str(i+1)

        # to get the data of target subject
        total_data = scipy.io.loadmat(root + subject + 'T.mat')
        train_data = total_data['data']
        train_label = total_data['label']

        train_data = np.transpose(train_data, (2, 1, 0))
        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        test_tmp = scipy.io.loadmat(root + subject + 'E.mat')
        test_data = test_tmp['data']
        test_label = test_tmp['label']

        test_data = np.transpose(test_data, (2, 1, 0))
        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)

        allData = np.concatenate((allData, train_data, test_data), 0) 
        allLabel = np.concatenate((allLabel, train_label[0], test_label[0]), 0)

    subject = 'A09'
    total_data = scipy.io.loadmat(root + subject + 'T.mat')
    train_data = total_data['data']
    train_label = total_data['label']

    train_data = np.transpose(train_data, (2, 1, 0))
    train_data = np.expand_dims(train_data, axis=1)
    train_label = np.transpose(train_label)

    test_tmp = scipy.io.loadmat(root + subject + 'E.mat')
    test_data = test_tmp['data']
    test_label = test_tmp['label']

    test_data = np.transpose(test_data, (2, 1, 0))
    test_data = np.expand_dims(test_data, axis=1)
    test_label = np.transpose(test_label)

    testData = np.concatenate((train_data, test_data), 0)
    testLabel = np.concatenate((train_label[0], test_label[0]), 0)

    # standardize
    target_mean = np.mean(allData)
    target_std = np.std(allData)
    allData = (allData - target_mean) / target_std
    testData = (testData - target_mean) / target_std

    # tmp_alldata = np.transpose(np.squeeze(allData), (0, 2, 1))
    # Wb = csp(tmp_alldata, allLabel-1)  # common spatial pattern
    # allData = np.einsum('abcd, ce -> abed', allData, Wb)
    # testData = np.einsum('abcd, ce -> abed', testData, Wb)
    allData = allData.squeeze(1)
    testData = testData.squeeze(1)

    allData = rearrange(allData, 'b h w -> b w h')
    testData = rearrange(testData, 'b h w -> b w h')

    all_shuff_num = np.random.permutation(len(allData))
    allData = allData[all_shuff_num]
    allLabel = allLabel[all_shuff_num]
    ### dataset split
    valid_set_fraction = 0.2
    valid_set_size = int(allData.shape[0] * valid_set_fraction)
    valData = allData[:valid_set_size]
    valLabel = allLabel[:valid_set_size]
    trainData = allData[valid_set_size:]
    trainLabel = allLabel[valid_set_size:]
    # augmentation
    if aug:
        trainData, trainLabel = data_augmentation(trainData, trainLabel, dilation=dilation)
    # trainset, validset = random_split(trainset_full, [train_set_size, valid_set_size])
    trainset_full = EEGDataset(allData, allLabel, tf_tensor)
    trainset = EEGDataset(trainData, trainLabel, tf_tensor)
    validset = EEGDataset(valData, valLabel, tf_tensor)
    testset = EEGDataset(testData, testLabel, tf_tensor)

    # trainset = Subset(trainset_full, list(range(0, train_set_size)))
    # validset = Subset(trainset_full, list(range(train_set_size, len(trainset_full))))

    # return self.allData, self.allLabel, self.testData, self.testLabel

    return trainset, validset, testset, trainset_full

# import sys
# sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
# from common.transforms import *
# from torch.utils.data import Dataset, DataLoader
# tf_tensor = ToTensor()
# trainset, validset, testset, trainset_full = load_dataset('datapath', 'A01', tf_tensor, aug=False, dilation=4)
# train_dataloader = DataLoader(trainset, batch_size=16, shuffle=True)
# for batch_x, batch_y in train_dataloader:
#     print(batch_x.size())

# if __name__ == '__main__':

#     # setname = 'bcicomp2005IVa'
#     # datapath = 'E:/bcicompetition/bci2005/IVa/'
#     # subjects = ['aa', 'al', 'av', 'aw', 'ay']
#     setname = 'bcicomp2008IIa'
#     datapath = 'D:\\Trans_EEG\\data\\BCIIV_2a\\'
#     subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

#     import os
#     if not os.path.isdir(datapath + 'processed/'):
#         os.mkdir(datapath + 'processed/')

#     for ss in range(len(subjects)):
#         subject = subjects[ss]

#         print('Load and extract continuous EEG into epochs for subject '+subject)
#         dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)

#         np.savez(datapath+'processed/'+subject+'.npz',
#                  dataTrain=dataTrain, targetTrain=targetTrain, dataTest=dataTest, targetTest=targetTest)



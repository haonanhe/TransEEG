# -*- coding: utf-8 -*-

import os
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
def load_dataset(datapath, subject, tf_tensor):
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

    # mix train and test data
    # all_data = np.concatenate((dataTrain, dataTest), 0)
    # all_label = np.concatenate((labelTrain, labelTest), 0)
    # dataTrain = all_data[:516]
    # labelTrain = all_label[:516]
    # dataTest = all_data[516:]
    # labelTest = all_label[516:]

    # # dataTrain:(288, 1000, 22)
    # Wb = csp(dataTrain, labelTrain-1)  # common spatial pattern
    # # Wb:(22, 16)
    # dataTrain = dataTrain.transpose((0, 2, 1)) 
    # dataTest = dataTest.transpose((0, 2, 1)) 
    # dataTrain = np.einsum('acd, ce -> aed', dataTrain, Wb) # dataTrain:(288, 16, 1000)
    # dataTest = np.einsum('acd, ce -> aed', dataTest, Wb)

    trainset_full = EEGDataset(dataTrain, labelTrain, tf_tensor)
    testset = EEGDataset(dataTest, labelTest, tf_tensor)
    ### dataset split
    valid_set_fraction = 0.2
    valid_set_size = int(len(trainset_full) * valid_set_fraction)
    train_set_size = len(trainset_full) - valid_set_size
    # trainset, validset = random_split(trainset_full, [train_set_size, valid_set_size])
    trainset = Subset(trainset_full, list(range(0, train_set_size)))
    validset = Subset(trainset_full, list(range(train_set_size, len(trainset_full))))

    return trainset, validset, testset, trainset_full
    

if __name__ == '__main__':

    # setname = 'bcicomp2005IVa'
    # datapath = 'E:/bcicompetition/bci2005/IVa/'
    # subjects = ['aa', 'al', 'av', 'aw', 'ay']
    setname = 'bcicomp2008IIa'
    datapath = 'D:\\Trans_EEG\\data\\BCIIV_2a\\'
    subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

    import os
    if not os.path.isdir(datapath + 'processed/'):
        os.mkdir(datapath + 'processed/')

    for ss in range(len(subjects)):
        subject = subjects[ss]

        print('Load and extract continuous EEG into epochs for subject '+subject)
        dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)

        np.savez(datapath+'processed/'+subject+'.npz',
                 dataTrain=dataTrain, targetTrain=targetTrain, dataTest=dataTest, targetTest=targetTest)



# -*- coding:utf-8 -*-

import numpy as np
from common.linear import *
from common.spatialfilter import *
from motorimagery.mireader import *


"""
setname = 'bcicomp2005IVa'
fs = 100
n_classes = 2
chanset = np.arange(118)
# chanset = [
#     np.arange(13,22),
#     np.arange(32,39),
#     np.arange(49,58),
#     np.arange(67,76),
#     np.arange(86,95),
#     np.array([103, 105, 107, 111, 112, 113])
#     ]
# chanset = np.hstack(chanset)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IVa/'
subjects = ['aa', 'al', 'av', 'aw', 'ay']
"""
"""
setname = 'bcicomp2005IIIa'
fs = 250
n_classes = 4
chanset = np.arange(60)
n_channels = len(chanset)
datapath = 'E:/bcicompetition/bci2005/IIIa/'
subjects = ['k3', 'k6', 'l1']
"""

setname = 'bcicomp2008IIa'
fs = 250
n_classes = 4
chanset = np.arange(22)
n_channels = len(chanset)
datapath = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/'
# datapath = '/Users/yuty2009/data/bcicompetition/bci2008/IIa/'
subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

order, fstart, fstop, fstep, ftrans = 4, 4, 40, 4, 2
f1s = np.arange(fstart, fstop, fstep)
f2s = np.arange(fstart+fstep, fstop+fstep, fstep)
fbanks = np.hstack((f1s[:,None], f2s[:,None]))
# fbanks = [[7, 30], [30, 40]]
n_fbanks = len(fbanks)

timewin = [1.0, 4.0] # 0.5s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

n_filters = 3

train_accus = np.zeros(len(subjects))
test_accus = np.zeros(len(subjects))
for ss in range(len(subjects)):
    subject = subjects[ss]
    print('Load EEG epochs for subject ' + subject)
    dataTrain, targetTrain, dataTest, targetTest = load_eegdata(setname, datapath, subject)
    print('Extract multi-band features from epochs for subject ' + subject)
    featTrain_bands = []
    featTest_bands = []
    for k in range(n_fbanks):
        f1, f2 = fbanks[k]
        fpass = [f1*2.0/fs, f2*2.0/fs]
        fstop =  [(f1-ftrans)*2.0/fs, (f2+ftrans)*2.0/fs]
        # fb, fa = signal.butter(order, fpass, btype='bandpass')
        fb, fa = signal.cheby2(order, 30, fstop, btype='bandpass')
        featTrain, labelTrain = extract_rawfeature(dataTrain, targetTrain, sampleseg, chanset, [fb, fa])
        featTest, labelTest = extract_rawfeature(dataTest, targetTest, sampleseg, chanset, [fb, fa])
        featTrain_bands.append(featTrain)
        featTest_bands.append(featTest)
    featTrain_bands = np.transpose(np.stack(featTrain_bands, axis=0), [1, 0, 2, 3])
    featTest_bands = np.transpose(np.stack(featTest_bands, axis=0), [1, 0, 2, 3])
    n_train = featTrain.shape[1]
    n_test = featTest.shape[1]

    fbcsp = FBCSP(n_filters)
    wfbcsp = fbcsp.fit(featTrain_bands, labelTrain)

    x_train = fbcsp.transform(featTrain_bands, wfbcsp)
    y_train = labelTrain - 1 # 

    model = SoftmaxClassifier()
    model.fit(x_train, y_train)

    y_pred_train, _ = model.predict(x_train)
    train_accus[ss] = np.mean(np.array(y_pred_train == y_train).astype(int))

    x_test = fbcsp.transform(featTest_bands, wfbcsp)
    y_test = labelTest - 1

    # y_pred = model.predict(x_test)
    y_pred, _ = model.predict(x_test)
    test_accus[ss] = np.mean(np.array(y_pred == y_test).astype(int))

    print(f'Subject {subject} train_accu: {train_accus[ss]: .3f}, test_accu: {test_accus[ss]: .3f}')

print(f'Overall accuracy: {np.mean(test_accus): .3f}')

import matplotlib.pyplot as plt
x = np.arange(len(test_accus))
plt.bar(x, test_accus*100)
plt.title('Averaged accuracy for all subjects')
plt.xticks(x, subjects)
plt.ylabel('Accuracy [%]')
plt.grid(which='both', axis='both')
plt.show()

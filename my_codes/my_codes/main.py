import time
import numpy as np
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset

import sys
sys.path.append("/home/scutbci/public/hhn/Trans_EEG/codes") 
from common.transforms import *
from common.torchutils import RememberBest
from common.stopcriteria import Or, MaxEpochs, NoIncrease, ColumnBelow
from common.torchutils import train_epoch, evaluate
# from motorimagery.mireader import *

from motorimagery.convnet import CSPNet, EEGNet, ShallowConvNet, DeepConvNet
from motorimagery.fbcnet import deepConvNet, eegNet
# from motorimagery.transformer import EEGTransformer
from model import EEGTransformer, NaiveTransformer
from conv_model import ConvNaiveTransformer
from data_processing import *
from utils import *

############ Settings ############
### Bcicomp2008IIa
setname = 'bcicomp2008IIa'
fs = 250
n_classes = 4
chanset = np.arange(22)
n_channels = len(chanset)
datapath = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/'
subjects = ['A01']#, 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']

### data processing 
order, f1, f2, ftrans = 4, 2.1, 48, 2
fpass = [f1*2.0/fs, f2*2.0/fs]
fstop =  [(f1-ftrans)*2.0/fs, (f2+ftrans)*2.0/fs]
# fb, fa = signal.butter(order, fpass, btype='bandpass')
fb, fa = signal.cheby2(order, 30, fstop, btype='bandpass')
# show_filter(fb, fa, fs)
timewin = [0.5, 4.5] # 0.5 s pre-task data
sampleseg = [int(fs*timewin[0]), int(fs*timewin[1])]
n_timepoints = sampleseg[1] - sampleseg[0]

### EEGDataset
tf_tensor = ToTensor()

### GPU or CPU
torch.manual_seed(7)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

### output path
outpath = '/home/scutbci/public/hhn/Trans_EEG/training_results/'
if not os.path.exists(outpath):
    os.makedirs(outpath)
    
### Training settings
n_epochs = 300#1500
n_epochs_full = 300#600
n_epochs_nochange = 50#200
batch_size = 16
monitor_items = [
    'train_accu', 'train_loss',
    'valid_accu', 'valid_loss',
    ]
train_after_earlystop = True

model_type = 'ConvNaiveTransformer'
def get_model(model_type):
    if model_type == 'CSPNet':
        model = CSPNet(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_type == 'EEGNet':
        model = EEGNet(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_type == 'ShallowConvNet':
        model = ShallowConvNet(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_type == 'DeepConvNet':
        model = DeepConvNet(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes).to(device)
    elif model_type == 'EEGTransformer':
        model = EEGTransformer(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes, nhead=8, num_layers=2).to(device)
    elif model_type == 'NaiveTransformer':
        model = NaiveTransformer(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes, n_head=8, num_layers=2).to(device)
    elif model_type == 'ConvNaiveTransformer':
        model = ConvNaiveTransformer(n_timepoints=n_timepoints, n_channels=n_channels, n_classes=n_classes, n_head=8, num_layers=2).to(device)
    return model

# paths
figpath = outpath + 'figs/' + model_type + '/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
modelpath = outpath + 'model_params/' + model_type + '/'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)
logpath = outpath + 'logs/'
if not os.path.exists(logpath):
    os.makedirs(logpath)
    
############ Train & Test ############
test_accus = np.zeros(len(subjects))
log_list = {}
for ss in range(len(subjects)):
    subject = subjects[ss]
    ### load data
    trainset, validset, testset = load_dataset(datapath, subject, tf_tensor)
    ### model/criterion/optimizer
    model = get_model(model_type)
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,  betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    # adjust lr
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    ### 
    epochs_df = pd.DataFrame(columns=monitor_items)
    remember_best = RememberBest('valid_accu', order=-1)
    stop_criterion = Or( # for early stop
        [
            MaxEpochs(n_epochs),
            NoIncrease("valid_accu", n_epochs_nochange),
        ]
    )
    ### Start training & val & test
    print('Start training for subject ' + subject + '...')
    epoch = 0
    stop = False
    early_stop = False
    while not stop:
    # for epoch in range(1, n_epochs+1):
        epoch = epoch + 1
        start_time = time.time()
        ## train & val
        train_accu, train_loss = train_epoch(
            model, trainset, criterion=criterion, optimizer=optimizer,
            batch_size=batch_size, device=device)
        valid_accu, valid_loss  = evaluate(model, validset, criterion=criterion, batch_size=batch_size, device=device)
        ## result logging
        epochs_df = epochs_df.append({
            'train_accu': train_accu, 'train_loss': train_loss,
            'valid_accu': valid_accu, 'valid_loss': valid_loss,
        }, ignore_index=True)
        remember_best.remember_epoch(epochs_df, model, optimizer)
        stop = stop_criterion.should_stop(epochs_df)
        # if stop:
        #     # first load the best model
        #     remember_best.reset_to_best_model(epochs_df, model, optimizer)
        #     # Now check if we should continue training:
        #     if train_after_earlystop:
        #         if not early_stop:
        #             stop = False
        #             early_stop = True
        #             print('Early stop reached now continuing with full trainset')
        #             epoch = 0
        #             # epochs_df.drop(epochs_df.index, inplace=True)
        #             trainset = trainset_full # Use the full train dataset
        #             remember_best = RememberBest('valid_loss', order=1)
        #             stop_criterion = Or( # for full trainset training
        #                 [
        #                     MaxEpochs(n_epochs_full),
        #                     ColumnBelow("valid_loss", train_loss),
        #                 ]
        #             )

        test_accu, test_loss  = evaluate(model, testset, criterion=criterion, batch_size=batch_size, device=device)
        
        # save model
        torch.save(model.state_dict(), modelpath + subject + '.pth')
        
        print((f"Epoch: {epoch}, "
               f"Train accu: {train_accu:.3f}, loss: {train_loss:.3f}, "
               f"Valid accu: {valid_accu:.3f}, loss: {valid_loss:.3f}, "
               f"Test accu:  {test_accu:.3f},  loss: {test_loss:.3f}, "
               f"Epoch time = {time.time() - start_time: .3f} s"))
    test_accus[ss] = test_accu
    log_list[subject] = epochs_df
    for item in monitor_items:
        plot_figs(epochs_df, item, figpath, subject)

with open(logpath + 'acc.txt', 'a+') as f:
    f.write(model_type)
    for i in test_accus:
        f.write(str(i) + ' ')
    f.write('\n')

print(f'Overall accuracy: {np.mean(test_accus): .3f}')

# save monitor items
np.save(logpath + model_type + '.npy', log_list)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # for 'OMP: Error #15:'
# plot bars
plot_bar(test_accus, subjects, figpath)



    
    
    
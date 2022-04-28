import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def plot_figs(records, type, figpath, subject):
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    print(len(records.index))
    print(list(range(0, len(records.index))))
    plt.plot(list(range(0, len(records.index))), records[type])
    plt.xlabel('Epochs')
    if type[-4:] == 'loss':
        plt.ylabel('Loss')
        plt.savefig(figpath + subject + type + '.png')
    else:
        plt.ylabel('Accuracy [%]')
        plt.savefig(figpath + subject + type + '.png')
    plt.show()
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
    plt.savefig(figpath + 'bars.png')
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

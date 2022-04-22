import os
import numpy as np
import matplotlib.pyplot as plt

def plot_figs(records, type, figpath, subject):
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    plt.plot(records.index, records[type])
    plt.xlabel('Epochs')
    if type[-4:] == 'loss':
        plt.ylabel('Loss')
        plt.savefig(figpath + subject + 'loss.png')
    else:
        plt.ylabel('Accuracy [%]')
        plt.savefig(figpath + subject + 'acc.png')
    plt.show()
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
    print('Done')
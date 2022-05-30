import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import seaborn as sns
from einops import rearrange

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

seed = 123

outpath = '/home/scutbci/public/hhn/Trans_EEG/training_results/'
figpath = outpath + 'figs/'

def fashion_scatter(x, colors, figpath):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)-1])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.savefig(figpath + 'tsne.png', dpi=500, bbox_inches = 'tight')
    # return f, ax, sc, txts

def plot_raw_eeg(data, figpath):
    # data.shape = [T, C]
    t, c = data.shape[0], data.shape[1]
    x = np.arange(t)
    for i in range(c):
        y = data[:, i]
        plt.plot(x, y)
    plt.savefig(figpath + 'eeg2.png')

def plot_tsne(data, label, figpath):
    # data: two dimension
    fashion_tsne = TSNE(random_state=3).fit_transform(data)
    fashion_scatter(fashion_tsne, label, figpath)

def plot_attn_heatmap(attn, figpath):
    # attn: (n, n)
    
    sns.set_theme()
    plt.figure(figsize=(8, 8))
    sns.heatmap(attn, vmax=0.01, vmin=0, square=True, robust=True, cbar=True)
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig(figpath+'attn.png', dpi=500, bbox_inches='tight')
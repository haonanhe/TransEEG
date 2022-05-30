# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal


def bandpower(x, fs, bands, method='fft'):
    """ Compute power in each frequency bin specified by bands from FFT result of
        x. By default, x is a real signal.
        Refer to https://github.com/forrestbao/pyeeg/blob/master/pyeeg/spectrum.py

        Parameters
        -----------
        x
            list
            a 1-D real time series.
        fs
            integer
            the sampling rate in physical frequency.
        bands
            list
            boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
            [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
            You can also use range() function of Python to generate equal bins and
            pass the generated list to this function.
            Each element of Band is a physical frequency and shall not exceed the
            Nyquist frequency, i.e., half of sampling frequency.

        method
            string
            power estimation method fft/burg/welch, default fft.

        Returns
        -------
        power
            list
            spectral power in each frequency bin.
        """

    L = len(x)
    if method == 'fft':
        pxx = np.fft.fft(x)
        pxx = abs(pxx)
        pxx = pxx[:L//2]
        f = np.arange(L / 2) / L
    elif method == 'welch':
        f, pxx = signal.welch(x, fs)
    elif method == 'periodogram':
        f, pxx = signal.periodogram(x, fs)
    else:
        assert 'unknown method'

    num_bands = len(bands) - 1
    pxx_bands = np.zeros(num_bands)
    for i in range(0, num_bands):
        f1 = float(bands[i]) / fs
        f2 = float(bands[i + 1]) / fs
        indices = np.argwhere((f >= f1) & (f < f2))
        pxx_bands[i] = sum(pxx[indices])
    return pxx_bands


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    datapath = '/home/scutbci/public/hhn/Trans_EEG/data/BCIIV2a/'
    subject = 'A09'
    print('Load raw features for subject ' + subject + '...')
    data = np.load(datapath+'rawfeatures/'+subject+'.npz')
    dataTrain, labelTrain = data['dataTrain'], data['labelTrain']
    dataTest, labelTest = data['dataTest'], data['labelTest']

    dataTrain = dataTrain[10:20, ]
    labelTrain = labelTrain[10:20, ]
    # # fs = 250
    # # n_classes = 4
    # # chanset = np.arange(22)
    # fs = 1000
    # T = 1 / fs
    # L = 1500
    # # L = 1000
    # t = np.arange(L) * T
    # f1 = np.arange(L / 2) / L
    # nfft = int(2 ** np.ceil(np.log2(L)))
    # s = 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    # x = s + 2 * np.random.randn(t.size)

    x = dataTrain[-2]#[:, 10]
    fs = 250
    L = 1000
    t = np.arange(L)
    c = 22
    print(x.shape)
    x = np.mean(x, 1)
    print(x.shape)
    print(labelTrain[-2])

    y = np.fft.fft(x)
    print(y.shape)
    pxx1 = abs(y)
    pxx1 = pxx1[:L//2]
    f2, pxx2 = signal.periodogram(x)
    pxx2[0] = np.median(pxx2[:3])
    f3, pxx3 = signal.welch(x)
    # print(f3.shape)
    # print(pxx3.shape)

    freqs, time, Zxx = signal.spectral.stft(x, fs=fs, 
                               nfft=fs//4, nperseg=fs//4, noverlap=fs//8,)
    
    plt.pcolormesh(time, freqs, np.abs(Zxx), #shading='gouraud' #开启这个参数就会图像插值，显示会更光滑
                    )

    plt.ylim(ymax=50)

    # plt.subplot(211)
    # plt.plot(t, x)
    # plt.title('Signal Corrupted with Zero-Mean Random Noise')
    # plt.xlabel('t (milliseconds)')
    # plt.ylabel('X(t)')

    # plt.subplot(212)
    # # plt.plot(f1 * fs, 10 * np.log10(pxx1))
    # plt.specgram(x, pxx3, 250)

    # # plt.subplot(412)
    # # plt.plot(f1 * fs, 10 * np.log10(pxx1))
    # # plt.title('Single-Sided Amplitude Spectrum of X(t)')
    # # plt.xlabel('f (Hz)')
    # # plt.ylabel('10*log10|P1(f)|')

    # # plt.subplot(413)
    # # plt.plot(f2, 10 * np.log10(pxx2))
    # # plt.xlabel('Normalized Frequency')
    # # plt.ylabel('Power (dB)')
    # # plt.title('Periodogram')

    # # plt.subplot(414)
    # # plt.plot(f3, 10 * np.log10(pxx3))
    # # plt.xlabel('Normalized Frequency')
    # # plt.ylabel('Power (dB)')
    # # plt.title('Welchs')

    plt.show()
    plt.savefig('/home/scutbci/public/hhn/Trans_EEG/training_results/figs/timefreq.png', dpi=500, bbox_inches = 'tight')
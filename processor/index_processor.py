"""
Our Index Filter calculate fft hr index
Author: Yunfei Tian
Date: 2025/03/26
"""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()  # bit_length返回表示二进制整数的位数。


def calculate_fft_hr_just(ppg_signal, fs=30, hr_pass=(0.75, 2.5), by_welch=True, nfft=None):
    global x_ppguu, y_ppguu
    ppg_signal = np.expand_dims(ppg_signal, 0)

    if nfft is None:
        nfft = _next_power_of_2(ppg_signal.shape[1])
    if nfft < 1:
        nfft = round(fs / nfft)

    if by_welch:
        n = ppg_signal.shape[1]
        if n < 256:
            seglength = n
            overlap = int(0.8 * n)  # fixed overlapping
        else:
            seglength = 256
            overlap = 200
        x_ppg, y_ppg = scipy.signal.welch(ppg_signal, nperseg=seglength, noverlap=overlap, fs=fs, nfft=nfft)

    else:
        x_ppg, y_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=nfft, detrend=False)

    hr_index = np.argwhere((x_ppg >= hr_pass[0]) & (x_ppg <= hr_pass[1]))  # index list
    mask_x_ppg = np.take(x_ppg, hr_index)
    mask_y_ppg = np.take(y_ppg, hr_index)
    index = np.argmax(mask_y_ppg, 0)
    fft_hr = np.take(mask_x_ppg, index)[0] * 60
    return fft_hr

def calculate_fft_hr_index(ppg_signal, index_filter1, fs=30, hr_pass=(0.75, 2.5), by_welch=True, nfft=None):
    ppg_signal = np.expand_dims(ppg_signal, 0)
    if nfft is None:
        nfft = _next_power_of_2(ppg_signal.shape[1])
    if nfft < 1:
        nfft = round(fs / nfft)

    if by_welch:
        n = ppg_signal.shape[1]
        if n < 256:
            seglength = n
            overlap = int(0.8 * n)  # fixed overlapping
        else:
            seglength = 256
            overlap = 200
        x_ppg, y_ppg = scipy.signal.welch(ppg_signal, nperseg=seglength, noverlap=overlap, fs=fs, nfft=nfft)

    else:
        x_ppg, y_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=nfft, detrend=False)

    hr_index = np.argwhere((x_ppg >= hr_pass[0]) & (x_ppg <= hr_pass[1]))  # index list
    mask_x_ppg = np.take(x_ppg, hr_index)
    mask_y_ppg = np.take(y_ppg, hr_index)
    index = np.argmax(mask_y_ppg, 0)

    index_use = index_filter1.filter(index)
    fft_hr = np.take(mask_x_ppg, index_use)[0] * 60
    return fft_hr



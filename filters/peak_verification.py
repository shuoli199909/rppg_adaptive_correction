"""
Peak Verification Filter
"""
# Author: Yunfei Tian
# Date: 2025/04/15

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter



def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()  # bit_length返回表示二进制整数的位数。


def compute_hr_3peak(mask_x_ppg, mask_y_ppg, prev_hr_3p):
    """
    Calculate the current heart rate:
    1. Find the top three points of power amplitude and their corresponding frequencies
    2. Calculate the heart rate candidate value (HR1, HR2, HR3)
    3. Calculate the weighted amplitude
    4. Select the final HR output

    :param mask_x_ppg: frequency array (numpy array, shape: [N, 1])
    :param mask_y_ppg: Power amplitude array (numpy array, shape: [N, 1])
    :param prev_hr_3p: The last calculated heart rate
    :return: Current final heart rate (HR)
    """

    # 1. Find the top three points of power
    top3_indices = np.argsort(mask_y_ppg.flatten())[-3:][::-1]  # 取前三大索引（倒序）

    # 2.Get the corresponding frequency and power amplitude
    top3_frequencies = mask_x_ppg[top3_indices].flatten()  # 取出对应频率
    top3_powers = mask_y_ppg[top3_indices].flatten()  # 取出对应功率幅度

    # 3.Calculate heart rate candidate values (HR1, HR2, HR3)
    hr_candidates = top3_frequencies * 60  # 频率转 BPM（心率）

    # 4.Calculate the new weighted power amplitude
    weighted_powers = np.array([
        power / abs(prev_hr_3p - hr) if prev_hr_3p != hr else power  # 避免除以 0
        for power, hr in zip(top3_powers, hr_candidates)
    ])

    # 5.Select HR with the largest weighted power amplitude as the final output
    final_hr = hr_candidates[np.argmax(weighted_powers)]

    return final_hr


prev_hr_3p = None
def calculate_fft_hr_3peak(ppg_signal, fs=30, hr_pass=(0.75, 2.5), by_welch=True, nfft=None):
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

    global prev_hr_3p
    if prev_hr_3p is None:
        index = np.argmax(mask_y_ppg, 0)
        current_hr = np.take(mask_x_ppg, index)[0] * 60
    else:
        # Calculate the current heart rate
        current_hr = compute_hr_3peak(mask_x_ppg, mask_y_ppg, prev_hr_3p)
    prev_hr_3p = current_hr
    return current_hr


# Index Filter + 3Peak Verification Filter
def compute_hr_index_3peak(index_filter2, mask_x_ppg, mask_y_ppg, prev_hr_id_3p):
    # 1.Find the top three points of power
    top3_indices = np.argsort(mask_y_ppg.flatten())[-3:][::-1]  # 取前三大索引（倒序）

    # 2.Get the corresponding frequency and power amplitude
    top3_frequencies = mask_x_ppg[top3_indices].flatten()  # 取出对应频率
    top3_powers = mask_y_ppg[top3_indices].flatten()  # 取出对应功率幅度

    # 3.Calculate heart rate candidate values (HR1, HR2, HR3)
    hr_candidates = top3_frequencies * 60  # 频率转 BPM（心率）

    # 4.Calculate the new weighted power amplitude
    weighted_powers = np.array([
        power / abs(prev_hr_id_3p - hr) if prev_hr_id_3p != hr else power  # 避免除以 0
        for power, hr in zip(top3_powers, hr_candidates)
    ])

    # 5.Select HR with the largest weighted power amplitude as the final output
    peak_index = top3_indices[np.argmax(weighted_powers)]
    # print("peak_index", peak_index)

    # 6.index
    index_use = index_filter2.filter(peak_index)

    final_hr = np.take(mask_x_ppg, index_use) * 60
    return final_hr


prev_hr_id_3p = None


def calculate_fft_hr_index_3peak(ppg_signal, index_filter2, fs=30, hr_pass=(0.75, 2.5), by_welch=True, nfft=None):
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

    global prev_hr_id_3p
    if prev_hr_id_3p is None:
        index = np.argmax(mask_y_ppg, 0)
        current_hr = np.take(mask_x_ppg, index)[0] * 60
    else:
        current_hr = compute_hr_index_3peak(index_filter2, mask_x_ppg, mask_y_ppg, prev_hr_id_3p)
    prev_hr_id_3p = current_hr
    return current_hr

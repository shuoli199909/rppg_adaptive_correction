import numpy as np
from collections import deque

class AMTC_Tracker:
    def __init__(self, fs=30, hr_pass=(0.65, 4.0),
                 buffer_len=30, window_size=180, lambda_reg=2.0, k=2, k2=5):
        """
        buffer_len: 参与trace分析的能量帧数量（如30, 即全局trace平滑30帧）
        window_size: 每帧分析窗（180为6秒）
        k2: 输出延迟帧数，影响平滑和实时性
        """
        self.fs = fs
        self.hr_pass = hr_pass
        self.buffer_len = buffer_len
        self.window_size = window_size
        self.lambda_reg = lambda_reg
        self.k = k
        self.k2 = k2
        self.mag_buffer = deque(maxlen=buffer_len)
        self.freq_ref = None

    def _compute_spectrogram(self, bvp_values):
        # 一帧能量谱
        fft_result = np.fft.fft(bvp_values)
        mag_spectrum = np.abs(fft_result)[:self.window_size // 2]
        freqs = np.fft.fftfreq(self.window_size, d=1.0/self.fs)[:self.window_size // 2]
        valid_idx = np.where((freqs >= self.hr_pass[0]) & (freqs <= self.hr_pass[1]))[0]
        return mag_spectrum[valid_idx], freqs[valid_idx]

    def update(self, bvp_values):
        """
        输入：bvp_values: 一帧信号（180点）
        返回：当前AMTC全局trace平滑心率（BPM）
        """
        self.window_size = len(bvp_values)
        mag_spectrum, freqs = self._compute_spectrogram(bvp_values)
        if self.freq_ref is None:
            self.freq_ref = freqs
        self.mag_buffer.append(mag_spectrum)
        # buffer未满时不输出心率
        if len(self.mag_buffer) < self.buffer_len:
            return np.nan

        Z = np.column_stack(list(self.mag_buffer))  # (频点, 帧数)
        M, N = Z.shape
        G = np.zeros((M, N))
        backtrack = np.zeros((M, N), dtype=int)
        G[:, 0] = Z[:, 0]
        for n in range(1, N):
            for m in range(M):
                idx = np.arange(max(0, m-self.k), min(M, m+self.k+1))
                candidates = G[idx, n-1]
                backtrack[m, n] = idx[np.argmax(candidates)]
                G[m, n] = Z[m, n] + np.max(candidates)
        # 回溯全局trace
        trace = np.zeros(N, dtype=int)
        trace[-1] = np.argmax(G[:, -1])
        for n in range(N-2, -1, -1):
            trace[n] = backtrack[trace[n+1], n+1]
        # online-AMTC输出延迟
        idx = -self.k2 if N > self.k2 else -1
        current_freq = self.freq_ref[trace[idx]]
        current_hr = current_freq * 60
        if not (40 <= current_hr <= 200):
            current_hr = np.nan
        return current_hr

    def reset(self):
        """处理下一个视频或新流时重置"""
        self.mag_buffer.clear()
        self.freq_ref = None

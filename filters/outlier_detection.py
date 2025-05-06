"""
Outlier Detection Filter
"""
# Author: Yunfei Tian
# Date: 2025/04/15

import numpy as np
from collections import deque


class OutlierFilter:
    def __init__(self, window_size=5, threshold_bpm=30, std_hr_factor=2):
        self.window_size = window_size  # 滑动窗口大小
        self.threshold_bpm = threshold_bpm  # 最大心率跳变阈值
        self.std_hr_factor = std_hr_factor  # 最大心率跳变阈值
        self.hr_values = deque(maxlen=window_size)  # 存储最近 N 个心率值

    def is_outlier(self, current_hr):
        """
        Determine whether the current heart rate is an outlier:
        - If the heart rate bet > threshold_bpm (maximum allowable jump per second), it is considered abnormal
        - If the current HR is far from the mean `2*standard deviation`, it is considered an outlier
        """
        if len(self.hr_values) < self.window_size:
            return False  # 不够数据，直接接受

        prev_hr = self.hr_values[-1]  # 取最近的心率
        mean_hr = np.mean(self.hr_values)
        std_hr = np.std(self.hr_values)

        # 跳变过大
        if abs(current_hr - prev_hr) > self.threshold_bpm:
            return True

        # 超过 2 倍标准差
        if abs(current_hr - mean_hr) > self.std_hr_factor * std_hr:
            return True

        return False

    def update(self, current_hr):
        """
        Processing new heart rate values:
        - If an outlier is detected, discard it and use the previous heart rate value instead
        - Otherwise update the window and return the smoothed heart rate
        """
        if self.is_outlier(current_hr):
            # print(f"异常心率检测: {current_hr} 被丢弃，使用前值 {self.hr_values[-1]}")
            return self.hr_values[-1]  # 返回前一个心率

        self.hr_values.append(current_hr)  # 添加新心率
        # return np.mean(self.hr_values)  # 计算滑动均值平滑心率
        return current_hr  # 计算滑动均值平滑心率


if __name__ == "__main__":
    tracker = OutlierFilter(window_size=5, threshold_bpm=30)

    hr_data = [75, 76, 77, 120, 78, 79, 130, 80, 81, 82]  # 120 和 130 可能是异常值
    print("Raw HR | Filtered HR")
    print("--------------------")

    for hr in hr_data:
        filtered_hr = tracker.update(hr)
        print(f"{hr}  |  {filtered_hr:.2f}")
        break
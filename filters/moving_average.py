"""
MovingAverageFilter
"""
# Author: Yunfei Tian
# Date: 2025/04/15

from collections import deque


class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size  # 滑动窗口大小
        self.values = deque(maxlen=window_size)  # 维护最近 N 个心率值

    def update(self, new_hr):
        self.values.append(new_hr)  # 添加新心率值
        return sum(self.values) / len(self.values)  # 计算平均值

if __name__ == "__main__":
    hr_data = [78, 80, 79, 77, 120, 76, 75, 74, 80, 77]  # 模拟心率数据
    ma_filter = MovingAverageFilter(window_size=5)  # 3 个值滑动窗口

    print("Raw HR | Filtered HR")
    print("--------------------")
    for hr in hr_data:
        print(f"{hr}  |  {ma_filter.update(hr):.2f}")
        break

"""
Our IndexFilter
"""

# Author: Yunfei Tian
# Date: 2025/04/20

import numpy as np


def calculate_median_list(lst):
    # 提取出单元素列表中的数字
    single_nums = [item[0] for item in lst]
    sorted_nums = sorted(single_nums)
    n = len(sorted_nums)
    if n % 2 == 0:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    else:
        median = sorted_nums[n // 2]
    return np.array([median], dtype='int64')


def calculate_median_int(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    if n % 2 == 0:
        median = (sorted_lst[n // 2 - 1] + sorted_lst[n // 2]) / 2
    else:
        median = sorted_lst[n // 2]
    return np.array(median, dtype='int64')


def calculate_minimum_list(lst):
    # 提取出单元素列表中的数字
    single_nums = [item[0] for item in lst]
    minimum = min(single_nums)
    return np.array([minimum], dtype='int64')


def calculate_minimum_int(lst):
    minimum = min(lst)
    return np.array(minimum, dtype='int64')


def calculate_mean_list(lst):
    single_nums = [item[0] for item in lst]
    n = len(single_nums)
    mean = sum(single_nums) // n
    return np.array([mean], dtype='int64')


class IndexFilter:
    def __init__(self, rise_factor=1, fall_factor=1):
        self.index_use = None
        self.begin_count = 0
        self.begin_vector = []
        self.begin_num = 10
        self.index_vector = []
        self.rise_factor = rise_factor
        self.fall_factor = fall_factor

    def filter(self, index):
        if self.index_use is None or self.begin_count < self.begin_num:
            self.begin_vector.append(index)
            self.index_use = index
            self.begin_count += 1
            return index

        if self.begin_count == self.begin_num:
            self.begin_count += 1
            if index.ndim == 0:
                self.index_use = calculate_median_int(self.begin_vector)
            if index.ndim == 1:
                self.index_use = calculate_median_list(self.begin_vector)

            return self.index_use

        index_deta = abs(self.index_use - index)

        if index_deta <= 1:
            self.index_use = index
            self.index_vector.clear()
        else:
            if not self.index_vector:
                self.index_vector.append(index)

            elif index == self.index_vector[-1]:
                self.index_vector.append(index)

            elif ((self.index_vector[-1] - self.index_use) > 0 and (index - self.index_use) > 0) or \
                    ((self.index_vector[-1] - self.index_use) < 0 and (index - self.index_use) < 0):
                self.index_vector.append(index)

            else:
                self.index_vector.clear()
                self.index_vector.append(index)

        # chage to larger more easy
        if index - self.index_use > 0 and len(self.index_vector) >= index_deta // self.rise_factor:
            self.index_use = index
            self.index_vector.clear()

        # chage to smaller
        if index - self.index_use <= 0 and len(self.index_vector) >= index_deta // self.fall_factor:
            self.index_use = index
            self.index_vector.clear()
        return self.index_use

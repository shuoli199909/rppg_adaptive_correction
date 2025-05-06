
"""
# Draw figure and save data
# Author: Yunfei Tian
# Date: 2025/04/13
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def bland_altman_plot_combined(hrs_no_all, hrs_id_all, hrs_gt_all, acceptable=10, title='Combined Bland-Altman Plot',
                               show=True):
    # Calculate the first set of data differences and averages
    diff_hrs_no = np.array(hrs_no_all) - np.array(hrs_gt_all)
    mean_hrs_no = (np.array(hrs_no_all) + np.array(hrs_gt_all)) / 2

    # Calculate the second set of data differences and averages
    diff_hrs_id = np.array(hrs_id_all) - np.array(hrs_gt_all)
    mean_hrs_id = (np.array(hrs_id_all) + np.array(hrs_gt_all)) / 2

    # Calculate the mean and standard deviation of the difference
    mean_diff_hrs_no = np.mean(diff_hrs_no)
    std_diff_hrs_no = np.std(diff_hrs_no)
    upper_limit_hrs_no = mean_diff_hrs_no + 1.96 * std_diff_hrs_no
    lower_limit_hrs_no = mean_diff_hrs_no - 1.96 * std_diff_hrs_no

    mean_diff_hrs_id = np.mean(diff_hrs_id)
    std_diff_hrs_id = np.std(diff_hrs_id)
    upper_limit_hrs_id = mean_diff_hrs_id + 1.96 * std_diff_hrs_id
    lower_limit_hrs_id = mean_diff_hrs_id - 1.96 * std_diff_hrs_id

    # Statistics the number of diff_hrs_no and diff_hrs_id in the range -10 to 10
    diff_hrs_no_counts = np.count_nonzero((diff_hrs_no >= -acceptable) & (diff_hrs_no <= acceptable))
    mean_hrs_counts = np.count_nonzero((diff_hrs_id >= -acceptable) & (diff_hrs_id <= acceptable))

    if not show:
        return diff_hrs_no_counts, mean_hrs_counts

    plt.figure(figsize=(10, 6), dpi=200)
    plt.scatter(mean_hrs_no, diff_hrs_no, color='blue', s=20, marker='o', alpha=0.5, label='without vs gt')
    plt.scatter(mean_hrs_id, diff_hrs_id, color='red', s=20, marker='s', alpha=0.5, label='ours vs gt')
    # Manually expand the x-axis to make it slightly larger than the data range, ensuring full fill
    x_min = min(np.min(mean_hrs_no), np.min(mean_hrs_id)) - 5
    x_max = max(np.max(mean_hrs_no), np.max(mean_hrs_id)) + 5
    plt.xlim(x_min, x_max)  # 设定整个 x 轴范围

    # Add transparent mask (ordinate -10 to 10, completely covering the entire x-axis)
    plt.fill_betweenx(
        np.linspace(-acceptable, acceptable, 100),  # Y 值范围
        x_min, x_max,  # 覆盖整个 X 轴范围
        color='lightgreen', alpha=0.2  # 浅绿色蒙版
    )

    # Draw the average difference line and upper and lower limit line (hrs_no),
    # set label to None to avoid displaying legend
    plt.axhline(mean_diff_hrs_no, color='blue', linestyle='--', label=None)
    plt.axhline(upper_limit_hrs_no, color='blue', linestyle='--', label=None)
    plt.axhline(lower_limit_hrs_no, color='blue', linestyle='--', label=None)

    # Draw the average difference line and upper and lower limit line (hrs_id),
    # set label to None to avoid displaying legend
    plt.axhline(mean_diff_hrs_id, color='red', linestyle='--', label=None)
    plt.axhline(upper_limit_hrs_id, color='red', linestyle='--', label=None)
    plt.axhline(lower_limit_hrs_id, color='red', linestyle='--', label=None)

    plt.title(title, fontsize=14)
    plt.xlabel('Mean', fontsize=12)
    plt.ylabel('Difference', fontsize=12)
    plt.legend()

    plt.grid(True)
    plt.show()
    return diff_hrs_no_counts, mean_hrs_counts


def list_average_stack(input_list, window_size):
    # New list, used to store the average value for each window
    result = []

    # Iterate through the list, group by window size and calculate the average
    for i in range(0, len(input_list), window_size):
        # 获取当前窗口的数据
        window = input_list[i:i + window_size]

        # Calculate the average value of this window
        avg = sum(window) / len(window)
        result.append(avg)

    return result


def sublist_average_stack(input_sublist, window_size):
    result = []
    for sublist in input_sublist:
        sub_result = []
        # Iterate through the current sublist,
        # group by window size and calculate the average value
        window_size = len(sublist) // 2
        for i in range(0, len(sublist), window_size):
            window = sublist[i:i + window_size]  # 获取窗口数据
            avg = sum(window) / len(window)  # 计算平均值
            sub_result.append(avg)

        result.extend(sub_result)  # 将当前子列表的平均值列表添加到结果中

    return result



def plot_boxplot(maes_dicts, title="", skip_double=False, y_min=None, y_max=None):
    """
    Draw a TEU chart of multiple error lists.
    parameter:
    - maes_dicts: A dictionary with the keys as the name of the algorithm and the
    value as the error list of the algorithm
    - title: (optional) Chart title
    - skip_double: (optional) Whether to skip items containing two underscores in the key,
    default to False
    """
    valid_labels = []
    valid_error_lists = []

    for key, value in maes_dicts.items():
        if skip_double and key.count("_") == 2:
            continue
        valid_labels.append(key)
        valid_error_lists.append(value)


    plt.figure(figsize=(10, 6), dpi=200)
    ax = sns.boxplot(data=valid_error_lists)

    try:
        median_value = np.median(maes_dicts["hrs_id"])
        print("plot_boxplot median_value", median_value)
        # Draw the red median line across the entire graph
        plt.axhline(y=median_value, color='red', linestyle='-', linewidth=2)
    except ValueError:
        pass

    plt.xticks(ticks=range(len(valid_labels)), labels=valid_labels, fontsize=10)
    plt.title(title, fontsize=14)
    plt.ylabel("Error", fontsize=12)
    plt.xlabel("Algorithms", fontsize=12)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(bottom=y_min)
    elif y_max is not None:
        plt.ylim(top=y_max)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_violinplot(maes_dicts, title="", skip_double=False, y_min=None, y_max=None):
    """
    Draw a violin diagram of multiple error lists.
    parameter:
    - maes_dicts: A dictionary with the keys as the name of the algorithm and the value as the error list of the algorithm
    - title: (optional) Chart title
    - skip_double: (optional) Whether to skip items containing two underscores in the key, default to False
    """
    valid_labels = []
    valid_error_lists = []

    for key, value in maes_dicts.items():
        if skip_double and key.count("_") == 2:
            continue
        valid_labels.append(key)
        valid_error_lists.append(value)

    plt.figure(figsize=(10, 6), dpi=200)
    ax = sns.violinplot(data=valid_error_lists)
    try:
        median_value = np.median(maes_dicts["hrs_id"])
        plt.axhline(y=median_value, color='red', linestyle='-', linewidth=2)
    except ValueError:
        pass

    plt.xticks(ticks=range(len(valid_labels)), labels=valid_labels, fontsize=10)

    plt.title(title, fontsize=14)
    plt.ylabel("Error", fontsize=12)
    plt.xlabel("Algorithms", fontsize=12)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(bottom=y_min)
    elif y_max is not None:
        plt.ylim(top=y_max)

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def save_hrs_to_csv(hrs_no, hrs_id, hrs_gt, idx_no, idx_id, name):
    if not (len(hrs_no) == len(hrs_id) == len(hrs_gt)):
        raise ValueError("All input lists must have the same length")
    filename = f"{name}.csv"
    data = {
        'hrs_no': hrs_no,
        'hrs_id': hrs_id,
        'hrs_gt': hrs_gt,
        'idx_no': idx_no,
        'idx_id': idx_id,
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data successfully written to {filename}")

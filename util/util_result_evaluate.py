"""
Evaluate to calc MAE, RMSE, CORR
# Author: Yunfei Tian
# Date: 2025/04/20
"""

import numpy as np
from dtaidistance import dtw


def evaluate(data1, data2):
    # Assume data1 and data2 are two columns of equal length data
    data1 = np.array(data1)
    data2 = np.array(data2)

    # 计算 MAE（Mean Absolute Error）
    mae = np.mean(np.abs(data1 - data2))

    # 计算 RMSE（Root Mean Squared Error）
    rmse = np.sqrt(np.mean((data1 - data2) ** 2))
    return mae.item(), rmse.item()


def evaluate_multi(data_dict, data2):
    # Assume that data_list is a list of dictionaries, each dictionary contains a key-value pair
    # data2 is an array containing the numeric value of the second parameter
    maes = {}
    rmses = {}
    # 获取字典中的 key 和 value
    for key, data1 in data_dict.items():
        # 计算 MAE 和 RMSE
        mae, rmse = evaluate(data1, data2)
        maes[key] = mae
        rmses[key] = rmse
    return maes, rmses


def evaluate_multi_corr(data_dict, data2):
    maes = {}
    rmses = {}
    correlations = {}
    for key, data1 in data_dict.items():
        mae, rmse = evaluate(data1, data2)  # 计算 MAE 和 RMSE
        maes[key] = mae
        rmses[key] = rmse
        if len(data1) == len(data2):
            # The smaller the value, the more similar it means [0,+inf]
            correlation = dtw.distance(data1, data2)
        else:
            correlation = np.nan  #Returns NaN when the length does not match
        correlations[key] = correlation
    return maes, rmses, correlations


if __name__ == "__main__":
    data_list = {"a": [2, 2, 2], "b": [5, 2, 3], "c": [10, 20, 30]}
    data2 = [2, 2, 2]
    # 计算 MAE 和 RMSE
    maes, rmses = evaluate_multi(data_list, data2)
    print(maes)
    print("--------")
    print(rmses)

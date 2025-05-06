"""
load raw data from rppg signal.
# Author: Yunfei Tian
# Date: 2025/04/15
"""

import pandas as pd


# load excel
def load_and_split_csv_(filename):
    df = pd.read_csv(filename, sep=',')  # 读取 CSV，假设是 Tab 分隔符
    # print("load_and_split_csv df.columns：",  df.columns)
    roi_groups = {roi: df[df['ROI'] == roi] for roi in df['ROI'].unique()}  # 根据 ROI 拆分数据
    return roi_groups


def load_and_split_csv(filename):
    df = pd.read_csv(filename, sep=',')  # 读取 CSV 文件
    if 'ROI' not in df.columns:
        df['ROI'] = 'glabella'  # 如果没有 ROI 列，添加一个默认值
    roi_groups = {roi: df[df['ROI'] == roi] for roi in df['ROI'].unique()}  # 按 ROI 拆分数据
    return roi_groups


def extract_item_data(roi_data, item_str, start_index, end_index):
    """
    Intercept BVP data from the specified ROI data between start_index and end_index
    """
    return roi_data.iloc[start_index:end_index][item_str].values


def load_extract_item_data(filename, item_str1=None, item_str2=None):
    df = pd.read_csv(filename, sep=',')
    item1, item2 = None, None
    if item_str1 is not None:
        item1 = df[item_str1].to_numpy()
    if item_str2 is not None:
        item2 = df[item_str2].to_numpy()
    return item1, item2

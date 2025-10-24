import os
import numpy as np
import glob
import os
import glob
import numpy as np
import pandas as pd

def read_wave_from_csv(folder_path, str="*_wave.csv"):
    """
    在指定文件夹中查找后缀为 _wave.csv 的文件，并读取其中一列数据为 BVP 信号。
    返回：np.ndarray 一维数组
    """
    # 搜索 _wave.csv 文件
    csv_files = glob.glob(os.path.join(folder_path, str))
    if not csv_files:
        raise FileNotFoundError(f"未找到 _wave.csv 文件于 {folder_path}")

    csv_file = csv_files[0]  # 只取第一个符合条件的文件
    print(f"读取文件: {csv_file}")

    # 加载数据（假设只有一列）
    raw_data = np.loadtxt(csv_file, delimiter=",")   # 这是一个 np.ndarray

    # # Step 1: 归一化（假设原始是像素值 0-255）
    # normalized = raw_data / 255.0
    # # Step 2: 中心化 + 标准化
    # standardized = (normalized - np.mean(normalized)) / np.std(normalized)

    # 标准化（均值为0，方差为1）
    standardized = (raw_data - np.mean(raw_data)) / np.std(raw_data)

    return standardized

def read_wave_mygt(folder_path, str_contains="mygt", col="BVP"):
        """
        从 folder_path 中查找包含 str_contains 的 csv 文件，并读取其中 col 列作为信号。
        数据将进行标准化后返回。
        """
        # 匹配包含 str_contains 的 csv 文件
        csv_files = glob.glob(os.path.join(folder_path, f"*{str_contains}*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"未找到包含 '{str_contains}' 的 CSV 文件于 {folder_path}")

        csv_file = csv_files[0]  # 默认取第一个匹配的
        print(f"读取文件: {csv_file}")

        # 使用 pandas 读取数据
        df = pd.read_csv(csv_file)

        # 如果只有一列，且没有表头
        if df.shape[1] == 1 and df.columns[0] == 0:
            signal = df.iloc[:, 0].values
        else:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 不存在于 {csv_file} 中")
            signal = df[col].values

        # 标准化
        standardized = (signal - np.mean(signal)) / np.std(signal)
        return standardized

# 示例用法：
folder = "/home/robbie/tyf_data/datasets/BUAA-MIHR_WE/Sub 13_lux 25.1"
bvp_signal = read_wave_mygt(folder)
print(bvp_signal.shape, bvp_signal[:100])

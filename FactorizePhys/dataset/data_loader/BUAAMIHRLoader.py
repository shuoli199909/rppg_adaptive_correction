"""The dataloader for UBFC-rPPG dataset.

Details for the UBFC-rPPG Dataset see https://sites.google.com/view/ybenezeth/ubfcrppg.
If you use this dataset, please cite this paper:
S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
"""
import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager
import pandas as pd
import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class BUAAMIHRLoader(BaseLoader):
    """The data loader for the UBFC-rPPG dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes an UBFC-rPPG dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- subject1/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |   |-- subject2/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                     |...
                     |   |-- subjectn/
                     |       |-- vid.avi
                     |       |-- ground_truth.txt
                -----------------
                name(string): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For UBFC-rPPG dataset)."""
        print("tyf get_raw_data data_path", data_path)
        data_dirs = glob.glob(data_path + os.sep + "Sub*") # tyf change
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        # dirs = [{"index": re.search(
        #     'subject(\d+)', data_dir).group(0), "path": data_dir} for data_dir in data_dirs]
        dirs = [{"index": os.path.basename(d), "path": d} for d in data_dirs]
        print("tyf get_raw_data dirs", dirs)
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            # frames = self.read_video(
            #     os.path.join(data_dirs[i]['path'],"vid.avi"))

            # tyf add  to read BUAA
            frames = self.read_video(data_dirs[i]['path'], "*.avi")

        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(data_dirs[i]['path'],'*.npy')))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            # bvps = self.read_wave(
            #     os.path.join(data_dirs[i]['path'],"ground_truth.txt"))
            # tyf add  to read BUAA
            # bvps = self.read_wave(data_dirs[i]['path'], str="*_wave.csv")
            bvps = self.read_wave(data_dirs[i]['path'])

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video_backup(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_video(folder_path, str):
        """Reads a video file, returns frames(T, H, W, 3) """
        # 搜索 str 文件
        vid_files = glob.glob(os.path.join(folder_path, str))
        if not vid_files:
            raise FileNotFoundError(f"未找到 {str} 文件于 {folder_path}")

        vid_file = vid_files[0]  # 只取第一个符合条件的文件
        print(f"读取文件: {vid_file}")
        video_file = vid_file
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave_backup(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)

    @staticmethod
    def read_wave_origin_bvp(folder_path, str):
        """
        在指定文件夹中查找后缀为 _wave.csv 的文件，并读取其中一列数据为 BVP 信号。
        返回：np.ndarray 一维数组
        """
        # 搜索 _wave.csv 文件
        csv_files = glob.glob(os.path.join(folder_path, str))
        if not csv_files:
            raise FileNotFoundError(f"未找到 {str} 文件于 {folder_path}")

        csv_file = csv_files[0]  # 只取第一个符合条件的文件
        print(f"读取文件: {csv_file}")

        # 加载数据（假设只有一列）
        raw_data = np.loadtxt(csv_file, delimiter=",")  # 这是一个 np.ndarray
        # 标准化（均值为0，方差为1）
        standardized = (raw_data - np.mean(raw_data)) / np.std(raw_data)
        return standardized



    @staticmethod
    def read_wave(folder_path, str_contains="mygt", col="BVP"):
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

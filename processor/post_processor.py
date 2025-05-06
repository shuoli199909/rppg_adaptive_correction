
"""
HeartRateProcessor combine all post processor method of HR
# Author: Yunfei Tian
# Date: 2025/04/20
"""

from filters.index_filter import IndexFilter
from filters.kalman_filter import KalmanFilter
from filters.moving_average import MovingAverageFilter
from filters.outlier_detection import OutlierFilter
from filters.peak_verification import calculate_fft_hr_3peak, calculate_fft_hr_index_3peak
from processor.index_processor import calculate_fft_hr_just, calculate_fft_hr_index
from util.util_load_excel import extract_item_data


class HeartRateProcessor:
    def __init__(self, roi_data, csv_type, window, stride_second, file_name):
        self.roi_data = roi_data
        self.csv_type = csv_type
        self.file_name = file_name

        self.fps_int = None
        self.stride = None
        self.window_size = None
        self.get_csv_fps(window, stride_second)

        self.index_filter1 = None
        self.kf_filter1 = None
        self.ma_filter1 = None
        self.ol_filter1 = None
        self.ol_filter2 = None
        self.index_filter2 = None
        self.ma_filter2 = None
        self.kf_filter2 = None

    def get_csv_fps(self, window, stride_second):
        # 获取 fps 数据  old
        if self.csv_type == "old":
            start_idx, end_idx = 0, 1  # 这里是示例范围，你可以修改
            fps_values = extract_item_data(self.roi_data, "time", start_idx, end_idx)
            fps_int = int(fps_values[0])
        else:
            # 获取 fps 数据  new
            # 按照 Window Number 分组
            grouped_by_wn = self.roi_data.groupby('Window Number')
            # 遍历每个 Window Number 组
            for window_number, group in grouped_by_wn:
                time_start = group['Time'].iloc[0]  # 获取第一行的时间
                time_end = group['Time'].iloc[-1]  # 获取最后一行的时间
                # print("time_start, time_end",time_start, time_end)
                duration = round(time_end - time_start)  # 计算时间并取整
                # 计算 FPS (避免除以零)
                fps_int = len(group) / duration if duration > 0 else float('inf')
                # print(f"Window Number: {window_number}")
                # print(f"Duration: {duration} seconds")
                # print("-" * 30)
                break
        # print("fps_int", fps_int)
        self.stride = int(fps_int * stride_second)  # 每次移动的步长 seconds -> pics
        self.window_size = window * fps_int  # 每个窗口的长度
        self.fps_int = fps_int

    def process_bvp_values(self, bvp_values, hrs_no, hrs_id, hrs_kf, hrs_ma, hrs_ol, hrs_3p,
                           hrs_3p_id, hrs_id_ma, hrs_id_ol, hrs_id_kf, start_idx):

        # method 1: no process
        hr_no = calculate_fft_hr_just(bvp_values, fs=self.fps_int, hr_pass=(0.65, 4.0), nfft=0.05)
        hrs_no.append(hr_no)

        # method 2: IndexFilter
        if start_idx == 0:
            self.index_filter1 = IndexFilter(rise_factor=2, fall_factor=1)
        hr_id = calculate_fft_hr_index(bvp_values, self.index_filter1, fs=self.fps_int, hr_pass=(0.65, 4.0),
                                              nfft=0.05)
        hrs_id.append(hr_id)

        # method 3: kalmanFilter
        if start_idx == 0:
            self.kf_filter1 = KalmanFilter(process_noise_Q=1.0, measurement_noise_R=4.0, estimate_error_P=2.0,
                                           initial_hr=hr_no)
        hr_kf = self.kf_filter1.update(hr_no)
        hrs_kf.append(hr_kf)

        # method 4: MovingAverageFilter
        if start_idx == 0:
            self.ma_filter1 = MovingAverageFilter(window_size=5)  # 5 个值滑动窗口
        hr_ma = self.ma_filter1.update(hr_no)
        hrs_ma.append(hr_ma)

        # method 5: OutlierFilter
        if start_idx == 0:
            self.ol_filter1 = OutlierFilter(window_size=50, threshold_bpm=30, std_hr_factor=2)
        hr_ol = self.ol_filter1.update(hr_no)
        hrs_ol.append(hr_ol)

        # method 6: PeakFilter
        hr_3p = calculate_fft_hr_3peak(bvp_values, fs=self.fps_int, hr_pass=(0.65, 4.0), nfft=0.05)
        hrs_3p.append(hr_3p)

        # method 7: PeakFilter + IndexFilter
        if start_idx == 0:
            self.index_filter2 = IndexFilter(rise_factor=1)
        hr_3p_id = calculate_fft_hr_index_3peak(bvp_values, self.index_filter2, fs=self.fps_int, hr_pass=(0.65, 4.0),
                                                nfft=0.05)
        hrs_3p_id.append(hr_3p_id)

        # method 8: IndexFilter + MovingAverageFilter
        if start_idx == 0:
            self.ma_filter2 = MovingAverageFilter(window_size=5)  # 5 个值滑动窗口
        hr_id_ma = self.ma_filter2.update(hr_id)
        hrs_id_ma.append(hr_id_ma)

        # method 9: IndexFilter + OutlierFilter
        if start_idx == 0:
            self.ol_filter2 = OutlierFilter(window_size=100, threshold_bpm=30, std_hr_factor=2)
        hr_id_ol = self.ol_filter2.update(hr_id)
        hrs_id_ol.append(hr_id_ol)

        # method 10: IndexFilter + kalmanFilter
        if start_idx == 0:
            self.kf_filter2 = KalmanFilter(process_noise_Q=1.0, measurement_noise_R=4.0, estimate_error_P=2.0,
                                           initial_hr=hr_no)
        hr_id_kf = self.kf_filter2.update(hr_id)
        hrs_id_kf.append(hr_id_kf)

    def extract_and_process_hrs(self, ):
        hrs_no, hrs_id, hrs_kf, hrs_ma, hrs_ol, hrs_3p, hrs_3p_id, hrs_id_ma, hrs_id_ol, hrs_id_kf = [], [], [], [], [], [], [], [], [], []

        if self.csv_type == "old":
            for start_idx in range(0, len(self.roi_data) - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                bvp_values = extract_item_data(self.roi_data, "BVP", start_idx, end_idx)
                self.process_bvp_values(bvp_values, hrs_no, hrs_id, hrs_kf, hrs_ma, hrs_ol, hrs_3p,
                                        hrs_3p_id, hrs_id_ma, hrs_id_ol, hrs_id_kf, start_idx)

        else:
            grouped_by_wn = self.roi_data.groupby('Window Number')
            for start_idx, group in grouped_by_wn:

                bvp_values = group['BVP']
                self.process_bvp_values(bvp_values, hrs_no, hrs_id, hrs_kf, hrs_ma, hrs_ol, hrs_3p,
                                        hrs_3p_id, hrs_id_ma, hrs_id_ol, hrs_id_kf, start_idx)

        return hrs_no, hrs_id, hrs_kf, hrs_ma, hrs_ol, hrs_3p, hrs_3p_id, hrs_id_ma, hrs_id_ol, hrs_id_kf

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import pickle


class HMRF_Predictor:
    """
    HMRF 风格的运动伪影可信度判别与心率修正器。
    - 训练：fit_and_save(...) 一步到位（含特征抽取/划分/训练/保存）
    - 推理：predict_and_correct(...) 批量；predict_one(...) 在线单步
    """

    def __init__(self, model_path: Optional[str] = None, rf_kwargs: Optional[dict] = None):
        """
        model_path: 可选，若提供则从磁盘加载模型
        rf_kwargs: 传给 RandomForestClassifier 的参数（未加载时生效）
        """
        self.clf: Optional[RandomForestClassifier] = None
        self.hr_no_trust: Optional[float] = None  # 在线单步时的不可信回退记忆

        if model_path is not None:
            with open(model_path, 'rb') as f:
                self.clf = pickle.load(f)
        else:
            if rf_kwargs is None:
                rf_kwargs = dict(n_estimators=50, max_depth=5,
                                 class_weight='balanced', random_state=42)
            self.clf = RandomForestClassifier(**rf_kwargs)

    # ------------------------ 训练与持久化 ------------------------

    def fit_and_save(
        self,
        hrs_gt_all: List[float],
        hrs_no_all: List[float],
        hrs_bvp_all: List[np.ndarray],
        hrs_position_all: List[np.ndarray],
        threshold: float = 6.0,
        test_size: float = 0.1,
        random_state: int = 42,
        save_path: str = 'HMRF_BUAA100.pkl'
    ) -> Tuple[float, float]:
        """
        端到端：特征抽取 -> 划分 -> 训练 -> 保存模型
        返回 (train_pos_ratio, test_pos_ratio) 便于检查数据平衡性
        """
        # 特征与标签
        snrs = [self.calc_snr(np.array(bvp)) for bvp in hrs_bvp_all]
        X = self.feature_extract(hrs_bvp_all, hrs_position_all, hrs_no_all, snrs)
        y = self.generate_labels(hrs_no_all, hrs_gt_all, threshold=threshold)

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=y
        )

        # 训练
        assert self.clf is not None, "Classifier not initialized."
        self.clf.fit(X_train, y_train)

        # 保存
        with open(save_path, 'wb') as f:
            pickle.dump(self.clf, f)

        train_pos_ratio = float(np.mean(y_train))
        test_pos_ratio = float(np.mean(y_test))
        return train_pos_ratio, test_pos_ratio

    def save(self, path: str) -> None:
        assert self.clf is not None, "No classifier to save."
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)

    @classmethod
    def load(cls, path: str) -> "HMRF_Predictor":
        with open(path, 'rb') as f:
            clf = pickle.load(f)
        obj = cls(model_path=None)
        obj.clf = clf
        return obj

    # ------------------------ 批量推理接口 ------------------------

    def predict_batch(
        self,
        hrs_bvp_all_new: List[np.ndarray],
        hrs_position_all_new: List[np.ndarray],
        hrs_no_all_new: List[float]
    ) -> np.ndarray:
        """
        仅返回 0/1 可信标签（与 clf.predict 行为一致）
        """
        assert self.clf is not None, "Model not loaded/trained."
        snrs_new = [self.calc_snr(np.array(bvp)) for bvp in hrs_bvp_all_new]
        X_new = self.feature_extract(hrs_bvp_all_new, hrs_position_all_new, hrs_no_all_new, snrs_new)
        labels = self.clf.predict(X_new)
        return labels

    def predict_and_correct(
        self,
        hrs_bvp_all_new: List[np.ndarray],
        hrs_position_all_new: List[np.ndarray],
        hrs_no_all_new: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测并用“最近邻有效值”修正心率（等价于  predict_and_correct_hrs）
        返回 (label_hmrf_all, hrs_hmrf_all)
        """
        labels = self.predict_batch(hrs_bvp_all_new, hrs_position_all_new, hrs_no_all_new)
        hrs_hmrf_all = self.fill_by_nearest_neighbor(hrs_no_all_new, labels)
        return labels, hrs_hmrf_all

    # ------------------------ 在线单步接口（已保留你的逻辑） ------------------------

    def predict_one(self, hrs_bvp: np.ndarray, hrs_position: np.ndarray, hr_no_current: float) -> Tuple[int, float]:
        """
        单次预测：
        - label==0: 不可信 -> 返回上一次可信的 hr（self.hr_no_trust）
        - label==1: 可信 -> 返回当前 hr，并更新 self.hr_no_trust
        """
        assert self.clf is not None, "Model not loaded/trained."
        if self.hr_no_trust is None:
            self.hr_no_trust = float(hr_no_current)

        snr = self.calc_snr(np.array(hrs_bvp))
        X = self.feature_extract([hrs_bvp], [hrs_position], [hr_no_current], [snr])
        label = int(self.clf.predict(X)[0])

        if label == 0:  # 不可信：回退
            hr_result = float(self.hr_no_trust)
        else:           # 可信：更新记忆
            hr_result = float(hr_no_current)
            self.hr_no_trust = float(hr_no_current)

        return label, hr_result

    # ------------------------ 下方是原先的辅助函数，封装为静态/类方法 ------------------------

    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    @classmethod
    def bandpass_filter(cls, data, lowcut, highcut, fs, order=3):
        b, a = cls.butter_bandpass(lowcut, highcut, fs, order)
        return filtfilt(b, a, data)

    @classmethod
    def calc_snr(cls, bvp, fs=30):
        bvp = detrend(cls.bandpass_filter(bvp, 0.7, 4, fs))
        f = np.fft.rfftfreq(len(bvp), 1 / fs)
        pxx = np.abs(np.fft.rfft(bvp)) ** 2
        peak_idx = np.argmax(pxx)
        peak_band = (f >= f[peak_idx] - 0.15) & (f <= f[peak_idx] + 0.15)
        snr = (np.sum(pxx[peak_band]) + 1e-6) / (np.sum(pxx[~peak_band]) + 1e-6)
        return float(snr)

    @classmethod
    def motion_features(cls, position):
        position = np.asarray(position)
        dx = np.diff(position[:, 0])
        dy = np.diff(position[:, 1])
        disp = np.sqrt(dx ** 2 + dy ** 2)

        kurtosis = pd.Series(disp).kurtosis()
        skewness = pd.Series(disp).skew()
        zero_crossing_rate = np.mean(np.diff(np.sign(disp)) != 0)
        first_deriv = np.diff(disp)

        time_feats = [
            float(kurtosis), float(skewness), float(zero_crossing_rate),
            float(np.max(first_deriv)),
            float(np.mean(np.abs(first_deriv))),
            float(np.var(first_deriv))
        ]

        fs = 30
        disp_filt = cls.bandpass_filter(disp, 0.1, 10, fs)
        disp_detr = detrend(disp_filt)
        f = np.fft.rfftfreq(len(disp_detr), 1 / fs)
        pxx = np.abs(np.fft.rfft(disp_detr)) ** 2
        band_energies = [float(np.sum(pxx[(f >= band) & (f < band + 0.5)]))
                         for band in np.arange(0.1, 10, 0.5)]
        return time_feats + band_energies

    @classmethod
    def feature_extract(cls, hrs_bvp_all, hrs_position_all, hrs_no_all, snrs=None):
        features = []
        if snrs is None:
            snrs = [cls.calc_snr(np.array(bvp)) for bvp in hrs_bvp_all]
        for i in range(len(hrs_bvp_all)):
            motion_feat = cls.motion_features(np.array(hrs_position_all[i]))
            feat = motion_feat + [float(snrs[i]), float(hrs_no_all[i])]
            features.append(feat)
        return np.array(features, dtype=float)

    @staticmethod
    def generate_labels(hrs_no_all, hrs_gt_all, threshold=6.0):
        diff = np.abs(np.array(hrs_no_all, dtype=float) - np.array(hrs_gt_all, dtype=float))
        labels = (diff <= float(threshold)).astype(int)
        return labels

    @staticmethod
    def fill_by_nearest_neighbor(data, valid_mask):
        data = np.array(data, dtype=float)
        valid_mask = np.array(valid_mask, dtype=int)
        valid_idxs = np.where(valid_mask == 1)[0]
        result = data.copy()
        for i in range(len(data)):
            if valid_mask[i] == 0:
                left = valid_idxs[valid_idxs < i]
                right = valid_idxs[valid_idxs > i]
                if len(left) == 0 and len(right) == 0:
                    result[i] = float(np.mean(data))
                elif len(left) == 0:
                    result[i] = float(data[right[0]])
                elif len(right) == 0:
                    result[i] = float(data[left[-1]])
                else:
                    if (i - left[-1]) <= (right[0] - i):
                        result[i] = float(data[left[-1]])
                    else:
                        result[i] = float(data[right[0]])
        return result

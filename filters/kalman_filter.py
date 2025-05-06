"""
One dinmention KalmanFilter
"""
# Author: Yunfei Tian
# Date: 2025/04/15

import numpy as np


class KalmanFilter:
    def __init__(self, process_noise_Q=1.0, measurement_noise_R=4.0, estimate_error_P=2.0, initial_hr=75.0):
        """
        Initialize the Kalman filter parameters:
        - process_noise_Q: process noise (larger is more sensitive to new measurements)
        - measurement_noise_R: Measurement noise (the greater the sensor data is not trusted)
        - estimate_error_P: Estimate error covariance
        - initial_hr: Initial heart rate estimate
        """
        self.Q = process_noise_Q
        self.R = measurement_noise_R
        self.P = estimate_error_P
        self.x = initial_hr  # 初始估计的心率

    def update(self, current_hr):
        """
        Process the new heart rate measurement and return the smoothed heart rate.
        """
        # Prediction stage
        self.P = self.P + self.Q  # Update error covariance
        # Calculate Kalman Gain
        K = self.P / (self.P + self.R)
        # Update estimates
        self.x = self.x + K * (current_hr - self.x)
        # Update error covariance
        self.P = (1 - K) * self.P
        return self.x

# Test code
if __name__ == "__main__":
    # 模拟心率数据（带噪声）
    def generate_noisy_hr(num_samples=20, true_hr=75, noise_level=5):
        """
        Generate noisy heart rate data and simulate sensor measurements.
        """
        np.random.seed(42)
        noise = np.random.normal(0, noise_level, num_samples)  # 生成随机噪声
        hr_data = true_hr + noise  # 真实心率 + 噪声
        return hr_data


    kf = KalmanFilter(process_noise_Q=1.0, measurement_noise_R=4.0, estimate_error_P=2.0, initial_hr=75.0)

    hr_data = generate_noisy_hr(20, true_hr=75, noise_level=5)  # 生成心率数据
    print("Raw HR  | Filtered HR")
    for hr in hr_data:
        filtered_hr = kf.update(hr)
        print(f"{hr:.1f}  |  {filtered_hr:.1f}")
        break

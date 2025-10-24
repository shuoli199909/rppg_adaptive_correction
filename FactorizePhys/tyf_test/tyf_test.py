import random

import numpy as np

def read_wave(bvp_file):
    """Reads a bvp signal file."""
    with open(bvp_file, "r") as f:
        str1 = f.read()
        str1 = str1.split("\n")
        bvp = [float(x) for x in str1[0].split()]
    return np.asarray(bvp)


a = read_wave("ground_truth.txt")
print(len(a), a[:100])
import tensorflow as tf
print("TF devices:", tf.config.list_physical_devices())

prediction = list(range(1, 101))
stride = 3
window_frame_size = 10
for i in range(0, len(prediction) - window_frame_size + 1, stride):
    pred_window = prediction[i:i + window_frame_size]
    print(pred_window)
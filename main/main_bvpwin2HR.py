"""
mian to calc final HR
Author: Yunfei Tian
Date: 2025/04/25
"""

import os
import glob
import numpy as np

from processor.post_processor import HeartRateProcessor
from util.util_load_excel import load_and_split_csv, load_extract_item_data
from util.util_result_count_proportion import calculate_error_rates
from util.util_result_evaluate import evaluate_multi_corr
from util.util_result_save_show import list_average_stack, bland_altman_plot_combined, plot_boxplot

# Define windows and path
csv_type = "new"
dir_name = "your/data/path"
dataset_name = "LGI-PPGI"             # movement
dataset_name = "BUAA-MIHR-LOWLIGHT"  # low light
dataset_name = "UBFC-rPPG"           # still

selected_roi = "glabella"
# selected_roi = "left malar"
# selected_roi = "right malar"

window = 6 #s
stride_second = 1 #s
base_path = "base/path/" + dir_name + dataset_name  + "/"
base_path_gt = "/base/gr_path/" + dataset_name  + "/gt/"
save_figure_path = "./save_figure_" + dataset_name + "/"
os.makedirs(save_figure_path, exist_ok=True) # 如果目录不存在，则创建
save_csv_path = "./save_csv/"
os.makedirs(save_csv_path, exist_ok=True) # 如果目录不存在，则创建
csv_files_path = glob.glob(f"{base_path}/*.csv")  # 获取所有 CSV 文件

before_index, after_index = [], []
x_ppgs, y_ppgs = [], []

maes_dicts = {}
rmses_dicts = {}
corr_dicts = {}

hrs_no_all = []
hrs_id_all = []
hrs_kf_all = []
hrs_ma_all = []
hrs_ol_all = []
hrs_3p_all = []

hrs_gt_all = []
csv_files_use = []

bad_num = 0

for i, file_path_name in enumerate(csv_files_path):

    file_name = os.path.basename(file_path_name)
    csv_files_use.append(file_name)
    file_name_gt = "_".join(file_name.split("_")[:-1]) + ".csv"

    data_by_roi = load_and_split_csv(file_path_name)
    roi_data = data_by_roi[selected_roi]

    # Focus: Different methods to calculate heart rate
    hrs_id = []
    hrs_no = []
    hrs_kf = []
    hrs_ma = []
    hrs_ol = []
    hrs_3p = []
    hrs_3p_id = []
    hrs_id_ma = []
    hrs_id_ol = []
    hrs_id_kf = []

    hr_processor = HeartRateProcessor(roi_data, csv_type, window, stride_second, file_name)
    hrs_no, hrs_id, hrs_kf, hrs_ma, hrs_ol, hrs_3p, hrs_3p_id, hrs_id_ma, hrs_id_ol, hrs_id_kf = \
        hr_processor.extract_and_process_hrs()

    # Create an x-axis time series
    x = list(range(len(hrs_id)))
    x = [i * stride_second + window for i in x]  # 从第一个window后开始有结果

    # Get GT's hr
    filename_gt = base_path_gt + file_name_gt
    time_gt, bpm_gt = load_extract_item_data(filename_gt, item_str1="Time", item_str2="BPM")

    # Find the index closest to x in time_gt
    nearest_indices = [np.argmin(np.abs(time_gt - t)) for t in x]
    # Extract the time closest to x and BPM according to the index
    time_gt_down = time_gt[nearest_indices]
    bpm_gt_down = bpm_gt[nearest_indices]

    assert len(hrs_id) == len(hrs_no) == len(hrs_kf) == len(hrs_ma) == len(hrs_ol) == len(hrs_3p)

    # calc and save MAE 和 RMSE
    hrs_all_dict = {
        "hrs_no": hrs_no,
        "hrs_id": hrs_id,
        "hrs_kf": hrs_kf,
        "hrs_ma": hrs_ma,
        "hrs_ol": hrs_ol,
        "hrs_3p": hrs_3p,
        "hrs_3p_id": hrs_3p_id,
        "hrs_id_ma": hrs_id_ma,
        "hrs_id_ol": hrs_id_ol,
        "hrs_id_kf": hrs_id_kf,
    }

    #calc error， cave to dict
    mae_dict, rmse_dict, corr_dict = evaluate_multi_corr(hrs_all_dict, bpm_gt_down)
    for key, value in mae_dict.items():
        if maes_dicts.get(key):
            maes_dicts[key].append(mae_dict[key])
        else:
            maes_dicts[key] = [mae_dict[key]]

    for key, value in rmse_dict.items():
        if rmses_dicts.get(key):
            rmses_dicts[key].append(rmse_dict[key])
        else:
            rmses_dicts[key] = [rmse_dict[key]]

    for key, value in corr_dict.items():
        if corr_dicts.get(key):
            corr_dicts[key].append(corr_dict[key])
        else:
            corr_dicts[key] = [corr_dict[key]]


    # Save multiple list FOR real cases dictionary
    npname = "save_figure_all_case/"+ dataset_name + "/"  + file_name[0:-4] + ".npz"
    np.savez(npname,
              hrs_no=hrs_no,
              hrs_id=hrs_id,
              bpm_gt_down=bpm_gt_down,
            )
    print("Lists saved successfully as", npname, len(x), len(time_gt_down))

    # for draw bland altman
    hrs_no_all.extend(hrs_no)
    hrs_id_all.extend(hrs_id)
    hrs_kf_all.extend(hrs_kf)
    hrs_ma_all.extend(hrs_ma)
    hrs_3p_all.extend(hrs_3p)
    hrs_ol_all.extend(hrs_ol)
    hrs_gt_all.extend(bpm_gt_down)

# To draw BA pictures
average_num = 25
hrs_id_all_average = list_average_stack(hrs_id_all, window_size=average_num)
hrs_no_all_average = list_average_stack(hrs_no_all, window_size=average_num)
hrs_kf_all_average = list_average_stack(hrs_kf_all, window_size=average_num)
hrs_ma_all_average = list_average_stack(hrs_ma_all, window_size=average_num)
hrs_3p_all_average = list_average_stack(hrs_3p_all, window_size=average_num)
hrs_ol_all_average = list_average_stack(hrs_ol_all, window_size=average_num)
hrs_gt_all_average = list_average_stack(hrs_gt_all, window_size=average_num)
diff_hrs_no_counts, mean_hrs_counts = bland_altman_plot_combined(hrs_no_all_average, hrs_id_all_average,
                                                                 hrs_gt_all_average, acceptable=10,
                                                                 title=dataset_name + " " + str(average_num))
# In order to calculate the proportion of the error_range
calculate_error_rates(hrs_no_all_average, hrs_id_all_average, hrs_kf_all_average,
                      hrs_ma_all_average, hrs_3p_all_average, hrs_ol_all_average,
                          hrs_gt_all_average, error_range=10)

plot_boxplot(maes_dicts, title="",skip_double=True, )
plot_boxplot(rmses_dicts, title="",skip_double=True, )
plot_boxplot(corr_dicts, title="",skip_double=True, )
print("Happy Ending")
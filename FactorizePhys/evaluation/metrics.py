import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

from dtaidistance import dtw


def summarize_mae_dtw(maes_before, dtws_before, maes_after, dtws_after):
    def summary_stats(data, name):
        mean_val = np.mean(data)
        median_val = np.median(data)
        print(f"{name} - 平均数: {mean_val:.4f}, 中位数: {median_val:.4f}")

    summary_stats(maes_before, "MAE Before")
    summary_stats(dtws_before, "DTW Before")
    summary_stats(maes_after, "MAE After")
    summary_stats(dtws_after, "DTW After")


def compute_dtw_and_mae(data1, data2):
    data1 = np.array(data1)
    data2 = np.array(data2)

    # 计算 DTW 距离
    dtw_distance = dtw.distance(data1, data2)

    # 计算 MAE（Mean Absolute Error）
    mae = np.mean(np.abs(data1 - data2))

    return mae, dtw_distance



def calculate_mean_list(lst):
    single_nums = [item[0] for item in lst]
    n = len(single_nums)
    mean = sum(single_nums) // n
    return np.array([mean], dtype='int64')



class IndexFilterTen:
    def __init__(self, rise_factor=1, fall_factor=1):
        self.index_use = None
        self.begin_count = 0
        self.begin_vector_inde = []
        self.begin_num = 14
        self.index_vector = []
        self.rise_factor = rise_factor
        self.fall_factor = fall_factor

    def filter(self, index):
        if self.index_use is None or self.begin_count < self.begin_num:
            self.begin_vector_inde.append(index)
            self.index_use = index
            self.begin_count += 1
            return index

        if self.begin_count == self.begin_num:
            self.begin_count += 1

            index_inde = calculate_mean_list(self.begin_vector_inde)

            self.index_use = index_inde
            return self.index_use

        index_deta = abs(self.index_use - index)

        if index_deta <= 1:
            self.index_use = index
            self.index_vector.clear()
        else:
            if not self.index_vector:
                self.index_vector.append(index)

            elif index == self.index_vector[-1]:
                self.index_vector.append(index)

            # 方向一致
            elif ((self.index_vector[-1] - self.index_use) > 0 and (index - self.index_use) > 0) or \
                    ((self.index_vector[-1] - self.index_use) < 0 and (index - self.index_use) < 0):
                self.index_vector.append(index)

            # 忽大忽小
            else:
                self.index_vector.clear()
                self.index_vector.append(index)

        # chage to larger more easy
        if index - self.index_use > 0 and len(self.index_vector) >= index_deta // self.rise_factor:
            self.index_use = index
            self.index_vector.clear()

        # chage to smaller
        if index - self.index_use <= 0 and len(self.index_vector) >= index_deta // self.fall_factor:
            self.index_use = index
            self.index_vector.clear()
        return self.index_use


def calculate_error_rates_errorfirst(hrs_no_all, hrs_gt_all, error_range):
    if not (len(hrs_no_all) == len(hrs_gt_all)):
        raise ValueError("All input lists must have the same length")

    def compute_ratio(pred_list, gt_list, error_range, n=25):
        # 计算每个点的误差
        point_errors = [abs(pred - gt) for pred, gt in zip(pred_list, gt_list)]

        # 分组取平均
        grouped_errors = []
        for i in range(0, len(point_errors), n):
            group = point_errors[i:i + n]
            avg_error = sum(group) / len(group)
            grouped_errors.append(avg_error)

        # 统计有多少组平均误差小于等于 error_range
        count_within_range = sum(1 for err in grouped_errors if err <= error_range)
        return round((count_within_range / len(grouped_errors)) * 100, 4)

    results = {
        "hrs_no_all": compute_ratio(hrs_no_all, hrs_gt_all, error_range),
    }
    for key, value in results.items():
        print(f"{key}: {value:.4f}%")
    return results





def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    predict_hr_index_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    maes_before = list()
    dtws_before = list()
    maes_after = list()
    dtws_after = list()
    print("Calculating metrics!")
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])
        print("tyf calculate_metrics prediction, label",len(prediction),len(label))
        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        # for i in range(0, len(prediction), window_frame_size):
        #     pred_window = prediction[i:i+window_frame_size]
        #     label_window = label[i:i+window_frame_size]
        stride = 30  # 步长为 30
        start_idx = True
        predict_hr_fft_one = list()
        predict_hr_index_fft_one = list()
        gt_hr_fft_one = list()
        for i in range(0, len(prediction) - window_frame_size + 1, stride):
            pred_window = prediction[i:i + window_frame_size]
            label_window = label[i:i + window_frame_size]


            print("tyf calculate_metrics window_frame_size", window_frame_size)
            print("tyf calculate_metrics len(pred_window), len(label_window)", len(pred_window), len(label_window))
            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, _, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                print("tyf calculate_metrics len pred_window", len(pred_window), len(label_window))
                if start_idx:
                    index_filter1 = IndexFilterTen(rise_factor=2, fall_factor=1)
                    start_idx = False
                    print("tyf a new time for index filter")
                gt_hr_fft, pred_hr_fft, pred_hr_index_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT', index=index_filter1)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                predict_hr_index_fft_all.append(pred_hr_index_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
                # tyf for a video
                gt_hr_fft_one.append(gt_hr_fft)
                predict_hr_fft_one.append(pred_hr_fft)
                predict_hr_index_fft_one.append(pred_hr_index_fft)

            else:
                raise ValueError("Inference evaluation method name wrong!")

        mae_before, dtw_before = compute_dtw_and_mae(predict_hr_fft_one, gt_hr_fft_one)
        mae_after,  dtw_after = compute_dtw_and_mae(predict_hr_index_fft_one, gt_hr_fft_one)
        maes_before.append(mae_before)
        dtws_before.append(dtw_before)
        maes_after.append(mae_after)
        dtws_after.append(dtw_after)
        print("a video finished")
    # calc my error
    summarize_mae_dtw(maes_before, dtws_before, maes_after, dtws_after)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_train':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        predict_hr_index_fft_all = np.array(predict_hr_index_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                # before
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
                print("FFT MAE (FFT Percent): range {0} is {1}".format(5,
                    calculate_error_rates_errorfirst(predict_hr_fft_all, gt_hr_fft_all, error_range=5)))
                print("FFT MAE (FFT Percent): range {0} is {1}".format(10,
                    calculate_error_rates_errorfirst(predict_hr_fft_all, gt_hr_fft_all, error_range=10)))
                # after
                MAE_FFT = np.mean(np.abs(predict_hr_index_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_index_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label Index): {0} +/- {1}".format(MAE_FFT, standard_error))
                print("FFT MAE (FFT Percent Index): range {0} is {1}".format(5,
                    calculate_error_rates_errorfirst(predict_hr_index_fft_all, gt_hr_fft_all, error_range=5)))
                print("FFT MAE (FFT Percent Index): range {0} is {1}".format(10,
                    calculate_error_rates_errorfirst(predict_hr_index_fft_all, gt_hr_fft_all, error_range=10)))

            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.sqrt(np.std(np.square(predict_hr_fft_all - gt_hr_fft_all))) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:  
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.sqrt(np.std(np.square(predict_hr_peak_all - gt_hr_peak_all))) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("PEAK SNR (PEAK Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("PEAK MACC (PEAK Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

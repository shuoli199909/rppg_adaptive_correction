"""
# Count proportion of all
# Author: Yunfei Tian
# Date: 2025/04/20
"""

def calculate_error_rates(hrs_no_all, hrs_id_all, hrs_kf_all, hrs_ma_all, hrs_3p_all, hrs_ol_all, hrs_gt_all,
                          error_range):
    if not (len(hrs_no_all) == len(hrs_id_all) == len(hrs_kf_all) == len(hrs_ma_all) == len(hrs_3p_all) == len(
            hrs_ol_all) == len(hrs_gt_all)):
        raise ValueError("All input lists must have the same length")

    def compute_ratio(pred_list, gt_list, error_range):
        count_within_range = sum(1 for pred, gt in zip(pred_list, gt_list) if abs(pred - gt) <= error_range)
        return round((count_within_range / len(gt_list)) * 100, 2)

    results = {
        "hrs_no_all": compute_ratio(hrs_no_all, hrs_gt_all, error_range),
        "hrs_id_all": compute_ratio(hrs_id_all, hrs_gt_all, error_range),
        "hrs_kf_all": compute_ratio(hrs_kf_all, hrs_gt_all, error_range),
        "hrs_ma_all": compute_ratio(hrs_ma_all, hrs_gt_all, error_range),
        "hrs_ol_all": compute_ratio(hrs_ol_all, hrs_gt_all, error_range),
        "hrs_3p_all": compute_ratio(hrs_3p_all, hrs_gt_all, error_range)
    }

    for key, value in results.items():
        print(f"{key}: {value:.2f}%")

    return results


def calculate_error_rates_errorfirst(hrs_no_all, hrs_id_all, hrs_kf_all, hrs_ma_all, hrs_3p_all, hrs_ol_all, hrs_gt_all,
                                     error_range):
    if not (len(hrs_no_all) == len(hrs_id_all) == len(hrs_kf_all) == len(hrs_ma_all) == len(hrs_3p_all) == len(
            hrs_ol_all) == len(hrs_gt_all)):
        raise ValueError("All input lists must have the same length")

    def compute_ratio(pred_list, gt_list, error_range, n=20):
        # 计算每个点的误差
        point_errors = [abs(pred - gt) for pred, gt in zip(pred_list, gt_list)]

        # 分组取平均
        grouped_errors = []
        for i in range(0, len(point_errors), n):
            group = point_errors[i:i + n]
            avg_error = sum(group) / len(group)
            grouped_errors.append(avg_error)

        # Statistics how many groups have an average error of less than or equal to error_range
        count_within_range = sum(1 for err in grouped_errors if err <= error_range)
        return round((count_within_range / len(grouped_errors)) * 100, 4)

    results = {
        "hrs_no_all": compute_ratio(hrs_no_all, hrs_gt_all, error_range),
        "hrs_id_all": compute_ratio(hrs_id_all, hrs_gt_all, error_range),
        "hrs_kf_all": compute_ratio(hrs_kf_all, hrs_gt_all, error_range),
        "hrs_ma_all": compute_ratio(hrs_ma_all, hrs_gt_all, error_range),
        "hrs_ol_all": compute_ratio(hrs_ol_all, hrs_gt_all, error_range),
        "hrs_3p_all": compute_ratio(hrs_3p_all, hrs_gt_all, error_range)
    }

    for key, value in results.items():
        print(f"{key}: {value:.4f}%")

    return results


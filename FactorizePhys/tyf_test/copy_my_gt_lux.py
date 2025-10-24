import os
import shutil
import glob
import re

# 原始 GT 文件路径
src_dir = "/home/robbie/tyf_file/rppg_index_code_data/data_gt/BUAA-MIHR-LOWLIGHT/gt"
# 目标根目录
dst_root = "/home/robbie/tyf_data/datasets/BUAA-MIHR_VERYLOW"

# 指定的 lux 条件
# target_luxes = ["lux6.3", "lux10.0", "lux15.8", "lux25.1"]
target_luxes = ["lux1.0", "lux1.6", "lux2.5", "lux4.0"]



# 遍历所有 csv 文件
count = 0
for i, csv_path in enumerate(glob.glob(os.path.join(src_dir, "*.csv"))):
    filename = os.path.basename(csv_path)  # eg: 04_lux6.3.csv

    # 判断是否是目标 lux 文件
    if any(lux in filename for lux in target_luxes):
        # 提取 subject 和 lux
        match = re.match(r"(\d+)_lux(\d+\.\d+)\.csv", filename)
        if not match:
            print(f"⚠️ 文件名不符合格式，跳过: {filename}")
            continue

        subject_id = match.group(1).zfill(2)  # 保证是两位数，比如 "04"
        lux_str = match.group(2)  # 比如 "6.3"

        # 构造目标文件夹路径
        dst_folder = os.path.join(dst_root, f"Sub {subject_id}_lux {lux_str}")
        if not os.path.exists(dst_folder):
            print(f"❌ 目标文件夹不存在: {dst_folder}")
            continue

        # 拷贝文件
        dst_path = os.path.join(dst_folder, "mygt_" + filename)

        # shutil.copy2(csv_path, dst_path)
        count += 1
        print(f"✅ 拷贝: {count},  {filename} -> {dst_path}")

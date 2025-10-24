

import os
import shutil

# 原始路径和目标路径
src_root = "/home/robbie/tyf_data/datasets/BUAA-MIHR_ORIGIN"
dst_root = "/home/robbie/tyf_data/datasets/BUAA-MIHR_VERYLOW"

# 要保留的 lux 文件夹名称
target_lux_list = ["lux 1.0", "lux 1.6", "lux 2.5", "lux 4.0"]

# 遍历所有 Sub 目录（Sub 01 ~ Sub 13）
for sub_dir in sorted(os.listdir(src_root)):
    sub_path = os.path.join(src_root, sub_dir)
    if not os.path.isdir(sub_path) or not sub_dir.startswith("Sub"):
        continue  # 忽略非目录或非 Sub 文件夹

    for lux in target_lux_list:
        lux_src_path = os.path.join(sub_path, lux)
        if not os.path.isdir(lux_src_path):
            print(f"跳过不存在的目录：{lux_src_path}")
            continue

        dst_dir_name = f"{sub_dir}_{lux}"
        dst_path = os.path.join(dst_root, dst_dir_name)

        # 拷贝整个目录
        # shutil.copytree(lux_src_path, dst_path)
        print(f"✅ 拷贝完成: {lux_src_path} -> {dst_path}")


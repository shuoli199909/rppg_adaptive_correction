
import os
import shutil
import glob
import re

# 原始 GT 文件路径

src_dir = "/home/robbie/tyf_file/rppg_index_code_data/data_gt/LGI-PPGI/gt"
# 目标根目录
dst_root = "/home/robbie/tyf_data/datasets/LGI-PPGI_WE"

import os
import shutil

def copy_files_to_named_subfolders(src_dir, dst_root):
    # 确保目标根目录存在
    os.makedirs(dst_root, exist_ok=True)

    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        if os.path.isfile(src_file):
            # 去除扩展名获取文件名
            name_without_ext = os.path.splitext(filename)[0]

            # 创建对应的子文件夹
            dst_subdir = os.path.join(dst_root, name_without_ext)
            os.makedirs(dst_subdir, exist_ok=True)

            # 复制文件到子文件夹
            dst_file = os.path.join(dst_subdir, filename)
            shutil.copy2(src_file, dst_file)
            print(f"已复制: {src_file} -> {dst_file}")


# copy_files_to_named_subfolders(src_dir, dst_root)

import os

def add_prefix_to_csv_files(dst_root, prefix="mygt_"):
    for subdir in os.listdir(dst_root):
        subdir_path = os.path.join(dst_root, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(".csv") and not filename.startswith(prefix):
                    old_path = os.path.join(subdir_path, filename)
                    new_filename = prefix + filename
                    new_path = os.path.join(subdir_path, new_filename)
                    os.rename(old_path, new_path)
                    print(f"已重命名: {old_path} -> {new_path}")

# add_prefix_to_csv_files(dst_root)

import os
import shutil


def categorize_subfolders(dst_root):
    categories = ["gym", "resting", "talk", "rotation"]

    for subfolder in os.listdir(dst_root):
        subfolder_path = os.path.join(dst_root, subfolder)

        # 跳过不是目录的项
        if not os.path.isdir(subfolder_path):
            continue

        # 避免将分类目录自身再分类
        if subfolder.lower() in categories:
            continue

        matched = False
        for category in categories:
            if category in subfolder.lower():  # 忽略大小写匹配
                target_dir = os.path.join(dst_root, category)
                os.makedirs(target_dir, exist_ok=True)

                dest_path = os.path.join(target_dir, subfolder)
                shutil.move(subfolder_path, dest_path)
                print(f"已移动: {subfolder_path} -> {dest_path}")
                matched = True
                break

        if not matched:
            print(f"未匹配任何类别: {subfolder}")


categorize_subfolders(dst_root)

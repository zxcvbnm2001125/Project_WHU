import os
import glob
import random

# ==== 设置路径 ====
voc_root = 'VOCdevkit/VOC2012'
segmentation_class_dir = os.path.join(voc_root, 'SegmentationClass')  # mask 文件所在目录
image_sets_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')  # 保存 train.txt, val.txt, test.txt 的目录

# 创建必要的文件夹
os.makedirs(image_sets_dir, exist_ok=True)

# ==== 获取 SegmentationClass 目录下的所有 mask 文件 ====
mask_files = glob.glob(os.path.join(segmentation_class_dir, '*.png'))  # 获取所有 mask 文件

# 划分比例
train_ratio = 0.7  # 70% 作为训练集
val_ratio = 0.2  # 20% 作为验证集
test_ratio = 0.1  # 10% 作为测试集

# 随机打乱文件顺序
random.shuffle(mask_files)

# 划分数据集
train_files = []
val_files = []
test_files = []

# 根据比例划分训练集、验证集和测试集
train_size = int(len(mask_files) * train_ratio)
val_size = int(len(mask_files) * val_ratio)

train_files = mask_files[:train_size]
val_files = mask_files[train_size:train_size + val_size]
test_files = mask_files[train_size + val_size:]

# 生成 train.txt、val.txt 和 test.txt
with open(os.path.join(image_sets_dir, 'train.txt'), 'w') as f_train, \
        open(os.path.join(image_sets_dir, 'val.txt'), 'w') as f_val, \
        open(os.path.join(image_sets_dir, 'test.txt'), 'w') as f_test:
    # 写入训练集文件列表
    for mask_file in train_files:
        file_name = os.path.splitext(os.path.basename(mask_file))[0]  # 获取文件名（不包括扩展名）
        f_train.write(f'{file_name}\n')

    # 写入验证集文件列表
    for mask_file in val_files:
        file_name = os.path.splitext(os.path.basename(mask_file))[0]
        f_val.write(f'{file_name}\n')

    # 写入测试集文件列表
    for mask_file in test_files:
        file_name = os.path.splitext(os.path.basename(mask_file))[0]
        f_test.write(f'{file_name}\n')

    print("[✓] train.txt, val.txt 和 test.txt 已生成")

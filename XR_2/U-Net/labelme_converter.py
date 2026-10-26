import os
import glob
import json
import numpy as np
from PIL import Image
from labelme import utils
import random


def get_voc_palette(num_cls=256):
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


# ==== VOC 类别写在脚本里 ====
label_names = [
    '_background_',
    'chicken',
    'cat',
    'dog',
]
label_name_to_value = {name: idx for idx, name in enumerate(label_names)}

# ==== 设置路径 ====
voc_root = 'VOCdevkit/VOC2012'
json_dir = os.path.join(voc_root, 'JPEGImages')  # 存放 JSON 标注文件
output_dir = os.path.join(voc_root, 'SegmentationClass')  # 输出 mask 文件的目录
image_sets_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')

# 创建必要的文件夹
os.makedirs(output_dir, exist_ok=True)
os.makedirs(image_sets_dir, exist_ok=True)

# ==== 调色板 ====
palette = get_voc_palette(256)

# ==== 处理所有 JSON ====
json_files = glob.glob(os.path.join(json_dir, '*.json'))  # 获取 JSON 文件路径

# 划分比例
train_ratio = 0.7  # 70% 作为训练集
val_ratio = 0.2  # 20% 作为验证集
test_ratio = 0.1  # 10% 作为测试集

# 随机打乱文件顺序
random.shuffle(json_files)

# 划分数据集
train_files = []
val_files = []
test_files = []

# 根据比例划分训练集、验证集和测试集
train_size = int(len(json_files) * train_ratio)
val_size = int(len(json_files) * val_ratio)

train_files = json_files[:train_size]
val_files = json_files[train_size:train_size + val_size]
test_files = json_files[train_size + val_size:]

# 生成 train.txt、val.txt 和 test.txt
with open(os.path.join(image_sets_dir, 'train.txt'), 'w') as f_train, \
        open(os.path.join(image_sets_dir, 'val.txt'), 'w') as f_val, \
        open(os.path.join(image_sets_dir, 'test.txt'), 'w') as f_test:
    # 写入训练集文件列表
    for json_file in train_files:
        file_name = os.path.splitext(os.path.basename(json_file))[0]
        f_train.write(f'{file_name}\n')

    # 写入验证集文件列表
    for json_file in val_files:
        file_name = os.path.splitext(os.path.basename(json_file))[0]
        f_val.write(f'{file_name}\n')

    # 写入测试集文件列表
    for json_file in test_files:
        file_name = os.path.splitext(os.path.basename(json_file))[0]
        f_test.write(f'{file_name}\n')

    print("[✓] train.txt, val.txt 和 test.txt 已生成")

# ==== 继续处理每个 JSON 文件并生成 mask ====
for json_file in json_files:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # 如果没有 imageData 字段，则从图像文件中读取数据并转换为 base64
        if data.get('imageData') is None:
            image_path = os.path.join(json_dir, data['imagePath'])
            with open(image_path, 'rb') as img_f:
                image_data = utils.img_to_b64(img_f.read()).decode('utf-8')
            data['imageData'] = image_data

        # 使用 labelme 的工具将 imageData 转换为图像数组
        img = utils.img_b64_to_arr(data['imageData'])

        # 使用统一类别映射（即从标签名称映射到像素值）
        label_img, _ = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        # 将生成的标签图像保存为 PNG 格式
        label_pil = Image.fromarray(label_img.astype(np.uint8), mode='P')
        label_pil.putpalette(palette)

        # 输出路径
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        out_path = os.path.join(output_dir, f'{base_name}.png')
        label_pil.save(out_path)

        print(f"[✓] Converted: {json_file} -> {out_path}")

    except Exception as e:
        print(f"[✘] Error processing {json_file}: {e}")

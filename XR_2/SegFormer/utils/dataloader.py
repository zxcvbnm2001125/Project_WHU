import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import cvtColor, preprocess_input


# 支持的影像 / 标签扩展名（WHU_0 为 .tif）
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class UnetDataset(Dataset):
    def __init__(self, data_path, input_shape, num_classes, augmentation=True, split: str = "train"):
        """
        语义分割数据集（WHU_0 目录式划分）。

        参数：
        - data_path:    数据集根目录（如 WHU_0）
        - input_shape:  网络输入尺寸 [h, w]
        - num_classes:  类别数（含背景；由 --num-classes 加 1 得到）
        - augmentation: 是否启用训练期数据增强
        - split:        划分名称，"train" 或 "val"，对应 <data_path>/<split>/ 目录

        目录结构（"lable" 为数据集原始拼写，同时兼容规范的 "label"）：
            <data_path>/<split>/image/*.tif   影像（RGB 三通道）
            <data_path>/<split>/lable/*.tif   标签（背景 0 / 建筑物 1）
        影像与标签按"文件名主干"自动配对，无需事先生成 train.txt 之类的划分文件。

        增强固定为 RandomResizedCrop + 翻转 + 90° 旋转 + 轻颜色抖动，全程不做填充，
        因此不会产生"灰色区域 = 类别 0"这类假标签；归一化固定 ImageNet 均值方差，
        与 MiT 主干预训练一致。
        """
        self.image_dir = os.path.join(data_path, split, "image")
        # 兼容数据集原始的 "lable" 拼写与规范的 "label" 拼写
        self.mask_dir = os.path.join(data_path, split, "lable")
        if not os.path.isdir(self.mask_dir):
            alt_dir = os.path.join(data_path, split, "label")
            if os.path.isdir(alt_dir):
                self.mask_dir = alt_dir

        assert os.path.isdir(self.image_dir), f"image dir '{self.image_dir}' does not exist."
        assert os.path.isdir(self.mask_dir), f"mask dir '{self.mask_dir}' does not exist."

        # 收集影像文件，并按文件名主干配对标签
        names = sorted(n for n in os.listdir(self.image_dir) if n.lower().endswith(IMG_EXTS))
        self.samples = []
        for name in names:
            stem = os.path.splitext(name)[0]
            mask_name = self._match_mask(stem)
            if mask_name is None:
                print(f"warning: no mask found for '{name}', skipped.")
                continue
            self.samples.append((os.path.join(self.image_dir, name),
                                 os.path.join(self.mask_dir, mask_name)))
        assert len(self.samples) > 0, f"no matched image/mask pairs found in '{self.image_dir}'."

        # 初始化其他参数
        self.length = len(self.samples)  # 数据集的长度
        self.input_shape = input_shape  # 输入图像的形状（宽和高）
        self.num_classes = num_classes  # 类别数目
        self.augmentation = augmentation  # 是否在训练阶段，用来控制是否使用数据增强
        self.data_path = data_path  # 数据集路径

    def _match_mask(self, stem: str):
        """按文件名主干在标签目录中查找同名标签文件，未找到返回 None。"""
        for ext in IMG_EXTS:
            candidate = stem + ext
            if os.path.isfile(os.path.join(self.mask_dir, candidate)):
                return candidate
        return None

    def _load_mask_array(self, mask_path: str) -> np.ndarray:
        """
        读取标签图并统一为单通道 uint8 的类别索引数组。

        - num_classes <= 2（二分类，如 WHU 建筑）：
              mode='1' / 0-255 / 0-1 三种存法都归一成 0/1；
        - num_classes  > 2（多类，如 LoveDA 7 类）：
              **原样保留索引**，只统一 dtype。这里绝不能做 (mask > 0) 的二值化，
              否则 6 个地物类别会被压成一个"前景"，模型只能学会"背景 vs 非背景"，
              预测结果自然只剩两类，而且 mIoU 会因其余类 union=0 被排除而虚高。
              越界值（如 255）留给 __getitem__ 统一置为忽略类。
        """
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:                 # 偶发的三通道标签，取第一通道
            mask = mask[..., 0]

        if self.num_classes <= 2:
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            elif mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)
        return mask.astype(np.uint8)

    def class_pixel_counts(self, indices=None) -> np.ndarray:
        """
        统计指定样本范围内每个类别的像素总数，用于计算类别权重。

        Args:
            indices (iterable, optional): 参与统计的样本下标；None 表示整个子集。

        Returns:
            np.ndarray: 长度为 num_classes 的像素计数（int64）。
        """
        if indices is None:
            indices = range(len(self.samples))

        counts = np.zeros(self.num_classes, dtype=np.int64)
        for i in indices:
            flat = self._load_mask_array(self.samples[i][1]).reshape(-1)
            # 只统计落在有效类别范围内的像素（超范围的值视为忽略）
            flat = flat[flat < self.num_classes]
            if flat.size == 0:
                continue
            counts += np.bincount(flat, minlength=self.num_classes)[:self.num_classes]
        return counts

    def observed_classes(self, indices=None, max_samples=50):
        """
        抽样统计标签中实际出现过的类别值，供训练开始前做数据自检。

        用途：多类任务如果标签被错误地二值化（历史上出现过这个 bug），
        抽样只会看到 {0,1}，而模型仍按 N 类训练，指标会虚高且预测只有两类。
        这里把观测结果打出来，能第一时间发现。

        Args:
            indices:     参与抽样的样本下标；None 表示整份数据
            max_samples: 最多看多少张（每张都要解码 1024 图，给个小上限）
        Returns:
            list[int]: 出现过的类别值（升序）
        """
        if indices is None:
            indices = range(len(self.samples))
        indices = list(indices)[:max_samples]

        seen = set()
        for i in indices:
            seen |= set(np.unique(self._load_mask_array(self.samples[i][1])).tolist())
        return sorted(int(v) for v in seen)

    def __len__(self):
        # 返回数据集的大小
        return self.length

    def __getitem__(self, index):
        # 读取单个样本
        image_path, mask_path = self.samples[index]

        # 读取影像（WHU_0 为 RGB 三通道 tif）
        jpg = Image.open(image_path)

        # 读取标签，统一为单通道 0/1 的 'L' 图
        png = Image.fromarray(self._load_mask_array(mask_path), mode="L")

        # 如果是训练阶段，进行随机数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.augmentation)

        # 图像预处理：ImageNet 归一化后转成 CHW
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64), "imagenet"), [2, 0, 1])
        # 标签转换成numpy数组
        png = np.array(png)

        # 将标签值大于等于类别数的部分设置为类别数（即忽略区域，如标注时的 255）
        png[png >= self.num_classes] = self.num_classes

        # 将标签转换为one-hot编码，通道数为 num_classes + 1（最后一位是忽略类）
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]

        # 重塑标签为目标形状
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 返回图像、标签以及one-hot编码的标签
        return jpg, png, seg_labels

    # 生成随机数的函数
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """
        对图像和标签做同步的几何 / 颜色增强。

        参数：
        - image:       PIL 图像
        - label:       PIL 标签图
        - input_shape: 目标尺寸 [h, w]
        - jitter/hue/sat/val: 仅 v1 使用
        - random:      False 表示验证/推理阶段，只做等比缩放 + 居中粘贴（保持原逻辑不变）
        """
        # 将图像转为RGB格式
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))

        iw, ih = image.size  # 获取图像的宽和高
        h, w = input_shape  # 获取目标图像的高和宽

        if not random:
            # 验证 / 推理阶段：等比缩放 + 居中粘贴，逻辑保持原样
            iw, ih = image.size
            scale = min(w / iw, h / ih)  # 计算缩放比例
            nw = int(iw * scale)  # 根据比例计算缩放后的宽
            nh = int(ih * scale)  # 根据比例计算缩放后的高

            # 缩放图像并进行中心裁剪
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))  # 创建一个灰色背景的图像
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将缩放后的图像粘贴到目标图像中

            # 缩放标签图像并进行中心裁剪
            label = label.resize((nw, nh), Image.NEAREST)   # 标签使用最近邻插值
            new_label = Image.new('L', [w, h], (0))  # 创建一个空白标签图像
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))  # 将缩放后的标签图像粘贴到目标图像中
            return new_image, new_label

        # 训练阶段：RandomResizedCrop + 翻转 + 90°旋转 + 轻颜色抖动（全程不填充）
        return self._get_random_data_v2(image, label, input_shape)

    # ---------------------------------------------------------------- #
    #   v1：原始增强实现（保留用于对照 / 复现历史结果）
    #       注意：会把灰色填充区域标注为类别 0，属于已知缺陷
    # ---------------------------------------------------------------- #
    def _get_random_data_v2(self, image, label, input_shape,
                            scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                            hue=0.05, sat=0.3, val=0.2):
        h, w = input_shape
        iw, ih = image.size

        # 1) RandomResizedCrop
        image, label = self._random_resized_crop(image, label, (w, h), iw, ih, scale, ratio)

        # 2~4) 翻转 + 90° 旋转 + 颜色扰动
        return self._flip_rot_color(image, label, hue, sat, val)

    # ---------------------------------------------------------------- #
    #   v3：整图增强（不裁剪、不缩放）
    #       用于"一景 1024 直接进网络"的训练方式：保留完整视野，
    #       只做翻转、90° 旋转与温和颜色扰动，不产生任何子窗口。
    #       与 v2 的差别仅在于去掉了 RandomResizedCrop。
    # ---------------------------------------------------------------- #
    def _flip_rot_color(self, image, label, hue=0.05, sat=0.3, val=0.2):
        """翻转 + 90° 旋转 + 温和 HSV 扰动（v2 / v3 共用）。"""
        # 随机水平 / 垂直翻转
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)

        image_data = np.array(image, np.uint8)
        label_data = np.array(label, np.uint8)

        # 随机 90° 旋转（k 取 1/2/3，k=0 保持不变）
        # 注意：非正方形输入旋转会交换 H/W，与网络输入尺寸冲突，故只在正方形时启用
        if image_data.shape[0] == image_data.shape[1]:
            k = np.random.randint(0, 4)
            if k > 0:
                image_data = np.ascontiguousarray(np.rot90(image_data, k))
                label_data = np.ascontiguousarray(np.rot90(label_data, k))

        # 温和的颜色扰动
        image_data = self._color_jitter(image_data, hue, sat, val)
        return image_data, label_data

    def _random_resized_crop(self, image, label, size, iw, ih, scale, ratio, max_try=10):
        """
        从原图中随机裁剪一块区域，再缩放到目标尺寸。

        参数：
        - image/label: PIL 图像与标签
        - size:        目标尺寸 (w, h)
        - iw/ih:       原图宽高
        - scale:       裁剪面积占原图面积的比例范围
        - ratio:       裁剪框的宽高比范围
        返回：
        - 缩放后的 (image, label)
        """
        area = iw * ih
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for _ in range(max_try):
            target_area = area * self.rand(*scale)
            ar = math.exp(self.rand(*log_ratio))
            cw = int(round(math.sqrt(target_area * ar)))
            ch = int(round(math.sqrt(target_area / ar)))
            # 裁剪框必须合法
            if 0 < cw <= iw and 0 < ch <= ih:
                x0 = np.random.randint(0, iw - cw + 1)
                y0 = np.random.randint(0, ih - ch + 1)
                image_c = image.crop((x0, y0, x0 + cw, y0 + ch))
                label_c = label.crop((x0, y0, x0 + cw, y0 + ch))
                return (image_c.resize(size, Image.BICUBIC),
                        label_c.resize(size, Image.NEAREST))

        # 兜底：裁剪框始终非法时直接整图缩放（同样不产生填充）
        return image.resize(size, Image.BICUBIC), label.resize(size, Image.NEAREST)

    def _color_jitter(self, image_data, hue=0.05, sat=0.3, val=0.2):
        """
        在 HSV 空间做温和的色调 / 饱和度 / 亮度随机扰动。

        参数：
        - image_data: HWC 的 uint8 RGB 数组
        返回：
        - 扰动后的 HWC uint8 RGB 数组
        """
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        h_ch, s_ch, v_ch = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)          # 色调按 180 取模（OpenCV 的 H 通道范围）
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)   # 饱和度裁剪到 [0, 255]
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)   # 亮度裁剪到 [0, 255]

        out = cv2.merge((cv2.LUT(h_ch, lut_hue), cv2.LUT(s_ch, lut_sat), cv2.LUT(v_ch, lut_val)))
        return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)


# DataLoader中collate_fn使用
def unet_dataset_collate(batch):
    # 初始化三个列表，用于存储每个批次中的图像、标签和one-hot编码标签
    images = []        # 用来存储图像数据
    pngs = []          # 用来存储原始标签（通常是类别标签）
    seg_labels = []    # 用来存储one-hot编码的标签

    # 遍历当前批次中的每个样本（img, png, labels）
    for img, png, labels in batch:
        images.append(img)            # 将图像添加到images列表中
        pngs.append(png)              # 将原始标签添加到pngs列表中
        seg_labels.append(labels)     # 将one-hot标签添加到seg_labels列表中

    # 将列表转换为NumPy数组，然后转换为torch张量
    # images的张量需要是float类型，通常用于输入图像
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # pngs的张量需要是long类型，通常用于标签索引
    pngs = torch.from_numpy(np.array(pngs)).long()
    # seg_labels的张量需要是float类型，通常用于标签的one-hot编码
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)

    # 返回三个张量，分别对应图像、原始标签和one-hot标签
    return images, pngs, seg_labels

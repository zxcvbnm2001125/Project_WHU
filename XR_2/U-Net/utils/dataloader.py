import torch
from utils.utils import cvtColor, preprocess_input
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2

def load_mask(path):
    """读取 WUH 标签，统一返回 mode='L'、取值 {0,1} 的 PIL 图像。

    WUH 的标签 tif 全部是 PIL `mode='1'`（双色位图），np.array 出来是 bool：
      - 直接喂给 np.eye(n)[mask] 会被当成布尔掩码，抛 IndexError；
      - 用 Image.convert('L') 会把 True 映成 255，标签值域被破坏。
    因此这里先转 uint8 再回建 'L' 图，保证下游拿到的永远是 0/1。
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:          # 万一被存成 RGB
        arr = arr[..., 0]
    return Image.fromarray(arr.astype(np.uint8), mode="L")


class UnetDataset(Dataset):
    # 影像与标签支持的扩展名（WUH 为 .tif，按 stem 配对）
    IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    LBL_EXTS = (".tif", ".tiff", ".png", ".jpg", ".bmp")

    def __init__(self, data_path, input_shape, num_classes, augmentation=True, split: str = "train"):
        """直接读 WUH 原生目录结构，不依赖 VOC 视图。

        <data_path>/<split>/image/*.tif   影像
        <data_path>/<split>/label/*.tif   标签（背景 0 / 建筑 1）
        """
        self.data_path = data_path
        self.split = split
        self.image_dir = os.path.join(data_path, split, "image")
        # 兼容数据集原始的 "lable" 拼写
        self.mask_dir = os.path.join(data_path, split, "label")
        if not os.path.isdir(self.mask_dir):
            alt = os.path.join(data_path, split, "lable")
            if os.path.isdir(alt):
                self.mask_dir = alt

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"影像目录不存在：{self.image_dir}")

        names = sorted(n for n in os.listdir(self.image_dir) if n.lower().endswith(self.IMG_EXTS))
        if not names:
            raise FileNotFoundError(f"{self.image_dir} 下没有匹配 {self.IMG_EXTS} 的文件")

        self.samples = []
        for n in names:
            stem = os.path.splitext(n)[0]
            self.samples.append((stem,
                                 os.path.join(self.image_dir, n),
                                 self._find_stem(self.mask_dir, stem, self.LBL_EXTS)))

        # 初始化其他参数
        self.length = len(self.samples)  # 数据集的长度
        self.input_shape = input_shape  # 输入图像的形状（宽和高）
        self.num_classes = num_classes  # 类别数目
        self.augmentation = augmentation  # 是否在训练阶段，用来控制是否使用数据增强

    @staticmethod
    def _find_stem(folder, stem, exts):
        """按文件名主干查找实际文件，兼容 .tif / .png / .jpg 混用。"""
        for ext in exts:
            p = os.path.join(folder, stem + ext)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"在 {folder} 下找不到 {stem} 的 {'/'.join(exts)} 文件")

    def __len__(self):
        # 返回数据集的大小
        return self.length

    def __getitem__(self, index):
        # 读取单个样本（影像与标签在 __init__ 里已按 stem 配对）
        name, img_path, lbl_path = self.samples[index]

        # 读取影像
        jpg = Image.open(img_path)
        # 读取标签：统一转成 mode='L'、取值 {0,1}
        png = load_mask(lbl_path)

        # 如果是训练阶段，进行随机数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.augmentation)

        # 图像预处理
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        # 标签转换成numpy数组
        png = np.array(png)
        if png.ndim == 3:
            png = png[..., 0]
        # 兜底：load_mask 之后理论上已是 uint8，这里再防一次 bool
        if png.dtype == np.bool_:
            png = png.astype(np.uint8)

        # 将标签值大于类别数的部分设置为类别数（忽略这些区域）
        png[png >= self.num_classes] = self.num_classes

        # 将标签转换为one-hot编码
        seg_labels = np.eye(self.num_classes + 1)[np.clip(png.reshape([-1]), 0, self.num_classes)]

        # 重塑标签为目标形状
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # 返回图像、标签以及one-hot编码的标签
        return jpg, png, seg_labels

    # 生成随机数的函数
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    # 对图像和标签进行随机数据增强的函数
    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # 将图像转为RGB格式
        image = cvtColor(image)
        # WHU 标签可能是 mode '1'（bool 数组），fromarray 前统一转 uint8
        label = Image.fromarray(np.array(label).astype(np.uint8))

        iw, ih = image.size  # 获取图像的宽和高
        h, w = input_shape  # 获取目标图像的高和宽

        if not random:
            # 如果不进行随机增强（例如在验证阶段）
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

        # 获取一个新的宽高比（通过调整宽和高的比例）
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)  # 随机缩放比例
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        # 随机翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # 在图像周围随机添加灰色边框
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))   # 将图像粘贴到新图像中
        new_label.paste(label, (dx, dy))   # 将标签粘贴到新标签中
        image = new_image
        label = new_label

        # 转换图像为数组
        image_data = np.array(image, np.uint8)


        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1  # 随机调整色调、饱和度和亮度
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))  # 转为HSV色域
        dtype = image_data.dtype  # 获取数据类型
        x = np.arange(0, 256, dtype=r.dtype)  # 获取颜色值范围
        lut_hue = ((x * r[0]) % 180).astype(dtype)  # 应用色调变换
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 应用饱和度变换
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 应用亮度变换

        # 使用查找表（LUT）应用变换
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)  # 转换回RGB色域

        return image_data, label  # 返回经过增强的图像和标签


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


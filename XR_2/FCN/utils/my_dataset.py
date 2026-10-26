import os

import numpy as np
import torch.utils.data as data
from PIL import Image

# WUH 子集为 .tif，按文件名主干配对影像与标签
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
MASK_EXTS = (".tif", ".tiff", ".png", ".jpg", ".bmp")


def find_by_stem(folder, stem, exts):
    for ext in exts:
        p = os.path.join(folder, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"在 {folder} 下找不到 {stem} 的 {'/'.join(exts)} 文件")


def load_mask(path):
    """读取 WUH 标签，统一返回 mode='L'、取值 {0,1} 的 PIL 图像。

    这里必须是 'L' 而不是保持原来的 mode='1'，原因是本工程的 RandomCrop 在图像
    短边小于 crop 尺寸时用 `pad_if_smaller(target, size, fill=255)` 填充标签，
    靠 CE 的 ignore_index=255 把填充区排除掉。mode='1' 下 255 会被截成 True，
    填充区就变成了前景，直接教坏模型。

    WUH 标签本身全是 mode='1'（np.array 为 bool），所以先转 uint8 再回建 'L'，
    保证取值为 0/1；注意不能用 Image.convert('L')，那会把 True 映成 255。
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return Image.fromarray(arr.astype(np.uint8), mode="L")


class WUHDataset(data.Dataset):
    """WUH 原生目录结构数据集。

        <root>/<split>/image/*.tif   影像（RGB 三通道）
        <root>/<split>/label/*.tif   标签（背景 0 / 建筑 1）

    split 取 "train" 或 "val"。不再依赖 VOC 视图（JPEGImages + SegmentationClass + txt）。
    """

    def __init__(self, root, transforms=None, split: str = "train"):
        super(WUHDataset, self).__init__()

        self.root = root
        self.split = split
        self.image_dir = os.path.join(root, split, "image")
        # 兼容数据集原始的 "lable" 拼写
        self.mask_dir = os.path.join(root, split, "label")
        if not os.path.isdir(self.mask_dir):
            alt = os.path.join(root, split, "lable")
            if os.path.isdir(alt):
                self.mask_dir = alt

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"影像目录不存在：{self.image_dir}")

        names = sorted(n for n in os.listdir(self.image_dir) if n.lower().endswith(IMG_EXTS))
        if not names:
            raise FileNotFoundError(f"{self.image_dir} 下没有匹配 {IMG_EXTS} 的文件")

        # 根据文件名主干配对影像与标签
        self.images = [os.path.join(self.image_dir, n) for n in names]
        self.masks = [find_by_stem(self.mask_dir, os.path.splitext(n)[0], MASK_EXTS) for n in names]

        assert (len(self.images) == len(self.masks))

        # 保存图像预处理的转换操作（如果有的话）
        self.transforms = transforms

    def __getitem__(self, index):
        """
        获取一个样本（图像和标签）

        Args:
            index (int): 索引

        Returns:
            tuple: (图像, 标签) 其中标签是图像的语义分割标注
        """
        # 打开图像文件并转换为RGB模式
        img = Image.open(self.images[index]).convert('RGB')

        # 打开对应的标签文件，并统一成 mode='L'、取值 {0,1}
        target = load_mask(self.masks[index])

        # 如果定义了数据预处理（例如数据增强），则应用它
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 返回图像和标签
        return img, target

    def __len__(self):
        # 返回数据集的大小（图像的数量）
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
                这个方法用于合并一个batch的样本
                这里使用了cat_list函数将图像和标签打包成一个批次

                Args:
                    batch (list): 从数据集中获取的样本列表，包含了图像和标签

                Returns:
                    tuple: (batched_images, batched_targets)，一个批次的图像和标签
        """
        images, targets = list(zip(*batch))  # 拆解批次中的图像和标签
        batched_imgs = cat_list(images, fill_value=0)  # 使用cat_list函数合并图像，填充值为0
        batched_targets = cat_list(targets, fill_value=255)  # 使用cat_list函数合并标签，填充值为255
        return batched_imgs, batched_targets



def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size  # 计算最终批次的形状，(batch_size, channel, height, width)

    # 创建一个新的空张量，填充为fill_value（默认0），大小为batch_shape
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)

    # 将每个图像的内容复制到批次张量中的相应位置，确保所有图像的尺寸都一致
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)

    # 将每个图像的内容复制到批次张量中的相应位置，确保所有图像的尺寸都一致
    return batched_imgs

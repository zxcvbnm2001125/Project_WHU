import os
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import torch.utils.data as data


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        cmap[i] = np.array([r, g, b])

    return (cmap / 255.0) if normalized else cmap


def load_mask(path):
    """读取 WUH 标签，统一返回 mode='L'、取值 {0,1} 的 PIL 图像。

    WUH 的标签 tif 全是 PIL `mode='1'`（双色位图），np.array 出来是 bool。
    原实现里 `if target.mode not in ("L", "P"): target = target.convert("L")`
    会把 True 映成 255，而本工程 CE 的 ignore_index 正是 255——
    结果全部前景像素被当成"忽略区"丢掉，训练信号直接归零。
    所以这里改为 bool → uint8 再回建 'L'，取值严格是 0/1。
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:          # 万一被存成 RGB
        arr = arr[..., 0]
    return Image.fromarray(arr.astype(np.uint8), mode="L")


class WUHSegmentation(data.Dataset):
    """WUH 原生目录结构语义分割数据集。

        <root>/<split>/image/*.tif   影像（RGB 三通道）
        <root>/<split>/label/*.tif   标签（背景 0 / 建筑 1）

    split 取 "train" 或 "val"。不再依赖 VOC 视图（JPEGImages + SegmentationClass + txt）。
    """

    cmap = voc_cmap()

    # 按需增删扩展名；顺序=优先级（越靠前越优先）
    IMG_EXTS: List[str] = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]
    MASK_EXTS: List[str] = [".tif", ".tiff", ".png", ".jpg", ".bmp"]

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform=None,
    ):
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform

        self.image_dir = os.path.join(self.root, image_set, "image")
        # 兼容数据集原始的 "lable" 拼写
        self.mask_dir = os.path.join(self.root, image_set, "label")
        if not os.path.isdir(self.mask_dir):
            alt = os.path.join(self.root, image_set, "lable")
            if os.path.isdir(alt):
                self.mask_dir = alt

        # --------- 检查目录 ---------
        if not os.path.isdir(self.image_dir):
            raise RuntimeError(
                "找不到 WUH 影像目录。\n"
                "期望结构：<root>/<split>/image/*.tif 与 <root>/<split>/label/*.tif\n"
                f"当前给定 root = {self.root}，split = {image_set}"
            )
        if not os.path.isdir(self.mask_dir):
            raise RuntimeError(f"找不到标签目录：{self.mask_dir}")

        # --------- 枚举影像，按 stem 配对标签（多后缀）---------
        names = sorted(n for n in os.listdir(self.image_dir) if n.lower().endswith(tuple(self.IMG_EXTS)))
        if not names:
            raise RuntimeError(f"{self.image_dir} 下没有匹配 {self.IMG_EXTS} 的文件")

        self.images = []
        self.masks = []
        missing_masks = []

        for n in names:
            stem = os.path.splitext(n)[0]
            mask_path = self._find_existing(self.mask_dir, stem, self.MASK_EXTS)
            if mask_path is None:
                missing_masks.append(stem)
                continue
            self.images.append(os.path.join(self.image_dir, n))
            self.masks.append(mask_path)

        if missing_masks:
            raise RuntimeError(
                f"WUHSegmentation: {len(missing_masks)} 个影像找不到对应标签，"
                f"例如 {missing_masks[:10]}\n  支持的标签后缀：{self.MASK_EXTS}"
            )

        assert len(self.images) == len(self.masks)

    @staticmethod
    def _find_existing(dir_path: str, stem: str, exts: List[str]) -> Optional[str]:
        """在 dir_path 下按优先级尝试 stem+ext，返回第一个存在的路径，否则 None。"""
        for ext in exts:
            p = os.path.join(dir_path, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Image.Image]:
        img = Image.open(self.images[index]).convert("RGB")
        # 标签统一成 mode='L'、取值 {0,1}
        target = load_mask(self.masks[index])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @classmethod
    def decode_target(cls, mask: np.ndarray) -> np.ndarray:
        return cls.cmap[mask]

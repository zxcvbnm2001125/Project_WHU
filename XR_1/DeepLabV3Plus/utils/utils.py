from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
    """
    将已 normalize 的 tensor 反归一化回“原始值域”（便于可视化）。
    原 normalize: (x - mean) / std
    反归一化: x = x_norm * std + mean
    这里通过再做一次 normalize，并把 mean/std 变换成：
      _mean = -mean/std, _std = 1/std
    来达到反归一化效果。
    """
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    """
    反归一化的可调用对象版本（callable），用法更方便：
      denorm = Denormalize(mean, std)
      img = denorm(img_tensor)
    同时兼容：
      - numpy.ndarray (C,H,W)
      - torch.Tensor
    """
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        # 如果输入是 numpy：用手写公式还原
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        # 如果输入是 torch tensor：复用 torchvision 的 normalize
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    """
    设置模型里所有 BatchNorm2d 的 momentum。
    DeepLab 系列训练常把 BN momentum 设小一点（例如 0.01），让统计更平滑。
    main.py 里就是用它：utils.set_bn_momentum(model.backbone, momentum=0.01)
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    """
    冻结 BN：把所有 BatchNorm2d 设为 eval()，停止更新 running_mean/var。
    常用于：batch size 很小导致 BN 不稳定的情况。
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    """
    创建文件夹（若不存在）。
    main.py 用它创建 checkpoints 目录。
    """
    if not os.path.exists(path):
        os.mkdir(path)

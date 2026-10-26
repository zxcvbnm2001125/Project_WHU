import random

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
def seed_everything(seed=11, deterministic=False):
    """
    固定随机种子。

    deterministic=False（默认）：只固定种子，cudnn 用 benchmark=True 自动选最快卷积算法，
        实测能快 10~20%；代价是同一份代码不同轮次的数值会有极小抖动，
        对分割任务的结论没有影响。
    deterministic=True：启用 cudnn 确定性模式（benchmark=False），
        可复现性最好但明显更慢，只在需要"逐位复现"时打开。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


# ---------------------------------------------------#
#   设置Dataloader的种子
# ---------------------------------------------------#
def worker_init_fn(rank, seed=11):
    """
    DataLoader 的 worker 初始化函数。

    注意：PyTorch 调用它时**只传一个参数**（worker 的 id），
    所以 seed 必须带默认值，否则一旦 --workers > 0 就会抛
    "worker_init_fn() missing 1 required positional argument: 'seed'"。
    （此前默认 workers=0 掩盖了这个错误。）
    """
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------#
#   归一化常量
#   ImageNet mean/std 是 MiT 预训练权重训练时使用的口径。
#   如果加载了 ImageNet 预训练的 MiT 主干，却沿用 U-Net 的 image/255，
#   输入分布与预训练时不一致，预训练带来的收益会被明显削弱。
#   因此本项目把归一化方式做成可选项，默认按"有预训练"处理。
# ---------------------------------------------------#
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)


def preprocess_input(image, norm="imagenet"):
    """
    把 uint8 的 HWC 图像转成网络输入。

    参数：
    - image: HWC 的 RGB 图像（uint8 或 float，取值 0~255）
    - norm:  归一化方式
             "imagenet"：先 /255 再按 ImageNet mean/std 标准化（SegFormer 默认，与 MiT 预训练一致）
             "unit"    ：只做 image / 255.0（姊妹项目 U-Net / FCN / DeepLabV3+ 的口径）
    返回：
    - float64 的 HWC 数组
    """
    image = np.asarray(image, dtype=np.float64) / 255.0
    if norm in (None, "unit", "none"):
        return image
    if norm == "imagenet":
        return (image - IMAGENET_MEAN) / IMAGENET_STD
    raise ValueError(f"unsupported norm: {norm!r}，可选 'imagenet' / 'unit'")


# ---------------------------------------------------#
#   由类别像素计数计算“反频率”类别权重
#   用途：WHU_0 中背景约占 76%、建筑约占 24%，属于类别不平衡，
#        均匀权重下模型容易停在“全预测背景”的局部最优。
#        这里按像素频率的反比给少数类更高权重，缓解该问题。
#   公式：w_c = N / (C * N_c)，再把背景（类别 0）归一为 1.0，
#        因此权重可直接读作“相对背景的重要性倍数”。
# ---------------------------------------------------#
def compute_class_weights(counts, max_weight=10.0):
    """
    参数：
    - counts:     长度为 C 的各类别像素计数（array-like）
    - max_weight: 权重上限，防止极稀有类把损失带偏
    返回：
    - np.ndarray(float32)，长度为 C 的类别权重，其中 w[0]（背景）恒为 1.0
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)                  # 防止除零 / 空类
    total = counts.sum()
    weights = total / (len(counts) * counts)          # 反频率
    weights = weights / weights[0]                    # 以背景为基准归一
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    return weights.astype(np.float32)


# ---------------------------------------------------#
#   校验网络输入尺寸
#   主干的累计下采样倍数为 32（conv1 /2 × maxpool /2 × layer2 /2 × layer3 /2 × layer4 /2），
#   因此输入边长必须是 32 的整数倍，否则解码器上采样后的尺寸会与跳跃连接特征对不上。
# ---------------------------------------------------#
def validate_input_size(size, factor=32, name="input_shape"):
    """
    参数：
    - size:   网络输入边长（正方形）
    - factor: 主干的累计下采样倍数，ResNet50 为 32
    - name:   报错时显示的参数名
    返回：
    - 校验通过的尺寸
    """
    if size % factor != 0:
        lower = size // factor * factor
        upper = (size // factor + 1) * factor
        raise ValueError(
            f"{name}={size} 必须是 {factor} 的整数倍，"
            f"距离最近的两个可用值是 {lower} 和 {upper}")
    return size


# ---------------------------------------------------#
#   按参数名与形状匹配进行“部分加载”
#   用途：
#     1) 加载预训练主干（如 VOC / ImageNet）时，分类头类别数不同也能加载；
#     2) 网架结构升级（新增 BN 层）后，仍可用旧权重初始化能对上的部分。
#   这样做失败也不会抛异常，只会提示哪些参数没被加载。
# ---------------------------------------------------#
def load_weights_partial(model, weights_path, map_location='cpu', verbose=True):
    """
    参数：
    - model:        目标模型
    - weights_path: 权重文件路径
    - map_location: torch.load 的设备映射
    - verbose:      是否打印加载统计
    返回：
    - 加载后的模型
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=map_location)

    # 兼容权重被外层字典包裹（如 {'state_dict': ...} / {'model': ...}）的情况
    if isinstance(pretrained_dict, dict):
        for key in ('state_dict', 'model', 'net'):
            if key in pretrained_dict and isinstance(pretrained_dict[key], dict):
                pretrained_dict = pretrained_dict[key]
                break

    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        # 参数名存在且形状一致才加载
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)

    model_dict.update(temp_dict)
    # 这里必须用 strict=False：未匹配的参数保留模型自身的初始化
    model.load_state_dict(model_dict, strict=False)

    if verbose:
        print(f"[权重加载] {weights_path}")
        print(f"[权重加载] 成功加载 {len(load_key)}/{len(pretrained_dict)} 个参数")
        if no_load_key:
            print(f"[权重加载] 未加载 {len(no_load_key)} 个参数（形状不匹配或不存在），"
                  f"示例: {no_load_key[:3]}")
    return model
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    """
    交叉熵损失（带类别权重、支持忽略标签）。

    参数：
    - inputs:      模型输出，形状 [n, c, h, w]
    - target:      标签（类别索引），形状 [n, ht, wt]
    - cls_weights: 每个类别的权重，用于缓解类别不平衡
    - num_classes: 视作“忽略”的标签值（ignore_index）

    说明：
        原实现里尺寸判断写成了 `if h != ht and w != wt:`，
        只有高和宽同时不等时才会插值；只要有一个维度不等就会出错，
        这里修正为 `or`。
    """
    n, c, h, w = inputs.size()  # 输入尺寸，n=batch_size，c=类别数
    nt, ht, wt = target.size()  # 标签尺寸

    # 尺寸不一致时插值对齐（注意：这里必须是 or，否则会漏掉“只有一个维度不等”的情况）
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 展平为 [n*h*w, c] 与 [n*h*w]
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # 计算交叉熵损失
    CE_loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    """
    Focal Loss：通过降低易分样本的权重，让模型更关注难分的样本。

    参数：
    - inputs:      模型输出，形状 [n, c, h, w]
    - target:      标签（类别索引），形状 [n, ht, wt]
    - cls_weights: 每个类别的权重
    - num_classes: 忽略标签值（ignore_index）
    - alpha:       正负样本权重平衡因子
    - gamma:       聚焦参数，越大越关注难样本

    说明：
        原实现用 `pt = exp(-CE)`，但在带 cls_weights 时 -CE 已不等于 log(p)，
        概率退化成了加权量。这里改为先做 softmax 取真实类概率，再进行聚焦加权，
        数值上更稳定也更符合 Focal Loss 的定义。
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # 尺寸不一致时插值对齐（同样修正为 or）
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 逐像素取真实类别的预测概率 pt
    prob = torch.softmax(inputs, dim=1)                                   # [n, c, h, w]
    prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, c)              # [n*h*w, c]
    temp_target = target.view(-1)                                         # [n*h*w]

    # 有效像素掩码：忽略标签不参与损失
    valid_mask = temp_target != num_classes
    # 越界保护：把忽略标签临时替换为 0，避免 gather 索引越界
    safe_target = temp_target.clone()
    safe_target[~valid_mask] = 0

    pt = prob.gather(1, safe_target.view(-1, 1)).view(-1)

    # 逐像素的类别权重
    if cls_weights is not None:
        pixel_weight = cls_weights[safe_target]
    else:
        pixel_weight = torch.ones_like(pt)

    # Focal 权重：(1 - pt)^gamma，难样本（pt 小）权重更大
    focal_weight = (1.0 - pt).clamp(min=1e-6) ** gamma
    loss = -focal_weight * pixel_weight * torch.log(pt.clamp(min=1e-6))

    # alpha 用于平衡正负样本
    if alpha is not None:
        loss = loss * alpha

    # 只在有效像素上求平均
    loss = loss[valid_mask]
    if loss.numel() == 0:
        return inputs.sum() * 0.0
    return loss.mean()


def Dice_loss(inputs, target, beta=1, smooth=1e-5, ignore_index=None):
    """
    Dice Loss：衡量预测与真实标签的重叠程度，对类别不平衡鲁棒。

    参数：
    - inputs:       模型输出，形状 [n, c, h, w]
    - target:       one-hot 标签，形状 [n, ht, wt, ct]，其中 ct = c + 1（最后一维是忽略类）
    - beta:         beta^2 控制对假负例的惩罚力度
    - smooth:       平滑项，防止除零
    - ignore_index: 忽略类在最后一维的下标，默认取 ct-1

    说明：
        原实现中 `fp = sum(temp_inputs) - tp` 把忽略像素上的预测概率也计入了 FP，
        ignore 区域被错误地惩罚，会持续压低 Dice 分数。
        这里先用有效像素掩码把忽略像素的概率置零，再统计 tp/fp/fn。
    """
    n, c, h, w = inputs.size()          # 模型输出：c 个真实类别
    nt, ht, wt, ct = target.size()      # one-hot 标签：ct 个通道（多出的 1 个是忽略类）

    # 尺寸不一致时插值对齐
    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 默认把 one-hot 的最后一维当作忽略类
    if ignore_index is None:
        ignore_index = ct - 1

    # 预测概率：[n, h*w, c]
    temp_inputs = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), dim=-1)
    # 标签：[n, h*w, ct]
    temp_target = target.view(n, -1, ct)

    # 有效像素掩码：非忽略类即为有效，形状 [n, h*w, 1]
    valid_mask = 1.0 - temp_target[..., ignore_index].unsqueeze(-1)
    # 只保留与模型输出对齐的前 c 个类别
    temp_target = temp_target[..., :c]

    # 忽略像素的预测概率归零，避免污染 fp
    temp_inputs = temp_inputs * valid_mask
    temp_target = temp_target * valid_mask

    # tp：真实类别与预测类别的交集
    tp = torch.sum(temp_target * temp_inputs, dim=[0, 1])
    # fp：预测为该类但不是真实类别
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    # fn：真实为该类但没被预测出来
    fn = torch.sum(temp_target, dim=[0, 1]) - tp

    # Dice 系数，beta 控制对假负例的惩罚程度
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss


def weights_init(net, init_type='kaiming', init_gain=0.02, skip_prefixes=('resnet',)):
    """
    对网络做权重初始化。

    参数：
    - net:           目标网络
    - init_type:     初始化方式，可选 normal / xavier / kaiming / orthogonal
    - init_gain:     初始化增益（kaiming / orthogonal 下由激活函数自适应的不使用该值）
    - skip_prefixes: 需要跳过的顶层模块名前缀

    说明：
        原实现对整网（含 ResNet 主干）统一做了 N(0, 0.02) 初始化，
        会把 resnet_backbone.py 里为 ReLU 精心设计的 He 初始化直接覆盖掉，
        导致“不加载预训练权重时”收敛极其缓慢。
        这里默认跳过 resnet 主干，只初始化解码器与分割头。
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            # BN 权重初始化为 1，偏置初始化为 0，保证初始阶段近似恒等映射
            torch.nn.init.constant_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    # 按模块名遍历，跳过主干网络，避免破坏其原有的 He 初始化
    for name, module in net.named_modules():
        if skip_prefixes and name.split('.')[0] in skip_prefixes:
            continue
        init_func(module)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    """
    构造学习率调度函数。

    参数：
    - lr_decay_type:  衰减类型，"cos" 使用带 warmup 的余弦衰减，其余使用阶梯衰减
    - lr:             初始学习率
    - min_lr:         最小学习率
    - total_iters:    总迭代次数（本项目按 epoch 计）
    - step_num:       阶梯衰减的段数

    返回：
    - 一个接受当前迭代次数、返回学习率的函数
    """
    # 根据学习率衰减类型（lr_decay_type）选择不同的学习率调度方式
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        """
        YOLOX 风格的“warmup + 余弦衰减”学习率调度。

        参数：
        - lr:                基础学习率
        - min_lr:            最小学习率
        - total_iters:       总迭代次数
        - warmup_total_iters: warmup 阶段的迭代次数
        - warmup_lr_start:   warmup 起始学习率
        - no_aug_iter:       末尾保持最小学习率的迭代次数
        - iters:             当前迭代次数

        返回：
        - 当前学习率
        """
        if iters <= warmup_total_iters:
            # warmup 阶段：二次曲线从 warmup_lr_start 升到 lr
            lr = (lr - warmup_lr_start) * pow(iters / float(max(warmup_total_iters, 1)), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            # 末尾阶段：保持最小学习率
            lr = min_lr
        else:
            # 余弦衰减阶段
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / max(total_iters - warmup_total_iters - no_aug_iter, 1))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        """
        阶梯衰减的学习率调度。

        参数：
        - lr:         基础学习率
        - decay_rate: 每段衰减比例
        - step_size:  每段的迭代数
        - iters:      当前迭代次数

        返回：
        - 当前学习率
        """
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    # 根据学习率衰减类型选择合适的调度函数
    if lr_decay_type == "cos":
        # warmup 段：5% 的总轮次，且不超过 3 个 epoch
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        # 末尾保持最小学习率的段：5% 的总轮次，且不超过 15 个 epoch
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        # 阶梯衰减
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch, lr_ratios=None):
    """
    为优化器设置当前 epoch 的学习率。

    参数：
    - optimizer:         优化器
    - lr_scheduler_func: 学习率调度函数
    - epoch:             当前 epoch
    - lr_ratios:         各组学习率的相对倍率（用于主干/解码器差异化学习率），
                         例如 [0.1, 1.0] 表示主干学习率是基准的 1/10；
                         传 None 时改用每个参数组自带的 'lr_scale'
                         （SegFormer 的 layer-wise lr decay 用的就是这条路）

    说明：
        原实现把所有参数组的学习率设成同一个值，无法给预训练主干更小的学习率。
        这里增加 lr_ratios 支持，同时保持默认行为向后兼容。
    """
    # 计算本轮的基准学习率
    lr = lr_scheduler_func(epoch)

    for i, param_group in enumerate(optimizer.param_groups):
        if lr_ratios is not None and i < len(lr_ratios):
            # 显式给了倍率表，按位置取
            ratio = lr_ratios[i]
        else:
            # 没给倍率表时，用参数组自带的 lr_scale（默认 1.0）
            ratio = param_group.get('lr_scale', 1.0)
        param_group['lr'] = lr * ratio

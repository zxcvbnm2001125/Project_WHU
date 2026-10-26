import time

import numpy as np
import torch

from model.unet_training import CE_Loss, Dice_loss, Focal_Loss
from utils.utils import get_lr

# ------------------------------------------------------------------ #
#   混合精度（AMP）接口兼容层
#   PyTorch 2.4 起推荐使用 torch.amp.* 接口（不再有弃用警告），
#   更早的版本只有 torch.cuda.amp.*，这里做统一适配。
# ------------------------------------------------------------------ #
try:
    from torch.amp import autocast as _autocast_new
    from torch.amp import GradScaler

    def autocast(enabled=True, dtype=None):
        """
        返回 CUDA 上的 autocast 上下文管理器。

        dtype 为 None 时沿用 PyTorch 默认（float16）；
        传入 torch.bfloat16 即可切到 bf16 —— Transformer 类模型（SegFormer）
        在 fp16 下 LayerNorm/注意力容易 inf，bf16 不需要 loss scaling，更省心。
        """
        kwargs = {"device_type": "cuda", "enabled": enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return _autocast_new(**kwargs)
except ImportError:
    # 老版本 PyTorch 回退（不支持 bf16 autocast，dtype 参数被忽略）
    from torch.cuda.amp import autocast as _autocast_old
    from torch.cuda.amp import GradScaler

    def autocast(enabled=True, dtype=None):
        return _autocast_old(enabled=enabled)


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


# ------------------------------------------------------------------ #
#   单张图片级别的指标（保留原有实现，供外部脚本按需调用）
# ------------------------------------------------------------------ #
def pixel_accuracy(output, target):
    """像素准确率 PA = 预测正确的像素数 / 总像素数"""
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).float()
        correct_pixels = correct.sum().item()
        total_pixels = target.numel()
        return correct_pixels / total_pixels


def mean_accuracy(output, target, num_classes):
    """
    平均像素准确率 MPA（Mean Pixel Accuracy）。
    只在标签中真实出现过的类别上求平均。
    """
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)  # shape [N, H, W]

        accuracies = []
        for i in range(num_classes):
            target_mask = (target == i)
            predicted_mask = (predicted == i)
            # TP：预测正确的像素数
            intersection = torch.logical_and(target_mask, predicted_mask).sum().item()
            total = target_mask.sum().item()
            if total > 0:
                accuracies.append(intersection / total)

        if len(accuracies) == 0:
            return 0.0
        return sum(accuracies) / len(accuracies)


def mean_iou(output, target, num_classes):
    """平均交并比 mIoU，只在标签中出现过的类别上取平均。"""
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)  # (N, H, W)
        ious = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)
            intersection = torch.logical_and(target_mask, pred_mask).sum().item()
            union = torch.logical_or(target_mask, pred_mask).sum().item()
            if target_mask.sum().item() > 0:
                ious.append(intersection / union if union > 0 else 0.0)
        if len(ious) == 0:
            return 0.0
        return sum(ious) / len(ious)


def frequency_weighted_iou(output, target, num_classes):
    """按类别出现频率加权的 IoU（FWIoU）。"""
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        ious = []
        frequencies = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)
            intersection = torch.logical_and(target_mask, pred_mask).sum().item()
            union = torch.logical_or(target_mask, pred_mask).sum().item()
            freq = target_mask.sum().item()
            frequencies.append(freq)
            ious.append((intersection / union) if union > 0 else 0.0)

        total = sum(frequencies)
        if total == 0:
            return 0.0
        return sum(f * iou for f, iou in zip(frequencies, ious)) / total


# ------------------------------------------------------------------ #
#   混淆矩阵：在整个验证集上一次性统计，指标比“按 batch 求平均”更准确
# ------------------------------------------------------------------ #
class ConfusionMatrix:
    """
    语义分割混淆矩阵。

    为什么要用它：
        原实现是“每个 batch 各算一次指标，再对 batch 求平均”。
        由于最后一个 batch 的样本数往往更少，它与满 batch 的权重却完全相同，
        会让最终指标产生偏差；而且每个 batch 的“出现类别”不同，
        MPA 的平均口径也不一致。用混淆矩阵在全验证集上累计统计则没有这些问题。
    """

    def __init__(self, num_classes: int, ignore_index: int = None):
        """
        参数：
        - num_classes:  类别数
        - ignore_index: 需要忽略的标签值（如 255 对应的忽略类）
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # 行 = 真实类别，列 = 预测类别
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, outputs, target):
        """用一个 batch 的结果更新混淆矩阵。"""
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)  # [n, h, w]
            if self.ignore_index is not None:
                # 只统计有效像素
                valid = target != self.ignore_index
                pred = pred[valid]
                tgt = target[valid]
            else:
                tgt = target.reshape(-1)

            # 过滤掉越界标签，避免 bincount 越界
            valid2 = (tgt >= 0) & (tgt < self.num_classes)
            pred = pred[valid2]
            tgt = tgt[valid2]

            if tgt.numel() == 0:
                return

            idx = tgt * self.num_classes + pred
            binc = torch.bincount(idx, minlength=self.num_classes ** 2)
            self.mat += binc.cpu().numpy().reshape(self.num_classes, self.num_classes)

    def update_from_pred(self, pred, target):
        """
        用“已经确定类别”的预测图更新混淆矩阵。

        与 update() 的区别：update() 接收模型输出并自行 argmax；
        本方法接收成品预测图，供推理后处理（如去小连通域）之后使用。
        """
        with torch.no_grad():
            pred = pred.detach().cpu() if torch.is_tensor(pred) else torch.as_tensor(np.asarray(pred))
            tgt = target.detach().cpu() if torch.is_tensor(target) else torch.as_tensor(np.asarray(target))

            pred = pred.reshape(-1).long()
            tgt = tgt.reshape(-1).long()

            if self.ignore_index is not None:
                valid = tgt != self.ignore_index
                pred, tgt = pred[valid], tgt[valid]

            valid2 = (tgt >= 0) & (tgt < self.num_classes)
            pred, tgt = pred[valid2], tgt[valid2]

            if tgt.numel() == 0:
                return

            idx = tgt * self.num_classes + pred
            binc = torch.bincount(idx, minlength=self.num_classes ** 2)
            self.mat += binc.cpu().numpy().reshape(self.num_classes, self.num_classes)

    def compute(self):
        """根据累计的混淆矩阵计算 PA / MPA / mIoU / FWIoU。"""
        mat = self.mat.astype(np.float64)
        total = mat.sum()

        if total == 0:
            return {'Pixel Accuracy': 0.0, 'Mean Accuracy': 0.0,
                    'Mean IoU': 0.0, 'Frequency Weighted IoU': 0.0,
                    'Foreground IoU': 0.0,
                    'IoU per class': [0.0] * self.num_classes}

        # PA：对角线和 / 全部像素
        pixel_acc = np.diag(mat).sum() / total

        # 每类的召回率（= 该类准确率），只在真实出现过的类上平均
        gt_per_class = mat.sum(axis=1)          # 每类真实像素数
        pred_per_class = mat.sum(axis=0)        # 每类预测像素数
        tp = np.diag(mat)

        # MPA
        valid_cls = gt_per_class > 0
        mean_acc = (tp[valid_cls] / gt_per_class[valid_cls]).mean() if valid_cls.any() else 0.0

        # 每类 IoU = TP / (TP + FP + FN)
        union = gt_per_class + pred_per_class - tp
        valid_iou = union > 0
        mean_iou_val = (tp[valid_iou] / union[valid_iou]).mean() if valid_iou.any() else 0.0

        # 每类 IoU（背景 + 各前景类），以及只在前景类上取的平均。
        # 背景通常占 7 成以上、IoU 接近 1，会把 mIoU 抬高；
        # 真正反映目标提取质量的是前景 IoU，两者应一起看。
        iou_per_class = np.zeros(self.num_classes, dtype=np.float64)
        iou_per_class[valid_iou] = tp[valid_iou] / union[valid_iou]
        fg_ious = iou_per_class[1:][union[1:] > 0] if self.num_classes > 1 else np.array([])
        foreground_iou = float(fg_ious.mean()) if fg_ious.size else 0.0

        # FWIoU：按真实类别频率加权
        fw_iou = ((gt_per_class[valid_iou] / total) * (tp[valid_iou] / union[valid_iou])).sum() \
            if valid_iou.any() else 0.0

        return {
            'Pixel Accuracy': float(pixel_acc),
            'Mean Accuracy': float(mean_acc),
            'Mean IoU': float(mean_iou_val),
            'Frequency Weighted IoU': float(fw_iou),
            'Foreground IoU': foreground_iou,
            'IoU per class': iou_per_class.tolist(),
        }


def train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                    gpu_used, num_classes, scaler, epoch, train_epoch,
                    max_grad_norm=10.0, cls_weights=None, amp_dtype=None):
    """
    训练一个 epoch。

    参数：
    - model:          模型
    - optimizer:      优化器
    - train_loader:   训练 dataloader
    - device:         设备
    - dice_loss:      是否叠加 Dice Loss
    - focal_loss:     是否使用 Focal Loss 代替交叉熵
    - gpu_used:       当前显存占用（仅用于打印）
    - num_classes:    类别数（同时作为 ignore_index）
    - scaler:         混合精度 GradScaler，None 表示不做 loss scaling
                       （fp16 必须传；bf16 / fp32 传 None）
    - epoch:          当前 epoch（从 0 开始）
    - train_epoch:    总 epoch 数
    - max_grad_norm:  梯度裁剪阈值，<=0 表示不裁剪
    - cls_weights:    类别权重（长度 num_classes）；None 表示全 1（不加权）
    - amp_dtype:      autocast 的精度类型；None 表示不做 autocast（纯 fp32）
                       传 torch.bfloat16 可在不引入 GradScaler 的情况下做半精度前向
    返回：
    - 本 epoch 的平均损失
    """
    # 设置类别权重参数。它是用来处理类别不平衡的问题的
    # 默认全 1（等价于不做加权）；调用方可按训练集的像素频率传入权重
    if cls_weights is None:
        cls_weights = np.ones([num_classes], np.float32)
    else:
        cls_weights = np.asarray(cls_weights, dtype=np.float32)
    epoch_loss = 0.0  # 总的训练损失

    # 是否启用 autocast：显式给 amp_dtype，或者给了 scaler（fp16 的调用方式）
    use_autocast = (amp_dtype is not None) or (scaler is not None)

    # 设置模型为训练模式
    model.train()

    for iteration, batch in enumerate(train_loader):
        imgs, pngs, labels = batch  # 获取输入图像、标签和分割目标

        weights = torch.tensor(cls_weights).to(device)  # 转换类别权重并移动到GPU
        imgs = imgs.to(device)
        pngs = pngs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清除上一轮的梯度

        if not use_autocast:
            # 不使用混合精度
            outputs = model(imgs)
            loss = compute_loss(outputs, pngs, labels, weights, num_classes, dice_loss, focal_loss)

            loss.backward()
            # 梯度裁剪：抑制偶发的梯度爆炸，训练更稳定
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()  # 更新模型参数
        else:
            with autocast(dtype=amp_dtype):
                outputs = model(imgs)

            # 注意：损失统一在 FP32 下计算
            # Dice 里有 softmax / 除法，FP16 下容易溢出或精度损失
            loss = compute_loss(outputs.float(), pngs, labels, weights, num_classes, dice_loss, focal_loss)

            if scaler is not None:
                # 反向传播（fp16 混合精度）
                scaler.scale(loss).backward()
                if max_grad_norm and max_grad_norm > 0:
                    # 裁剪前需要先把梯度还原到真实尺度
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)  # 使用混合精度更新梯度
                scaler.update()  # 更新scaler
            else:
                # bf16 / 不做 loss scaling 的路径：与纯 fp32 一致，只是前向用了低精度
                loss.backward()
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # 累加训练损失
        epoch_loss += loss.item()

        # 打印标题（每个 epoch 的第一个 batch 打印一次）
        if iteration == 0:
            print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}"
                  )

        # 使用 \r 在同一行更新输出
        Epoch_len = len("Epoch") + 12 - len(str(f"{epoch + 1}/{train_epoch}"))
        batch_len = len("data_num") + 12 - len(str(f"{iteration + 1}/{len(train_loader)}"))
        GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
        Loss_len = len("Loss") + 12 - len(str(f"{loss.item():.8f}"))
        LR_len = len("LR") + 12 - len(str(f"{get_lr(optimizer):.8f}"))

        print(f"\r{epoch + 1}/{train_epoch}{' ' * Epoch_len}"
              f"{iteration + 1}/{len(train_loader)}{' ' * batch_len}"
              f"{gpu_used:.2f} MB{' ' * GPU_len}"
              f"{loss.item():.8f}{' ' * Loss_len}"
              f"{get_lr(optimizer):.8f}{' ' * LR_len}"
              f"{imgs.shape[2]}", end='', flush=True)

    # 每个epoch结束后打印一次
    print(f"{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    # 返回平均loss
    return epoch_loss / max(len(train_loader), 1)


def compute_loss(outputs, pngs, labels, weights, num_classes, dice_loss, focal_loss):
    """
    统一的损失计算入口：交叉熵 / Focal + 可选的 Dice。

    参数：
    - outputs:     模型输出 [n, c, h, w]
    - pngs:        类别索引标签 [n, h, w]
    - labels:      one-hot 标签 [n, h, w, c+1]
    - weights:     类别权重
    - num_classes: 类别数（同时作为 ignore_index）
    - dice_loss:   是否叠加 Dice Loss
    - focal_loss:  是否使用 Focal Loss
    """
    if focal_loss:
        loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
    else:
        loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

    if dice_loss:
        loss = loss + Dice_loss(outputs, labels)

    return loss


def evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes, cls_weights=None):
    """
    在验证集上评估模型。

    与旧实现的区别：指标由“逐 batch 求平均”改为“全验证集混淆矩阵精确统计”，
    口径更标准，也不会被最后一个不完整的 batch 拉偏。


    返回：
    - 包含 Pixel Accuracy / Mean Accuracy / Mean IoU / Frequency Weighted IoU / Loss 的字典

    说明：cls_weights 只影响返回的 Loss 数值，PA / MPA / mIoU / FWIoU 与其无关。
    """
    # 默认全 1（不加权）；与训练保持同一套权重可让 Loss 口径一致
    if cls_weights is None:
        cls_weights = np.ones([num_classes], np.float32)
    else:
        cls_weights = np.asarray(cls_weights, dtype=np.float32)
    val_loss = 0.0  # 记录总的验证损失

    # 设置模型为验证模式
    model.eval()

    # 混淆矩阵（num_classes 即 ignore_index）
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=num_classes)

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels = batch

            weights = torch.tensor(cls_weights).to(device)
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            # 损失同样在 FP32 下计算，保证与训练口径一致
            loss = compute_loss(outputs.float(), pngs, labels, weights, num_classes, dice_loss, focal_loss)
            val_loss += loss.item()

            # 累计混淆矩阵（全验证集汇总后再算指标，不是逐 batch 平均）
            cm.update(outputs, pngs)

            # 打印进度
            print(f"\r{LogColor.RED}val{LogColor.RESET} "
                  f"{iteration + 1}/{len(val_loader)}", end='', flush=True)

    num_batches = max(len(val_loader), 1)
    metrics = cm.compute()
    metrics['Loss'] = val_loss / num_batches

    print(f"\n{LogColor.GREEN}"
          f"PixelAcc {metrics['Pixel Accuracy']:.4f} | "
          f"MeanAcc {metrics['Mean Accuracy']:.4f} | "
          f"MeanIoU {metrics['Mean IoU']:.4f} | "
          f"FWIoU {metrics['Frequency Weighted IoU']:.4f} | "
          f"前景IoU {metrics['Foreground IoU']:.4f}{LogColor.RESET}", end='', flush=True)
    print()
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    return metrics

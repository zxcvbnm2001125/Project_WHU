import os

import torch
import numpy as np
from model.unet_training import CE_Loss, Dice_loss, Focal_Loss

from utils.utils import get_lr
from torch.cuda.amp import autocast, GradScaler
import time


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


def pixel_accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).float()
        correct_pixels = correct.sum().item()
        total_pixels = target.numel()
        return correct_pixels / total_pixels

def mean_accuracy(output, target, num_classes):
    """
    计算 Mean Pixel Accuracy (MPA).
    :param output: torch.Tensor, shape [N, C, H, W]
    :param target: torch.Tensor, shape [N, H, W]
    :param num_classes: int
    :return: float, mean pixel accuracy over valid classes
    """
    with torch.no_grad():
        # 取出每个像素的预测类别索引
        _, predicted = torch.max(output, dim=1)  # shape [N, H, W]

        accuracies = []
        for i in range(num_classes):
            # 找到该类别在标签和预测中的位置
            target_mask = (target == i)
            predicted_mask = (predicted == i)

            # 交集：预测正确的像素数（即 TP）
            intersection = torch.logical_and(target_mask, predicted_mask).sum().item()
            total = target_mask.sum().item()  # 标签中该类的总像素数

            if total > 0:
                acc = intersection / total
                accuracies.append(acc)
            # 如果该类别在 GT 中没有出现，则跳过，不计入平均

        # 防止所有类别都未出现
        if len(accuracies) == 0:
            return 0.0
        else:
            return sum(accuracies) / len(accuracies)


# 计算Mean IoU
def mean_iou(output, target, num_classes):
    """
    计算 mean IoU，只在 target 出现的类别中取平均
    """
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)  # (N, H, W)
        ious = []
        for i in range(num_classes):
            target_mask = (target == i)
            pred_mask = (predicted == i)

            intersection = torch.logical_and(target_mask, pred_mask).sum().item()
            union = torch.logical_or(target_mask, pred_mask).sum().item()

            if target_mask.sum().item() > 0:  # 只对 target 中存在的类求 IoU
                ious.append(intersection / union if union > 0 else 0.0)
        if len(ious) == 0:
            return 0.0
        return sum(ious) / len(ious)


def foreground_iou(output, target, fg_class=1):
    """前景类（默认 1=建筑物）的 IoU。

    二分类分割里背景占比通常远大于前景，Mean IoU 会被背景抬高，
    真正反映建筑提取质量的是前景 IoU；消融统一按这个指标选优。
    """
    with torch.no_grad():
        _, predicted = torch.max(output, dim=1)
        t = (target == fg_class)
        p = (predicted == fg_class)
        intersection = torch.logical_and(t, p).sum().item()
        union = torch.logical_or(t, p).sum().item()
        return intersection / union if union > 0 else 0.0


# 计算Frequency Weighted IoU
def frequency_weighted_iou(output, target, num_classes):
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
        fw_iou = sum(f * iou for f, iou in zip(frequencies, ious)) / total
        return fw_iou


def train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                    gpu_used, num_classes, scaler, epoch, train_epoch):

    # 设置类别权重参数。它是用来处理类别不平衡的问题的
    cls_weights = np.ones([num_classes], np.float32)
    epoch_loss = 0.0 # 总的训练损失

    # 设置模型为训练模式
    model_train = model.train()
    model_train = model_train.to(device)

    # 遍历训练数据
    for iteration, batch in enumerate(train_loader):
        imgs, pngs, labels = batch  # 获取输入图像、标签和分割目标

        # 数据准备阶段：使用 `.to(device)` 自动将数据移到设备上
        weights = torch.tensor(cls_weights).to(device)  # 转换类别权重并移动到GPU
        imgs = imgs.to(device)
        pngs = pngs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清除之前的梯度

        # 混合精度训练
        if scaler is None:
            # 前向传播
            outputs = model_train(imgs)

            # 损失计算
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                # 如果使用Dice Loss，则加上Dice损失
                main_dice = Dice_loss(outputs, labels)
                # main_dice = Dice_loss(outputs, pngs)
                loss = loss + main_dice

            # 反向传播
            loss.backward()
            optimizer.step()  # 更新模型参数
        else:
            with autocast():
                outputs = model_train(imgs)  # 通过模型获取预测结果

                # 损失计算
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # 使用混合精度更新梯度
            scaler.update()  # 更新scaler

        # 累加训练损失和F-score
        epoch_loss += loss.item()

        # 打印标题（每个epoch开始时打印一次）
        if iteration == 0:  # 只在第一个 batch 打印标题
            print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}"
                  )

        # 每10个batch打印一次信息
        if iteration % 1 == 0:
            if len(train_loader) < 1:
                a = len(train_loader)
            else:
                a = 1

        Epoch_len = len("Epoch") + 12 - len(str(f"{epoch + 1}/{train_epoch}"))
        batch_len = len("data_num") + 12 - len(str(f"{iteration + a}/{len(train_loader)}"))
        GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
        Loss_len = len("Loss") + 12 - len(str(f"{loss.item():.8f}"))
        LR_len = len("LR") + 12 - len(str(f"{get_lr(optimizer):.8f}"))

        # 使用 \r 在同一行更新输出
        print(f"\r{epoch + 1}/{train_epoch}{' ' * Epoch_len}"
              f"{iteration + a}/{len(train_loader)}{' ' * batch_len}" 
              f"{gpu_used:.2f} MB{' ' * GPU_len}"
              f"{loss.item():.8f}{' ' * Loss_len}"
              f"{get_lr(optimizer):.8f}{' ' * LR_len}"
              f"{imgs.shape[2]}", end='', flush=True)

    # 每个epoch结束后打印一次
    print(f"{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    # ➕ 返回平均loss
    return epoch_loss / len(train_loader)

def evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes):

    cls_weights = np.ones([num_classes], np.float32)
    val_loss = 0  # 记录总的验证损失

    # 设置模型为验证模式
    model_eval = model.eval()
    model_eval = model_eval.to(device)

    # 初始化累积变量
    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_iou = 0
    total_fw_iou = 0
    total_fg_iou = 0
    num_batches = len(val_loader)

    # 遍历验证数据，前向传播
    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels = batch  # 获取验证数据

            # 数据准备阶段
            weights = torch.tensor(cls_weights).to(device)  # 转换类别权重并移动到GPU
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            labels = labels.to(device)
            outputs = model_eval(imgs)

            # print("outputs", outputs.shape)
            # print("pngs", pngs.shape)
            # 损失计算
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                # main_dice = Dice_loss(outputs, pngs)
                loss = loss + main_dice

            # 计算各个指标
            pixel_acc = pixel_accuracy(outputs, pngs)
            mean_acc = mean_accuracy(outputs, pngs, num_classes)
            mean_iou_value = mean_iou(outputs, pngs, num_classes)
            fw_iou = frequency_weighted_iou(outputs, pngs, num_classes)
            fg_iou_value = foreground_iou(outputs, pngs, fg_class=1)

            # 累加到总结果
            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou
            total_fg_iou += fg_iou_value
            val_loss += loss.item()

            # 打印标题（每个epoch开始时打印一次）
            if iteration == 0:  # 只在第一个 batch 打印标题
                epoch_len = len("Epoch") + 12
                data_num_len = len("data_num") - len("data_num") + 12
                Pixelacc_len = len("GPU Mem") - len("Pixelacc") + 12
                Meanacc_len = len("Loss") - len("Meanacc") + 12
                Meaniou_len = len("LR") - len("Meaniou") + 12

                print(f"{' ' * epoch_len}"
                      f"{LogColor.RED}data_num{LogColor.RESET}{' ' * data_num_len}"
                      f"{LogColor.RED}Pixelacc{LogColor.RESET}{' ' * Pixelacc_len}"
                      f"{LogColor.RED}Meanacc{LogColor.RESET}{' ' * Meanacc_len}"
                      f"{LogColor.RED}Meaniou{LogColor.RESET}{' ' * Meaniou_len}"
                      f"{LogColor.RED}Fwiou{LogColor.RESET}{' ' * 6}"
                      f"{LogColor.RED}Fgiou{LogColor.RESET}")

    # 计算平均值
    avg_pixel_acc = total_pixel_acc / num_batches
    avg_mean_acc = total_mean_acc / num_batches
    avg_mean_iou = total_mean_iou / num_batches
    avg_fw_iou = total_fw_iou / num_batches
    avg_fg_iou = total_fg_iou / num_batches
    avg_loss = val_loss / num_batches  # ➕ 平均 loss


    # 将结果保存到字典中
    metrics = {
        'Pixel Accuracy': avg_pixel_acc,
        'Mean Accuracy': avg_mean_acc,
        'Mean IoU': avg_mean_iou,
        'Frequency Weighted IoU': avg_fw_iou,
        'FG IoU': avg_fg_iou,
        'Loss': avg_loss  # ➕ 加入字典
    }

    epoch_len = len("Epoch") + 12
    batch_len = data_num_len + len("data_num") - len(str(f"{iteration + 1}/{len(val_loader)}"))
    # 指标统一打印 4 位小数：2 位时轮间变化常被四舍五入吃掉，看不出模型是否在动
    avg_pixel_acc_len = Pixelacc_len + len("Pixelacc") - len(str(f"{avg_pixel_acc:.4f}"))
    avg_mean_acc_len = Meanacc_len + len("Meanacc") - len(str(f"{avg_mean_acc:.4f}"))
    avg_Mean_iou_len = Meaniou_len + len("Meaniou") - len(str(f"{avg_mean_iou:.4f}"))

    # 使用 \r 在同一行更新输出
    print(f"{' ' * (epoch_len)}"
          f"{iteration + 1}/{len(val_loader)}{' ' * batch_len}"
          f"{avg_pixel_acc:.4f}{' ' * avg_pixel_acc_len}"
          f"{avg_mean_acc:.4f}{' ' * avg_mean_acc_len}"
          f"{avg_mean_iou:.4f}{' ' * avg_Mean_iou_len}"
          f"{avg_fw_iou:.4f}{' ' * 8}"
          f"{avg_fg_iou:.4f}", end='', flush=True)
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    return metrics
import torch
from torch import nn
import time


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


# 计算Pixel Accuracy
def pixel_accuracy(output, target):
    with torch.no_grad():
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).float()
        correct_pixels = correct.sum().item()
        total_pixels = target.numel()
        return correct_pixels / total_pixels


# 计算Mean Accuracy
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


def foreground_iou(output, target, fg_class=1):
    """前景类（1=建筑物）IoU。二分类下 Mean IoU 被背景抬高，选优应看这个。"""
    import torch as _t
    with _t.no_grad():
        _, predicted = _t.max(output, dim=1)
        t = (target == fg_class)
        pp = (predicted == fg_class)
        inter = _t.logical_and(t, pp).sum().item()
        union = _t.logical_or(t, pp).sum().item()
        return inter / union if union > 0 else 0.0


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


def criterion(inputs, target):
    """
    计算交叉熵损失（Cross-Entropy Loss），用于多类分类任务，
    并且忽略目标中值为 255 的像素（通常表示目标边缘或填充区域）。

    Args:
        inputs (dict): 模型的预测结果，字典形式，每个键是预测的名称（例如 'out' 或 'aux'），每个值是对应的预测张量。
        target (Tensor): 目标标签，包含每个像素的真实类别。值为 255 的像素通常被视为忽略的区域。

    Returns:
        Tensor: 计算得到的损失值。如果 `inputs` 包含多个预测结果，则返回加权损失；否则，返回单个损失。
    """
    losses = {}  # 初始化一个空字典，用于存储每个预测结果的损失

    # 遍历每个输入（通常是不同层的输出），计算其损失
    for name, x in inputs.items():
        # 使用交叉熵损失计算，忽略目标标签中值为 255 的像素（背景或填充区域）
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    # 如果只有一个损失（即只有一个输出），直接返回该损失
    if len(losses) == 1:
        return losses['out']

    # 如果有多个损失（例如主分类器和辅助分类器的损失），
    # 返回主分类器的损失加上辅助分类器损失的加权和
    return losses['out'] + 0.5 * losses['aux']


def train_one_epoch(model, optimizer, data_loader, device, epoch, train_epoch, gpu_used, lr_scheduler, scaler=None):
    # 将模型切换到训练模式
    model.train()
    epoch_loss = 0.0
    for batch_idx, (image, target) in enumerate(data_loader):
        # 将图像和标签转移到指定的设备（GPU/CPU）
        image, target = image.to(device), target.to(device)

        # 如果启用了混合精度训练，使用autocast管理精度
        with torch.cuda.amp.autocast(enabled=scaler is not None):
        # with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):  # 在pytorch2.7.1版本可以运用该代码可以去除警告信息
            # 前向传递：模型生成输出
            output = model(image)
            # 计算损失
            loss = criterion(output, target)

        # 梯度清零
        optimizer.zero_grad()

        # 如果启用了混合精度训练，使用GradScaler进行反向传播和优化步骤
        if scaler is not None:
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新优化器
            scaler.update()  # 更新缩放因子
        else:
            # 普通的反向传播和优化步骤
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        # 更新学习率调度器
        lr_scheduler.step()

        # 获取当前学习率
        lr = optimizer.param_groups[0]["lr"]

        # 打印标题（每个epoch开始时打印一次）
        if batch_idx == 0:  # 只在第一个 batch 打印标题
            print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
                  f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}"
                  )

        # 每10个batch打印一次信息
        if batch_idx % 1 == 0:
            if len(data_loader) < 1:
                a = len(data_loader)
            else:
                a = 1

        Epoch_len = len("Epoch") + 12 - len(str(f"{epoch + 1}/{train_epoch}"))
        batch_len = len("data_num") + 12 - len(str(f"{batch_idx + a}/{len(data_loader)}"))
        GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
        Loss_len = len("Loss") + 12 - len(str(f"{loss.item():.8f}"))
        LR_len = len("LR") + 12 - len(str(f"{lr:.8f}"))

        # 使用 \r 在同一行更新输出
        print(f"\r{epoch + 1}/{train_epoch}{' ' * Epoch_len}"
              f"{batch_idx + a}/{len(data_loader)}{' ' * batch_len}" 
              f"{gpu_used:.2f} MB{' ' * GPU_len}"
              f"{loss.item():.8f}{' ' * Loss_len}"
              f"{lr:.8f}{' ' * LR_len}"
              f"{image.shape[2]}", end='', flush=True)



    # 每个epoch结束后打印一次
    print(f"{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    # ➕ 返回平均loss
    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, device, num_classes):
    # 将模型设置为评估模式，禁用dropout和batch normalization的训练行为
    model.eval()

    # 初始化累积变量
    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_iou = 0
    total_fw_iou = 0
    total_fg_iou = 0
    total_loss = 0
    num_batches = len(data_loader)

    # 在不计算梯度的情况下进行评估
    with torch.no_grad():
        for batch_idx, (image, target) in enumerate(data_loader):
            # 将图像和目标标签移到指定的设备（GPU/CPU）
            image, target = image.to(device), target.to(device)

            # 前向传播：获取模型输出
            output = model(image)
            # 假设模型输出包含一个字典，'out'键是最终的预测结果

            # ➕ 计算 loss
            loss = criterion(output, target)

            output = output['out']
            # print(output.keys())  # 打印模型输出的键名

            # 计算各个指标
            pixel_acc = pixel_accuracy(output, target)
            mean_acc = mean_accuracy(output, target, num_classes)
            mean_iou_value = mean_iou(output, target, num_classes)
            fw_iou = frequency_weighted_iou(output, target, num_classes)
            fg_iou_value = foreground_iou(output, target, fg_class=1)

            # 累加到总结果
            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou
            total_fg_iou += fg_iou_value
            total_loss += loss.item()  # ➕ 累加 loss

            # 打印标题（每个epoch开始时打印一次）
            if batch_idx == 0:  # 只在第一个 batch 打印标题
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
    avg_loss = total_loss / num_batches  # ➕ 平均 loss

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
    batch_len = data_num_len + len("data_num") - len(str(f"{batch_idx + 1}/{len(data_loader)}"))
    # 指标统一打印 4 位小数：2 位时轮间变化常被四舍五入吃掉，看不出模型是否在动
    avg_pixel_acc_len = Pixelacc_len + len("Pixelacc") - len(str(f"{avg_pixel_acc:.4f}"))
    avg_mean_acc_len = Meanacc_len + len("Meanacc") - len(str(f"{avg_mean_acc:.4f}"))
    avg_Mean_iou_len = Meaniou_len + len("Meaniou") - len(str(f"{avg_mean_iou:.4f}"))

    # 使用 \r 在同一行更新输出
    print(f"{' ' * (epoch_len)}"
          f"{batch_idx + 1}/{len(data_loader)}{' ' * batch_len}"
          f"{avg_pixel_acc:.4f}{' ' * avg_pixel_acc_len}"
          f"{avg_mean_acc:.4f}{' ' * avg_mean_acc_len}"
          f"{avg_mean_iou:.4f}{' ' * avg_Mean_iou_len}"
          f"{avg_fw_iou:.4f}{' ' * 8}"
          f"{avg_fg_iou:.4f}", end='', flush=True)
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快

    return metrics



def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    # 创建学习率更新的函数f
    def f(x):
        """
               根据当前训练步数x返回学习率倍率因子。
               这使得学习率在训练过程中可以进行变化。
               注意：PyTorch会在训练开始前调用lr_scheduler.step()一次。
        """
        # 如果启用了warmup并且当前步骤数在warmup阶段内
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # 在warmup阶段，学习率从warmup_factor增加到1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup结束后，学习率从1衰减到0
            # 参考DeepLabV2的学习率策略：采用指数衰减（pow(0.9)）
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    # 返回一个LambdaLR调度器，使用上述定义的学习率更新函数
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

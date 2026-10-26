# train.py
# -*- coding: utf-8 -*-
"""
DeepLabV3/DeepLabV3+ 训练脚本（按 epoch 训练）。

控制台输出格式与 U-Net / FCN 两个工程一致（绿色 Epoch + 黄色训练表头，
红色验证表头，指标 4 位小数），见 utils/console.py。
"""
import argparse
import os
import subprocess

import torch
import torch.nn as nn

import network
from utils.engine import (setup_device_and_seed, resolve_num_classes,
                          build_loaders, build_model, build_metrics, build_optimizer, build_scheduler,
                          build_criterion, save_checkpoint, validate, restore_from_checkpoint)

from utils.create_exp_folder import create_exp_folder
from utils.seg_metrics_logger import SegMetricsLogger
from utils.console import (print_train_header, print_train_line, print_train_end,
                           print_val_header, print_val_line)


def get_gpu_usage():
    """整卡已用显存（MB），与 U-Net / FCN 的实现一致。

    注意这是 nvidia-smi 报的设备级占用，不是本进程峰值，
    不能当作"本模型的显存开销"填进对比表。
    """
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, total = map(int, result.strip().split(','))
        return used
    except Exception:
        return 0.0

# =============================================================================
# 运行配置（W3 · 消融 A：模型对比）
# 直接改下面的数值即可，命令行不传参时全部生效；想临时覆盖仍可用命令行传参。
# 统一口径：WUH 固定子集 train 1184 / val 474 · crop 512 · 20 轮 · bs8 · seed 11
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../XR_1/DeepLabV3Plus
_ABLATION = os.path.dirname(os.path.dirname(_HERE))          # .../消融实验

CONFIG = dict(
    data_root=os.path.join(_ABLATION, r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25"),
    num_classes=2,        # 类别数（含背景）；WHU 二分类 = 2
    model="deeplabv3plus_xception16",
    ckpt=os.path.join(_HERE, "weight", "best_deeplabv3plus_xception16_os16_num21.pth"),
    continue_training=False,   # True = 断点续训（恢复 optimizer/scheduler）；False = 当预训练权重
    loss_type="cross_entropy",
    dice_loss=True,      # 消融 B 开关：False = 纯 CE；True = CE + Dice（权重 1:1）
    weight_decay=1e-4,
    output_stride=16,
    separable_conv=False,
    epochs=25,
    lr=0.001,             # backbone 用 0.1*lr，classifier 用 lr
    lr_policy="poly",
    step_size=10000,
    batch_size=4,
    val_batch_size=4,     # crop_val=False 时引擎会强制改为 1
    crop_size=512,
    crop_val=False,
    gpu_id="0",
    random_seed=11,       # 上游默认 1，与 U-Net / FCN 的 11 统一
)

def main(opts):
    # ---------- 1) 补全 num_classes ----------
    # 如果你没有显式传 --num_classes，就按数据集默认：
    resolve_num_classes(opts)

    # ---------- 2) 设备与随机种子 ----------
    device = setup_device_and_seed(opts)
    print(f"Device: {device}")

    # ---------- 3) 数据集与 DataLoader ----------
    train_dst, val_dst, train_loader, val_loader = build_loaders(opts)
    print(f"Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    # ---------- 3.1) 创建本次实验的文件夹结构：run/train/exp*/weights ----------
    #   exp_dir   : 用来放本次实验的训练日志、图表等（后面可以往这里写曲线图、配置文件）
    #   weights_dir : 用来放 latest / best 的权重文件
    exp_dir, weights_dir = create_exp_folder()
    print(f"[Folder] Experiment dir: {exp_dir}")
    print(f"[Folder] Weights dir   : {weights_dir}")
    # ---------- 3.2) 指标记录器：每个 epoch 写入 exp_dir/metrics.csv，并在训练结束后保存曲线图 ----------
    metrics_logger = SegMetricsLogger(exp_dir)

    # 这次跑的是 CE 还是 CE+Dice 必须留下来，否则两个 exp 目录事后分不清
    loss_mode = "CE+Dice" if opts.dice_loss else "CE"
    print(f"损失函数: {loss_mode} (loss_type={opts.loss_type})")
    with open(os.path.join(exp_dir, "train_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"loss_mode: {loss_mode}\n")
        f.write(f"dice_loss: {opts.dice_loss}\n")
        for k, v in sorted(vars(opts).items()):
            f.write(f"{k}: {v}\n")

    # 计算总迭代次数（供 PolyLR 使用）= 训练轮数 * 每个 epoch 的 iteration 数
    if len(train_loader) == 0:
        raise RuntimeError("train_loader 为空，检查数据集路径或划分是否正确。")
    opts.total_itrs = opts.epochs * len(train_loader)

    # ---------- 4) 模型 / 指标 / 优化器 / scheduler / loss ----------
    model = build_model(opts)
    metrics = build_metrics(opts)
    optimizer = build_optimizer(opts, model)
    scheduler = build_scheduler(opts, optimizer)
    criterion = build_criterion(opts)

    # ---------- 5) checkpoint 恢复 ----------
    model, best_score, cur_itrs, start_epoch = restore_from_checkpoint(
        opts=opts,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
    )

    # ---------- 6) DataParallel + to(device) ----------
    # 只在 CUDA 上包 DataParallel：CPU 上 device_ids 为空，nn.DataParallel 会抛
    # IndexError: list index out of range（纯 CPU 调试时必踩）
    if device.type == "cuda" and torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
    model.to(device)

    # --------------------------
    # 7) 训练主循环（按 epoch）
    # --------------------------
    best_epoch = start_epoch  # 记录最优 mIoU 出现的 epoch（1-based 会用 epoch+1）
    best_ckpt_path = None  # 记录最优权重的保存路径
    for epoch in range(start_epoch, opts.epochs):
        # 每个epoch开始前，把模型切到“训练模式”（启用 BN、Dropout 的训练行为）
        model.train()

        # 统计本 epoch 的训练信息
        epoch_loss = 0.0  # 累计当前 epoch 中所有 batch 的 loss 之和
        correct_pixels = 0  # 累计预测正确的像素数（忽略 label=255 的像素）
        total_pixels = 0  # 累计参与统计的像素总数（label != 255）
        num_batches = 0  # 当前 epoch 已经跑了多少个 batch

        # 训练期表头（配色与字段顺序同 U-Net / FCN）
        gpu_used = get_gpu_usage()
        print_train_header()

        # 一个 epoch 内按 batch 遍历训练集
        for images, labels in train_loader:
            cur_itrs += 1  # 全局迭代步数 +1（方便保存 ckpt 和恢复训练）
            num_batches += 1  # 该 epoch 内的 batch 计数 +1

            # 把数据搬到对应设备（GPU / CPU），并设定数据类型
            images = images.to(device, dtype=torch.float32)  # 图像用 float32
            labels = labels.to(device, dtype=torch.long)  # 标签是类别 id，用 long

            # --------- 前向 + 反向 + 更新参数 ---------
            optimizer.zero_grad()  # 清空上一轮迭代的梯度
            outputs = model(images)  # 前向传播，得到预测 logits，形状 [N, C, H, W]
            loss = criterion(outputs, labels)  # 计算当前 batch 的损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 根据梯度更新模型参数

            # ------- 统计训练 loss（用于 epoch 平均 loss 与 CSV） -------
            batch_loss = float(loss.detach().cpu().numpy())  # 转成 float，避免张量残留在图里
            epoch_loss += batch_loss  # 累加当前 batch loss

            # ------- 统计像素级准确率（忽略 label=255 的像素）-------
            # 这里在 no_grad 下做统计，避免额外梯度开销
            with torch.no_grad():
                # 取每个像素预测类别：在通道维上取 argmax，得到 [N, H, W]
                preds = outputs.detach().max(dim=1)[1]

                # 有效像素掩码：label=255 的像素是 ignore，不参与准确率计算
                valid_mask = (labels != 255)

                # 统计这一个 batch 有多少有效像素
                total_pixels += valid_mask.sum().item()
                # 在有效像素位置上预测正确的像素数
                correct_pixels += ((preds == labels) & valid_mask).sum().item()

            # 学习率调度器仍然按 iteration 更新（每个 batch 调一次）
            scheduler.step()

            # 训练期逐 batch 原地刷新一行（字段与 U-Net / FCN 一致）
            print_train_line(
                epoch=epoch + 1,
                total_epochs=opts.epochs,
                it=num_batches,
                total_it=len(train_loader),
                gpu_used=gpu_used,
                loss=batch_loss,
                lr=optimizer.param_groups[-1]["lr"],  # classifier 组 = 基础 lr
                img_size=images.shape[-1],
            )

        print_train_end()

        # ------- 每个 epoch 结束后，在验证集上做一次完整 validate -------
        print_val_header()
        val_score = validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=metrics,
            criterion=criterion,  # 把 loss 函数传进去，这样验证时也能算 val loss
            epoch=epoch + 1,
            total_epochs=opts.epochs,
        )
        # 验证期结果行：字段与配色同 U-Net / FCN（Pixelacc / Meanacc / Meaniou / Fwiou / Fgiou）
        print_val_line(
            n_done=len(val_loader),
            n_total=len(val_loader),
            pixel_acc=float(val_score.get("Overall Acc", 0.0)),
            mean_acc=float(val_score.get("Mean Acc", 0.0)),
            mean_iou=float(val_score.get("Mean IoU", 0.0)),
            fw_iou=float(val_score.get("FreqW Acc", 0.0)),
            fg_iou=float((val_score.get("Class IoU") or {}).get(1, 0.0)),
        )


        # ---------- 记录本 epoch 的 train/val 指标到 CSV ----------
        # 训练平均 loss（按 epoch）
        train_loss_epoch = epoch_loss / max(1, num_batches)
        # 训练像素准确率（你在循环里一直在累计 correct_pixels/total_pixels，这里直接复用）
        train_pacc_epoch = correct_pixels / total_pixels if total_pixels > 0 else 0.0

        # 当前学习率：建议记录 classifier/head 的 lr（你的 optimizer 第 2 组是 classifier）
        # 如果你后面改了 param_groups 顺序，这里相应调整即可
        lr_now = optimizer.param_groups[-1]["lr"]

        # 验证平均 loss：建议从 val_score 里拿（需要 validate() 把 Loss 写进去）
        val_loss_epoch = float(val_score.get("Loss", 0.0))

        metrics_logger.log_epoch(
            epoch=epoch + 1,
            lr=lr_now,
            train_loss=train_loss_epoch,
            train_pacc=train_pacc_epoch,
            val_loss=val_loss_epoch,
            val_score=val_score,
        )


        # ------- 保存 latest ckpt（“当前进度”的快照）到当前实验的 weights 下 -------
        latest_path = os.path.join(weights_dir,f"last_{opts.model}_os{opts.output_stride}_num{opts.num_classes}.pth")
        save_checkpoint(latest_path, model, optimizer, scheduler, cur_itrs, best_score, model_name=opts.model)

        # ------- 若当前验证「前景（建筑）IoU」刷新历史最优，则保存 best ckpt -------
        # 二分类下 Mean IoU 被背景抬高，与 U-Net / FCN / SegFormer 统一按前景 IoU 选优
        cur_fg_iou = float(val_score.get("Class IoU", {}).get(1, 0.0))
        if cur_fg_iou > best_score:
            best_score = cur_fg_iou
            best_epoch = epoch + 1  # 记录发生在第几个 epoch（1-based）
            best_ckpt_path = os.path.join(
                weights_dir,
                f"best_{opts.model}_os{opts.output_stride}_num{opts.num_classes}.pth",
            )
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, cur_itrs, best_score, model_name=opts.model)

    # 所有 epoch 跑完
    print("Training finished.")

    # ---------- 训练结束：保存曲线图到 exp_dir ----------
    metrics_logger.save_plots()
    print(f"[Plot] Saved curves to: {exp_dir} (loss_curve.png / pacc_curve.png / val_metrics_curve.png)")

    # 如果在本次训练过程中出现过“更好的 mIoU”，打印最优轮次和权重路径
    if best_ckpt_path is not None:
        print(f"[Best] Best FG IoU = {best_score:.4f} at epoch {best_epoch}")
        print(f"[Best] Best weights path: {best_ckpt_path}")
    else:
        # 没有超过初始 best_score（例如从一个 best ckpt 续训，但这次训练没刷新）
        print(f"[Best] No new best FG IoU in this run. Current best_score = {best_score:.4f}")
        print(f"[Best] Check previous run's best checkpoint if needed.")


if __name__ == "__main__":
    # ---------------- 模型相关 ----------------
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower()
        and not (name.startswith("__") or name.startswith("_"))
        and callable(network.modeling.__dict__[name]))

    # 这里定义“训练脚本”的命令行参数（默认值全部取文件顶部的 CONFIG）
    parser = argparse.ArgumentParser()
    # ---------------- 训练超参数（按 epoch） ----------------
    parser.add_argument("--data_root", type=str,
                        default=CONFIG["data_root"],
                        help="WUH 数据集根目录（内含 train|val 两个子目录，各含 image 与 label）")
    parser.add_argument("--num_classes", type=int, default=CONFIG["num_classes"], help="类别数（含背景）。不写，写错都会报错")
    parser.add_argument("--model", type=str, default=CONFIG["model"], choices=available_models,
                        help="模型名称，对应 network.modeling 里的工厂函数")
    parser.add_argument("--ckpt", type=str, default=CONFIG["ckpt"], help="用于加载预训练/中断权重")
    parser.add_argument("--continue_training", action="store_true", default=CONFIG["continue_training"], help="是否从 ckpt 继续训练（恢复 optimizer/scheduler/cur_itrs）")
    parser.add_argument("--loss_type", type=str, default=CONFIG["loss_type"], choices=["cross_entropy", "focal_loss"], help="基础损失类型")
    parser.add_argument("--dice_loss", action=argparse.BooleanOptionalAction, default=CONFIG["dice_loss"],
                        help="消融 B 开关：开启则基础损失 + Dice（权重 1:1），关闭则只用 --loss_type 指定的基础损失")
    parser.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"], help="权重衰减系数（L2 正则），默认 1e-4")
    parser.add_argument("--output_stride", type=int, default=CONFIG["output_stride"], choices=[8, 16], help="编码器输出特征的下采样倍数：8 或 16")
    parser.add_argument("--separable_conv", action="store_true", default=CONFIG["separable_conv"], help="是否在 ASPP 和 decoder 中使用 depthwise separable conv")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"], help="训练总轮数（epoch 数），默认 20")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="基础学习率（backbone 用 0.1*lr，classifier 用 lr）")
    parser.add_argument("--lr_policy", type=str, default=CONFIG["lr_policy"],  choices=["poly", "step"], help="学习率策略：poly / step")
    parser.add_argument("--step_size", type=int, default=CONFIG["step_size"], help="当 lr_policy=step 时，每多少个 iteration 衰减一次学习率")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], help="训练 batch size")
    parser.add_argument("--val_batch_size", type=int, default=CONFIG["val_batch_size"], help="验证 batch size（VOC + 不 crop 时会被强制改为 1）")
    parser.add_argument("--crop_size", type=int, default=CONFIG["crop_size"], help="训练/验证裁剪尺寸（WHU 瓦片为 512）")
    parser.add_argument("--crop_val", action="store_true", default=CONFIG["crop_val"], help="验证时是否也做 crop（默认 False，为原图评估）")
    parser.add_argument("--gpu_id", type=str, default=CONFIG["gpu_id"], help="GPU ID（写入 CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--random_seed", type=int, default=CONFIG["random_seed"],  help="随机种子，方便复现实验")
    opt = parser.parse_args()
    main(opt)



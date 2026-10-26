# -*- coding: utf-8 -*-
"""逐 epoch 指标落盘。

为什么需要：训练日志里的指标只在控制台一闪而过（而且是 4 位小数的文本），
训练结束后无法回答"第几轮开始收敛"这类问题。这里把每个 epoch 的验证指标
追加写进 <exp_folder>/metrics.csv，训练中断也不丢历史。

表结构刻意与 SegFormer 的 train_history.csv 对齐（epoch / loss / 各类 IoU），
方便四个模型横向汇总。

用法：
    logger = EpochMetricsLogger(os.path.join(exp_folder, "metrics.csv"))
    ...
    logger.log(epoch=epoch + 1, lr=lr_now, train_loss=loss,
               val_loss=metrics["Loss"], val_metrics=metrics,
               is_best=is_best, epoch_time=dt)
"""

import csv
import os
import time

FIELDNAMES = [
    "epoch",
    "lr",
    "train_loss",
    "val_loss",
    "pixel_acc",
    "mean_acc",
    "mean_iou",
    "fw_iou",
    "fg_iou",
    "is_best",
    "epoch_time_s",
]


class EpochMetricsLogger:
    """追加写 metrics.csv，一次一个 epoch。

    表头用 utf-8-sig（Excel 直接打开不乱码），数据行用 utf-8 追加——
    注意不能对追加也用 utf-8-sig，那样每行前面都会插一个 BOM。
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                header = f.readline().strip().split(",")
            if header != FIELDNAMES:
                # 旧表头（例如没有 fg_iou 列）无法继续追加，先备份再重建，避免静默丢列
                stamp = time.strftime("%Y%m%d_%H%M%S")
                backup = csv_path[:-4] + f".{stamp}.csv"
                os.replace(csv_path, backup)
                print(f"[CSV] 表头与当前版本不一致，旧文件已备份为：{backup}")

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log(self, *, epoch, lr, train_loss, val_loss, val_metrics, is_best, epoch_time):
        """写一行。val_metrics 用 evaluate() 返回的字典，缺列时按 0 处理。"""
        row = {
            "epoch": int(epoch),
            "lr": float(lr),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "pixel_acc": float(val_metrics.get("Pixel Accuracy", 0.0)),
            "mean_acc": float(val_metrics.get("Mean Accuracy", 0.0)),
            "mean_iou": float(val_metrics.get("Mean IoU", 0.0)),
            "fw_iou": float(val_metrics.get("Frequency Weighted IoU", 0.0)),
            "fg_iou": float(val_metrics.get("FG IoU", 0.0)),
            "is_best": int(bool(is_best)),
            "epoch_time_s": round(float(epoch_time), 2),
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
        return row

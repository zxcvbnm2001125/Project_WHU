import os
import time
import datetime
import random

import numpy as np
import torch
from model import fcn_resnet50
from utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
from utils.my_dataset import WUHDataset

from utils.create_exp_folder import create_exp_folder
from utils.segmentation_transforms import get_transform
from utils.plot_results import plot_training_curves
from utils.metrics_logger import EpochMetricsLogger  # 逐 epoch 指标落盘
import subprocess

# =============================================================================
# 运行配置（W3 · 消融 A：模型对比）
# 直接改下面的数值即可，命令行不传参时全部生效；想临时覆盖仍可用命令行传参。
# 统一口径：WUH 固定子集 train 1184 / val 474 · 输入 512×512 · 20 轮 · bs8 · seed 11
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../XR_1/FCN
_ABLATION = os.path.dirname(os.path.dirname(_HERE))          # .../消融实验

CONFIG = dict(
    weights=os.path.join(_HERE, "weights", "fcn_resnet50_coco.pth"),
    data_path=os.path.join(_ABLATION, r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25"),
    num_classes=1,        # 前景类数（不含背景）；WHU 二分类填 1
    device="cuda",
    batch_size=8,
    epochs=25,
    workers=0,            # Windows 下 >0 反而可能更慢，先保持 0
    lr=1e-4,              # SGD + warmup，本工程 lr 直接生效
    momentum=0.9,
    weight_decay=1e-4,
    amp=True,
    seed=11,
)


def seed_everything(seed=11):
    """固定随机种子。

    本工程上游版本没有任何种子设置，同一份数据两次训练结果不可复现，
    与计划里"固定种子 + 固定子集"的前提冲突，这里补上。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(num_classes, weights):
    # 使用FCN和ResNet50作为骨干网络创建模型
    model = fcn_resnet50(num_classes=num_classes)

    # 如果需要加载预训练权重
    if weights:
        # 加载COCO数据集上训练的FCN模型权重
        weights_dict = torch.load(weights, map_location='cpu')

        # 如果类别数不等于21（COCO数据集的类别数），需要删除与类别相关的权重
        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        # 将加载的权重字典应用到模型中，`strict=False`表示不严格要求权重匹配（允许有部分缺失或多余的权重）
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

        # 如果有缺失的权重或不匹配的权重，打印出来进行调试
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    # 返回创建好的模型
    return model


def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # 输出类似： '1203, 6144\n'
    used, total = map(int, result.strip().split(','))

    return used
    # print(f"GPU 显存占用: {used} MB / {total} MB")


def train(args):
    seed_everything(args.seed)  # 固定随机种子（上游版本缺失，此处补齐）

    batch_size = args.batch_size  # 设置batch size和类别数
    num_classes = args.num_classes + 1  # 类别加上背景类
    train_epoch = args.epochs  # 训练轮次
    num_workers = args.workers  # 计算可用的工作线程数，通常取CPU核心数、batch_size和8中的最小值

    # 选择设备（GPU 如果可用，否则使用 CPU）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 调用函数获取新的exp文件夹和weights文件夹路径
    exp_folder, weights_folder = create_exp_folder()

    # 逐 epoch 指标落盘到 exp_folder/metrics.csv（含 FG IoU，便于判断是否收敛）
    metrics_logger = EpochMetricsLogger(os.path.join(exp_folder, "metrics.csv"))

    # 加载训练数据集, 使用训练时的数据增强（直接读 WUH 原生 train/image|label）
    train_dataset = WUHDataset(args.data_path, transforms=get_transform(train=True), split="train")

    # 加载验证数据集, 验证时无需数据增强
    val_dataset = WUHDataset(args.data_path, transforms=get_transform(train=False), split="val")

    # 加载训练集的DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,  # 打乱数据顺序
                                               pin_memory=True,
                                               # 加速数据加载，将数据加载到 固定内存（pinned memory）中，这对于在使用 GPU 加速时非常有用。
                                               collate_fn=train_dataset.collate_fn)

    # 加载验证集的DataLoader
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,  # 验证集一般使用batch_size为1
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # 创建模型
    model = create_model(num_classes=num_classes, weights=args.weights)
    model.to(device)

    # 准备优化器的参数，这里我们分别为模型的backbone和分类器部分设置参数
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    # 使用SGD优化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # 如果启用混合精度训练（amp），使用GradScaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # scaler = torch.amp.GradScaler(device='cuda') if args.amp else None  # 在pytorch2.7.1版本可以运用该代码可以去除警告信息

    # 学习率调度器，每步更新一次学习率
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # 训练开始
    start_time = time.time()

    # 初始化最优准确率
    best_acc = 0.0  # 最优准确率初始化为0
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")  # 最优模型保存路径
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")  # 最后一轮模型保存路径

    train_losses = []
    val_losses = []
    val_metrics_history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        gpu_used = get_gpu_usage()
        # 每个epoch进行训练
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch, train_epoch, gpu_used,
                               lr_scheduler=lr_scheduler, scaler=scaler)

        train_losses.append(loss)

        # 在验证集上评估模型
        metrics = evaluate(model, val_loader, device=device, num_classes=num_classes)

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        # 选优指标统一为前景（建筑）IoU
        current_acc = float(metrics.get("FG IoU", metrics["Mean Accuracy"]))

        # 更新最优准确率并保存最优模型
        # 保存最优模型
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
            torch.save(model.state_dict(), best_model_path)

        # 保存最后一次模型
        torch.save(model.state_dict(), last_model_path)

        # 逐 epoch 落盘（含 FG IoU），训练中断也不丢历史
        metrics_logger.log(epoch=epoch + 1, lr=optimizer.param_groups[0]["lr"],
                           train_loss=loss, val_loss=metrics["Loss"],
                           val_metrics=metrics, is_best=is_best,
                           epoch_time=time.time() - epoch_start)

    # 打印训练的总时长
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    # 🔥 最后画图
    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    # 以下默认值全部取文件顶部的 CONFIG，改配置只需要改 CONFIG 一处
    parser.add_argument("--weights", default=CONFIG["weights"],
                        help="预训练权重路径（已按脚本所在目录解析为绝对路径）")
    parser.add_argument("--data-path", default=CONFIG["data_path"],
                        help="WUH 数据集根目录（内含 train|val 两个子目录，各含 image 与 label）")
    parser.add_argument("--num-classes", default=CONFIG["num_classes"], type=int,
                        help="前景类数（不含背景）；WHU 二分类填 1")
    parser.add_argument("--device", default=CONFIG["device"], help="training device")
    parser.add_argument("--batch-size", default=CONFIG["batch_size"], type=int)
    parser.add_argument("--epochs", default=CONFIG["epochs"], type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--workers", default=CONFIG["workers"], type=int, metavar="N", help="number of data loading workers (default: 0, meaning data loading runs in main process)")
    parser.add_argument('--lr', default=CONFIG["lr"], type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=CONFIG["momentum"], type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=CONFIG["weight_decay"], type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--seed", default=CONFIG["seed"], type=int, help="随机种子（固定子集复现用）")
    # Mixed precision training parameters
    parser.add_argument("--amp", default=CONFIG["amp"], type=bool, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)

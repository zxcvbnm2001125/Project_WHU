# 导入标准库和第三方库
import os  # 用于操作系统功能，如文件和路径管理
from functools import partial  # 用于部分应用函数

# 导入Numpy和PyTorch相关库
import numpy as np  # 用于数值计算
import torch  # 导入PyTorch库
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.utils.data import DataLoader  # 导入数据加载器模块
import time
import datetime
import subprocess

# 导入自定义模块和模型
from model.unet_resnet import Unet  # 导入U-Net模型
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init  # 导入与U-Net训练相关的函数（如学习率调度、设置优化器学习率和权重初始化）
from utils.dataloader import UnetDataset, unet_dataset_collate  # 导入U-Net数据集及其合并函数
from utils.utils import seed_everything, worker_init_fn, get_lr  # 导入一些工具函数（如随机种子设置、展示配置和初始化工作线程）
from utils.train_and_eval import train_one_epoch, evaluate  # 导入训练一个epoch的函数

from utils.create_exp_folder import create_exp_folder  # 用于创建实验目录
from utils.plot_results import plot_training_curves  # 绘制模型结果图
from utils.metrics_logger import EpochMetricsLogger  # 逐 epoch 指标落盘

# =============================================================================
# 运行配置（W3 · 消融 A：模型对比）
# 直接改下面的数值即可，命令行不传参时全部生效；想临时覆盖仍可用命令行传参。
# 统一口径：WUH 固定子集 train 1184 / val 474 · 输入 512×512 · 25 轮 · bs8 · seed 11
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../XR_1/U-Net
_ABLATION = os.path.dirname(os.path.dirname(_HERE))          # .../消融实验

CONFIG = dict(
    weights=os.path.join(_HERE, "weights", "unet_resnet_voc.pth"),
    data_path=os.path.join(_ABLATION, r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25"),
    num_classes=1,        # 前景类数（不含背景）；WHU 二分类填 1
    device="cuda",
    batch_size=8,
    epochs=25,
    workers=0,            # Windows 下 >0 反而可能更慢，先保持 0
    lr=1e-4,              # 注意：get_optimizer_and_lr() 内把 lr 钳位为 1e-4，此项实际不生效
    momentum=0.9,
    weight_decay=1e-4,
    amp=True,
    seed=11,
    dice_loss=False,       # True = CE+Dice（上游默认）。消融 A 若要求三模型同损失，须改 False
    focal_loss=False,
)

# GPU占用计算函数
def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # 输出类似： '1203, 6144\n'
    used, total = map(int, result.strip().split(','))

    return used
    # print(f"GPU 显存占用: {used} MB / {total} MB")


def create_model(num_classes, weights):
    # 创建一个Unet模型实例，num_classes指定了分类的数量（输出通道数）
    model = Unet(num_classes=num_classes)
    # 使用 weights_init 函数初始化模型权重
    weights_init(model)

    # 检查是否提供了模型路径，即是否需要加载预训练权重
    if weights:
        # 获取当前模型的状态字典（即模型的参数）
        model_dict = model.state_dict()

        # 加载预训练模型的权重字典
        pretrained_dict = torch.load(weights, map_location='cpu')

        # 初始化三个列表和字典，用于存储加载和未加载的参数
        load_key, no_load_key, temp_dict = [], [], {}

        # 遍历预训练模型中的每个参数
        for k, v in pretrained_dict.items():
            # 如果当前模型中有对应的参数，并且形状匹配，则加载该参数
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v  # 将匹配的参数存储到临时字典中
                load_key.append(k)  # 记录成功加载的参数的Key
            else:
                no_load_key.append(k)  # 记录没有匹配上的参数的Key

        # 更新模型的状态字典，将预训练模型的参数加载到当前模型中
        model_dict.update(temp_dict)
        # 加载更新后的状态字典到模型中
        model.load_state_dict(model_dict)
    return model





def get_optimizer_and_lr(model, batch_size, train_epoch, momentum, weight_decay):
    # 初始化学习率（初始学习率为1e-4）
    Init_lr = 1e-4
    # 最小学习率是初始学习率的1%
    Min_lr = Init_lr * 0.01

    # 设置学习率衰减策略，'cos'代表余弦衰减
    lr_decay_type = 'cos'

    # 默认每个小批量（batch）的大小为16
    nbs = 16
    # 设置最大学习率限制
    lr_limit_max = 1e-4
    # 设置最小学习率限制
    lr_limit_min = 1e-4

    # 根据当前batch_size调整学习率，并确保它在最大和最小限制之间
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    # 根据当前batch_size调整最小学习率，并确保它在最大和最小限制之间
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 使用Adam优化器，设置学习率、动量、权重衰减等参数
    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay)
    # 获取学习率调度器函数，根据衰减类型、初始学习率、最小学习率和训练轮次来计算学习率的变化
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, train_epoch)
    # 返回优化器和学习率调度器
    return optimizer, lr_scheduler_func


def train(args):
    seed_everything(args.seed)  # 设置种子

    num_classes = args.num_classes + 1  # 类别加上背景类
    train_epoch = args.epochs  # 训练轮次
    batch_size = args.batch_size  # 设置batch size和类别数
    num_workers = args.workers  # 计算可用的工作线程数，通常取CPU核心数、batch_size和8中的最小值

    # 选择设备（GPU 如果可用，否则使用 CPU）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 调用函数获取新的exp文件夹和weights文件夹路径
    exp_folder, weights_folder = create_exp_folder()

    # 逐 epoch 指标落盘到 exp_folder/metrics.csv（含 FG IoU，便于判断是否收敛）
    metrics_logger = EpochMetricsLogger(os.path.join(exp_folder, "metrics.csv"))

    # 这次跑的是 CE 还是 CE+Dice 必须留下来，否则两个 exp 目录事后分不清
    loss_mode = "CE+Dice" if args.dice_loss else "CE"
    print(f"损失函数: {loss_mode}")
    with open(os.path.join(exp_folder, "train_config.txt"), "w", encoding="utf-8") as f:
        f.write(f"loss_mode: {loss_mode}\n")
        f.write(f"dice_loss: {args.dice_loss}\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    input_shape = [512, 512]  # 一定要是32的整数倍

    # 创建训练数据集对象（直接读 WUH 原生 train/image|label，按 split 指定划分）
    # args.data_path: 数据集根路径， input_shape: 输入图像的尺，num_classes: 输出类别数
    # augmentation=True: 是否采用数据增强，split="train": 指定划分（train / val）
    train_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=True, split="train")
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, split="val")

    # 加载训练集的DataLoader
    train_loader = DataLoader(train_dataset,
                              shuffle=True,  # 是否打乱数据（训练时一般打乱数据）
                              batch_size=batch_size,  # 每个批次加载的样本数量
                              num_workers=num_workers,  # 加载数据时使用的子进程数量（并行化加载）
                              pin_memory=True,  # 是否将数据复制到CUDA的内存中（如果使用GPU训练，通常设置为True）
                              drop_last=False,  # 如果数据集大小不能被batch_size整除，是否丢弃最后不完整的批次
                              collate_fn=unet_dataset_collate,  # 定义如何将多个样本合并成一个批次，通常是处理不同大小的图像
                              sampler=None,  # 是否使用自定义采样器，默认为None，表示按顺序加载
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))  # 初始化worker时的函数，通常用于设置随机种子

    # 加载验证集的DataLoader
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=unet_dataset_collate,
                            sampler=None,
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))

    # 创建模型
    model = create_model(num_classes=num_classes, weights=args.weights)

    # 如果启用混合精度训练（amp），使用GradScaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # scaler = torch.amp.GradScaler(device='cuda') if args.amp else None  # 在pytorch2.7.1版本可以运用该代码可以去除警告信息

    # 获取优化器和学习率调度器
    optimizer, lr_scheduler_func = get_optimizer_and_lr(model, batch_size, train_epoch,  args.momentum, args.weight_decay)

    # 训练开始
    start_time = time.time()

    best_acc = 0.0  # 最优准确率初始化为0
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")  # 最优模型保存路径
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")  # 最后一轮模型保存路径

    train_losses = []
    val_losses = []
    val_metrics_history = []

    # 是否使用focal loss来防止正负样本不平衡，是否给不同种类赋予不同的损失权值，默认是平衡的。
    # 两项均取自文件顶部 CONFIG，避免命令行布尔参数（type=bool）解析歧义。
    focal_loss = CONFIG["focal_loss"]
    #  种类少（几类）时，设置为True
    #  种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #  种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = args.dice_loss

    #   开始模型训练
    for epoch in range(train_epoch):
        epoch_start = time.time()
        gpu_used = get_gpu_usage() # 计算使用GPU内存
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)  # 学习率调度函数
        lr_now = get_lr(optimizer)  # 本 epoch 实际使用的学习率

        # 每个epoch进行训练
        loss = train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                               gpu_used, num_classes, scaler, epoch, train_epoch)

        train_losses.append(loss)  # 保存训练过程中的loss值

        # 在验证集上评估模型
        metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes)

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        # 选优指标统一为前景（建筑）IoU：二分类下 Mean Accuracy 会被背景主导
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
        metrics_logger.log(epoch=epoch + 1, lr=lr_now, train_loss=loss,
                           val_loss=metrics["Loss"], val_metrics=metrics,
                           is_best=is_best, epoch_time=time.time() - epoch_start)

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
    parser.add_argument("--workers", default=CONFIG["workers"], type=int, metavar="N",
                        help="number of data loading workers (default: 0, meaning data loading runs in main process)")
    parser.add_argument('--lr', default=CONFIG["lr"], type=float,
                        help='initial learning rate（本工程内会被 get_optimizer_and_lr 钳位，实际不生效）')
    parser.add_argument('--momentum', default=CONFIG["momentum"], type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=CONFIG["weight_decay"], type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--seed", default=CONFIG["seed"], type=int, help="随机种子（固定子集复现用）")
    # Mixed precision training parameters
    parser.add_argument("--amp", default=CONFIG["amp"], type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--dice_loss", action=argparse.BooleanOptionalAction, default=CONFIG["dice_loss"],
                        help="消融 B 开关：开启则 CE + Dice（权重 1:1），关闭则纯 CE")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
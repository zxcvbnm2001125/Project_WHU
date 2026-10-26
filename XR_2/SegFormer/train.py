# -*- coding: utf-8 -*-
"""
SegFormer 训练脚本（W3 · 消融 A 的第四个模型）。

本文件是 Project_WUD/SegFormer 的精简版：只保留纯 SegFormer（MiT 主干 + All-MLP 解码器）
与训练/验证/落盘必需的部分，去掉了后续迭代加上、消融对比并不需要的模块：

    去掉  1/2 分辨率精修（HighResRefine）、8 方向 TTA / 多尺度推理、
          梯度累积（accum_steps）、冻结主干 warmup（freeze_epochs）、
          增强版本切换（aug v1/v3）、归一化切换（imagenet/unit）、
          从 train 随机切验证集（val-ratio）、LoveDA 多类任务相关配置

    保留  断点续训、逐轮 train_history.csv、按前景 IoU 选优、layer-wise lr decay

数据口径与 XR_1 下另外三个工程一致：直接读 WUH_sub25 的 <split>/image|label，
train 1184 / val 474，输入 512，bs 8，20 轮，seed 11。

与三个 CNN 工程的配方差异（这是架构决定的，不是可选项）：
    AdamW + layer_decay=0.9 + bf16 autocast；wd=0.01；lr=6e-5
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.segformer import SegFormer
from model.unet_training import get_lr_scheduler, set_optimizer_lr
from utils.create_exp_folder import create_exp_folder
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.plot_results import plot_training_curves
from utils.train_and_eval import GradScaler, evaluate, train_one_epoch
from utils.utils import (compute_class_weights, load_weights_partial,
                         seed_everything, validate_input_size, worker_init_fn)

# =============================================================================
# 运行配置（W3 · 消融 A）
# 直接改下面的数值即可，命令行不传参时全部生效；想临时覆盖仍可用命令行传参。
# 统一口径：WUH 固定子集 train 1184 / val 474 · 输入 512×512 · 20 轮 · bs8 · seed 11
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))            # .../XR_1/SegFormer
_ABLATION = os.path.dirname(os.path.dirname(_HERE))          # .../消融实验

CONFIG = dict(
    variant="b3",
    weights=r"weights\mit_b3_imagenet.pth",                                                  # 整模型权重（热启动），留空=从主干初始化
    backbone_weights=r"weights\mit_b3_imagenet.pth",                                         # 留空则按 variant 自动找 weights/mit_{variant}_imagenet.pth
    data_path=os.path.join(_ABLATION, r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25"),
    num_classes=1,          # 前景类数（不含背景），程序内部 +1
    decoder_dim=256,        # 官方 256，128 更省显存
    dropout=0.1,
    drop_path=0.1,
    input_shape=512,
    device="cuda",
    batch_size=4,
    epochs=25,
    workers=0,              # Windows 下建议 0
    lr=6e-5,
    min_lr_ratio=0.01,
    wd=0.01,
    layer_decay=0.9,
    beta1=0.9,
    beta2=0.999,
    precision="bf16",       # bf16 无需 GradScaler，Transformer 首选
    max_grad_norm=1.0,
    cls_weights="1,3",      # 背景:建筑 = 1:3；留空则按训练集像素频率自动算
    dice=False,               # 是否叠加 Dice Loss（默认关闭，与姊妹项目口径一致）
    focal=False,
    best_metric="fg_iou",   # 二分类建筑提取，按前景 IoU 选优
    seed=11,
    deterministic=False,
    resume=r"",
)


# ------------------------------------------------------------------ #
#   小工具
# ------------------------------------------------------------------ #
def get_gpu_usage():
    """
    查询当前显存占用（MB）。

    只是用于训练日志里显示，取不到就返回 0，绝不能因为显存查询失败而中断训练。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=2)
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return 0.0


def parse_cls_weights(spec, num_classes):
    """
    解析 --cls-weights 并校验长度，训练开始前就报错。

    为什么需要这个校验：
        PyTorch 的 CrossEntropyLoss 在权重个数与类别数不一致时只会抛
        "weight tensor should be defined either for all N classes or no classes"，
        看不出是命令行少写了几个值；而且它是在数据加载完、模型建完、
        跑到第一个 batch 才炸。这里提前拦住。
    """
    values = [w.strip() for w in str(spec).split(",") if w.strip()]
    try:
        weights = np.array([float(w) for w in values], dtype=np.float32)
    except ValueError:
        raise SystemExit(f"--cls-weights 解析失败：{spec!r}（应为逗号分隔的数字）")

    if len(weights) != num_classes:
        raise SystemExit(
            f"--cls-weights 给了 {len(weights)} 个值，但当前类别数是 {num_classes}"
            f"（= --num-classes {num_classes - 1} + 背景）。\n"
            f"  WHU 二分类应写：--cls-weights \"1,3\"\n"
            f"  或干脆留空，按训练集像素频率自动计算")

    if (weights <= 0).any():
        raise SystemExit(f"--cls-weights 必须全为正数，当前为 {weights.tolist()}")
    return weights


def capture_rng_state():
    """把四路随机数状态打包存起来，续训时恢复，保证增强序列可复现。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """恢复 capture_rng_state() 保存的随机数状态（缺项时静默跳过）。"""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scaler, epoch, best_iou, best_epoch,
                    train_losses, val_losses, val_metrics_history, exp_folder,
                    args, schedule_epochs):
    """
    保存断点。

    先写 .tmp 再 os.replace 原子替换：训练被 Ctrl+C 打断或断电时，
    不会留下一个"写了一半"的 checkpoint 把上次的完好存档覆盖掉。
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics_history": val_metrics_history,
        "exp_folder": exp_folder,
        "args": vars(args),
        "schedule_epochs": schedule_epochs,
        "rng_state": capture_rng_state(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    """读取断点文件（weights_only=False，因为里面含 numpy 随机状态等非张量对象）。"""
    return torch.load(path, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------ #
#   逐轮训练记录（每轮重写一次 train_history.csv）
# ------------------------------------------------------------------ #
HISTORY_FIELDS = [
    "epoch", "timestamp", "epoch_time_s", "lr_decoder", "lr_backbone_min",
    "train_loss", "val_loss", "pixel_acc", "mean_acc", "mean_iou",
    "fw_iou", "fg_iou", "iou_bg", "iou_class1", "is_best",
]


def _fmt_cell(value, field=""):
    """
    把单元格格式化成短字符串。

    - 学习率用科学计数法：min_lr 只有 6e-7 量级，固定 6 位小数会被压成 0.000001；
    - 其余浮点保留 6 位小数，够区分 0.0001 级别的指标差异；
    - None / 空串写空（续训复原的轮次拿不到 lr 与耗时）。
    """
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if field in ("lr_decoder", "lr_backbone_min"):
            return f"{value:.3e}"
        return f"{value:.6f}"
    return value


def make_history_record(epoch, metrics, train_loss=None, lr_decoder=None,
                        lr_backbone_min=None, epoch_time=None, is_best=0,
                        timestamp=""):
    """组装一轮的训练记录；epoch 传 0 基，写入文件时统一 +1。"""
    ious = list(metrics.get("IoU per class", []))
    return {
        "epoch": epoch + 1,
        "timestamp": timestamp,
        "epoch_time_s": epoch_time,
        "lr_decoder": lr_decoder,
        "lr_backbone_min": lr_backbone_min,
        "train_loss": train_loss,
        "val_loss": metrics.get("Loss"),
        "pixel_acc": metrics.get("Pixel Accuracy"),
        "mean_acc": metrics.get("Mean Accuracy"),
        "mean_iou": metrics.get("Mean IoU"),
        "fw_iou": metrics.get("Frequency Weighted IoU"),
        "fg_iou": metrics.get("Foreground IoU"),
        # 逐类 IoU：0=背景，1=第一个前景类。
        # fg_iou 是"前景类平均"，iou_class1 只是"类别 1"的单类 IoU，两者列名分开以免混淆。
        "iou_bg": ious[0] if len(ious) > 0 else None,
        "iou_class1": ious[1] if len(ious) > 1 else None,
        "is_best": is_best,
    }


def rebuild_history_from_checkpoint(ckpt):
    """
    续训时用存档复原前面几轮的记录，保证最终文件覆盖从第 1 轮起的全程。

    train_losses 与 val_metrics_history 都完整存在 checkpoint 里，可直接重建；
    lr 与单轮耗时当时没有记录，只能留空（不影响指标比对）。
    """
    train_losses = ckpt.get("train_losses", []) or []
    metrics_history = ckpt.get("val_metrics_history", []) or []

    records = []
    for i, metrics in enumerate(metrics_history):
        records.append(make_history_record(
            epoch=i,
            metrics=metrics,
            train_loss=train_losses[i] if i < len(train_losses) else None,
        ))
    return records


def write_training_history(records, exp_folder, filename="train_history.csv"):
    """
    把逐轮记录写成 CSV（utf-8-sig，Excel 双击打开不乱码，pandas 也能直接读）。

    每轮重写一次（先写 .tmp 再原子替换），训练被中断也能保留到最后一轮。
    """
    path = os.path.join(exp_folder, filename)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: _fmt_cell(record.get(k), k) for k in HISTORY_FIELDS})
    os.replace(tmp_path, path)
    return path


def cli_provided_flags(argv):
    """
    从命令行里提取"用户显式写了哪些 --flag"。

    续训时用它来决定：用户没提到的参数沿用存档值，提到了就用命令行值。
    支持 "--flag" 和 "--flag=value" 两种写法。
    """
    flags = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0]
        if name:
            flags.add(name.replace("-", "_"))
    return flags


# ------------------------------------------------------------------ #
#   模型 / 优化器
# ------------------------------------------------------------------ #
def resolve_backbone_weights(args):
    """
    确定 MiT 主干预训练权重的路径。

    优先级：命令行 --backbone-weights > weights/mit_{variant}_imagenet.pth
    返回 None 表示没有可用的预训练权重（从零训练）。
    """
    if args.backbone_weights:
        return args.backbone_weights
    default_path = os.path.join(_HERE, "weights", f"mit_{args.variant.lower()}_imagenet.pth")
    if os.path.isfile(default_path):
        return default_path
    return None


def create_model(args, num_classes):
    """构造纯 SegFormer，并按需加载主干预训练权重。"""
    model = SegFormer(variant=args.variant,
                      num_classes=num_classes,
                      decoder_dim=args.decoder_dim,
                      dropout=args.dropout,
                      drop_path_rate=args.drop_path,
                      img_size=args.input_shape)

    backbone_weights = resolve_backbone_weights(args)
    if backbone_weights is None:
        print("[提示] 没有找到 MiT 主干预训练权重，主干将从随机初始化开始训练。")
        print("       建议先运行 python download_pretrained.py 下载 ImageNet 权重，")
        print("       否则 Transformer 在小数据集上从零训练几乎不可能收敛到好结果。")
    elif not os.path.isfile(backbone_weights):
        print(f"[警告] 指定的主干权重不存在：{backbone_weights}，主干将从随机初始化开始训练。")
    else:
        model.backbone.load_pretrained(backbone_weights)

    # 主干加载完之后再初始化分割头，避免把预训练参数覆盖掉
    model.init_head_weights()

    # 用户另外指定的整模型权重（热启动），优先级最高
    if args.weights and os.path.isfile(args.weights):
        print(f"[微调] 从整模型权重初始化：{args.weights}")
        load_weights_partial(model, args.weights)
    elif args.weights:
        # 静默降级是高危点：路径写错（如反斜杠被 shell 转义成 \t、\b）会让训练
        # 退化成从主干初始化，而日志里毫无提示。
        print("[微调][警告] --weights 指定的文件不存在，已跳过整模型权重加载：")
        print(f"           {args.weights!r}")
        print("           路径建议用正斜杠 /；若本意就是从主干初始化，请留空。")

    return model


def build_optimizer_and_scheduler(model, args, train_epoch):
    """
    AdamW + 逐层学习率衰减 + 余弦退火。

    参数组由 model.param_groups_with_layer_decay() 生成，每个组自带 'lr_scale'，
    set_optimizer_lr() 会把它乘到本轮的基础学习率上。
    """
    param_groups = model.param_groups_with_layer_decay(
        base_lr=args.lr,
        weight_decay=args.wd,
        layer_decay=args.layer_decay,
        verbose=True,
    )
    optimizer = torch.optim.AdamW(param_groups,
                                  lr=args.lr,
                                  betas=(args.beta1, args.beta2),
                                  eps=1e-8)

    min_lr = args.lr * args.min_lr_ratio
    lr_scheduler_func = get_lr_scheduler("cos", args.lr, min_lr, train_epoch)
    return optimizer, lr_scheduler_func, min_lr


# ------------------------------------------------------------------ #
#   训练主流程
# ------------------------------------------------------------------ #
def restore_args_from_checkpoint(args):
    """读取断点，并把存档里的训练配置恢复回 args。

    规则：命令行显式给出的参数优先，其余一律沿用存档值。

    必须放在建模型之前：只写 `--resume xxx.pth` 不带 --variant 时，
    args.variant 会停在默认值；存档若是别的规格，模型结构与权重对不上，
    load_state_dict 会直接报 size mismatch。
    """
    if not args.resume:
        return None
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"断点文件不存在：{args.resume}")

    ckpt = load_checkpoint(args.resume)
    provided = cli_provided_flags(sys.argv[1:])
    saved_args = ckpt.get("args", {})
    overridden = []
    for key, value in saved_args.items():
        if key == "resume" or key in provided or not hasattr(args, key):
            continue
        if getattr(args, key) != value:
            overridden.append(key)
        setattr(args, key, value)
    if overridden:
        print(f"[续训] 以下参数沿用存档值（想覆盖请在命令行显式给出）："
              f"{', '.join(sorted(overridden))}")
    return ckpt


def train(args):
    # 断点续训：先把训练配置恢复回来，**再**建模型/数据集
    ckpt = restore_args_from_checkpoint(args)

    # 输入尺寸必须是 32 的整数倍：MiT 四个 stage 累计下采样 32 倍，
    # 且第一级高效注意力的降采样卷积要求 1/4 特征图边长能被 8 整除
    side = validate_input_size(args.input_shape, 32, "--input-shape")
    input_shape = [side, side]

    num_classes = args.num_classes + 1        # 加上背景类
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 精度模式：bf16 无需 GradScaler；fp16 需要；fp32 都不需要
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("[警告] 当前显卡不支持 bf16，自动退回 fp32。")
        args.precision = "fp32"

    # ---------------- 数据集 ----------------
    # 直接读 WUH 原生目录；验证集固定用官方 val 划分，不从 train 切
    train_dataset = UnetDataset(args.data_path, input_shape, num_classes,
                                augmentation=True, split="train")
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes,
                              augmentation=False, split="val")

    train_size = len(train_dataset)
    val_size = len(val_dataset)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=unet_dataset_collate,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=unet_dataset_collate,
                            worker_init_fn=worker_init_fn)

    print(f"训练集 {train_size} 张 | 验证集 {val_size} 张（train/ 与 val/ 两个原生划分）")

    # ---------------- 数据自检（防止标签被二值化这类静默错误）----------------
    seen_classes = train_dataset.observed_classes(None, max_samples=50)
    print(f"数据自检：抽样 50 张，标签中出现的类别 {seen_classes}"
          f"（期望范围 0..{num_classes - 1}，{num_classes} 为忽略）")

    # ---------------- 类别权重 ----------------
    if args.cls_weights:
        cls_weights = parse_cls_weights(args.cls_weights, num_classes)
        print(f"类别权重（命令行指定）: {cls_weights.tolist()}")
    else:
        counts = train_dataset.class_pixel_counts(None)
        cls_weights = compute_class_weights(counts)
        print(f"类别像素统计（未指定 --cls-weights，按频率自动计算）: {counts.tolist()}")
        print(f"→ 反频率权重 {cls_weights.tolist()}")

    # ---------------- 模型 / 优化器 ----------------
    model = create_model(args, num_classes)
    model.to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型: SegFormer-{args.variant.upper()} | 参数量 {n_param / 1e6:.2f}M | 输入 {side}x{side}")
    # 消融 B：CE 与 CE+Dice 两次运行要能一眼区分
    print(f"损失函数: {'CE+Dice' if args.dice else 'CE'}"
          f"{' + Focal' if args.focal else ''} | 类别权重 {cls_weights.tolist()}")

    start_epoch = 0
    best_iou, best_epoch = 0.0, -1
    train_losses, val_losses, val_metrics_history = [], [], []
    epoch_records = []          # 逐轮记录，每轮重写 exp_folder/train_history.csv
    schedule_epochs = args.epochs
    resume_optimizer_state = None
    resume_scaler_state = None
    exp_folder = None

    # ---------------- 断点续训：恢复训练状态 ----------------
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])

        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0.0)
        best_epoch = ckpt.get("best_epoch", -1)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        val_metrics_history = ckpt.get("val_metrics_history", [])
        resume_optimizer_state = ckpt.get("optimizer")
        resume_scaler_state = ckpt.get("scaler")
        # 余弦调度的总轮次沿用存档值，否则续训会把 lr 曲线重置
        schedule_epochs = ckpt.get("schedule_epochs", args.epochs)
        restore_rng_state(ckpt.get("rng_state"))
        # 用存档里的历史复原前几轮的记录，最终 CSV 覆盖从第 1 轮起的全程
        epoch_records = rebuild_history_from_checkpoint(ckpt)

        print(f"从 {args.resume} 续训：已完成 {start_epoch}/{schedule_epochs} 轮，"
              f"当前最佳(存档口径)={best_iou:.4f}@{best_epoch}")

    # ---------------- 实验目录 ----------------
    if exp_folder is None:
        if ckpt is not None and ckpt.get("exp_folder") and os.path.isdir(ckpt.get("exp_folder")):
            # 续训写回原来的实验目录，日志和曲线不会被拆成两份
            exp_folder = ckpt["exp_folder"]
            weights_folder = os.path.join(exp_folder, "weights")
            os.makedirs(weights_folder, exist_ok=True)
        else:
            exp_folder, weights_folder = create_exp_folder()

    print(f"实验目录: {exp_folder}")

    # ---------------- 优化器（必须在续训读参之后建，否则 lr 配置是旧的）----------------
    optimizer, lr_scheduler_func, min_lr = build_optimizer_and_scheduler(model, args, schedule_epochs)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print("[续训] 优化器状态已恢复")
        except Exception as exc:  # 参数组数量变了（比如改了 layer_decay）就跳过
            print(f"[续训][警告] 优化器状态恢复失败，已改用新优化器：{exc}")

    scaler = None
    amp_dtype = None
    if args.precision == "fp16":
        scaler = GradScaler()
        amp_dtype = torch.float16
        if resume_scaler_state is not None:
            try:
                scaler.load_state_dict(resume_scaler_state)
            except Exception:
                pass
    elif args.precision == "bf16":
        amp_dtype = torch.bfloat16   # bf16 不需要 loss scaling
    print(f"精度: {args.precision} | 基础 lr={args.lr} | min_lr={min_lr:.2e} "
          f"| wd={args.wd} | layer_decay={args.layer_decay}")

    # ---------------- 训练配置落盘 ----------------
    config_path = os.path.join(exp_folder, "train_config.txt")
    mode = "a" if args.resume and os.path.exists(config_path) else "w"
    with open(config_path, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write(f"\n--- 续训 resume @ {datetime.now():%Y-%m-%d %H:%M:%S} ---\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        f.write(f"num_classes(含背景): {num_classes}\n")
        f.write(f"classes_weight: {cls_weights.tolist()}\n")
        f.write(f"train_size: {train_size}\n")
        f.write(f"val_size: {val_size}\n")
        f.write(f"trainable_params: {n_param}\n")

    # ---------------- 主循环 ----------------
    for epoch in range(start_epoch, schedule_epochs):
        epoch_start = time.time()

        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # 记录本轮实际生效的学习率：解码器（最大）与最浅主干层（最小）
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        lr_now_max, lr_now_min = max(lrs), min(lrs)

        train_loss = train_one_epoch(model, optimizer, train_loader, device,
                                     args.dice, args.focal, gpu_used, num_classes,
                                     scaler, epoch, schedule_epochs,
                                     max_grad_norm=args.max_grad_norm,
                                     cls_weights=cls_weights,
                                     amp_dtype=amp_dtype)

        metrics = evaluate(model, val_loader, device, args.dice, args.focal,
                           num_classes, cls_weights=cls_weights)

        train_losses.append(train_loss)
        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        # 选优指标：二分类建筑按前景 IoU；多类任务可改用 mean_iou
        current = metrics["Foreground IoU"] if args.best_metric == "fg_iou" else metrics["Mean IoU"]
        improved = current > best_iou
        if improved:
            best_iou, best_epoch = current, epoch
            torch.save(model.state_dict(), os.path.join(weights_folder, f"best_model_{num_classes}.pth"))

        torch.save(model.state_dict(), os.path.join(weights_folder, f"last_model_{num_classes}.pth"))
        save_checkpoint(os.path.join(weights_folder, f"checkpoint_{num_classes}.pth"),
                        model, optimizer, scaler, epoch, best_iou, best_epoch,
                        train_losses, val_losses, val_metrics_history,
                        exp_folder, args, schedule_epochs)

        # 逐轮记录：每轮重写一次 CSV，训练被中断也能保留到最后一轮
        epoch_records.append(make_history_record(
            epoch=epoch, metrics=metrics, train_loss=train_loss,
            lr_decoder=lr_now_max, lr_backbone_min=lr_now_min,
            epoch_time=time.time() - epoch_start, is_best=int(improved),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        write_training_history(epoch_records, exp_folder)

        print(f"Epoch {epoch + 1}/{schedule_epochs} 完成 | "
              f"MeanIoU {metrics['Mean IoU']:.4f} | "
              f"前景IoU {metrics['Foreground IoU']:.4f} | "
              f"MeanAcc {metrics['Mean Accuracy']:.4f} | "
              f"PixelAcc {metrics['Pixel Accuracy']:.4f} | "
              f"Best({args.best_metric}) {best_iou:.4f} @ epoch {best_epoch + 1}")

    # ---------------- 收尾 ----------------
    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)
    history_path = write_training_history(epoch_records, exp_folder)
    print(f"训练结束，最佳 {args.best_metric} = {best_iou:.4f} @ epoch {best_epoch + 1}")
    print(f"权重与曲线: {weights_folder}")
    print(f"逐轮记录: {history_path}（{len(epoch_records)} 轮，Excel / pandas 可直接读）")
    return best_iou


# ------------------------------------------------------------------ #
#   参数（默认值全部取自文件顶部的 CONFIG）
# ------------------------------------------------------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description="SegFormer 语义分割训练（W3 消融 A）")

    # ---- 模型 ----
    parser.add_argument("--variant", default=CONFIG["variant"], type=str,
                        choices=["b0", "b1", "b2", "b3"],
                        help="SegFormer 规格。8GB 显存建议 b0/b1；b2 需把 batch-size 降到 4")
    parser.add_argument("--num-classes", default=CONFIG["num_classes"], type=int,
                        help="前景类别数（不含背景，程序内部会自动 +1）。WHU 建筑填 1")
    parser.add_argument("--decoder-dim", default=CONFIG["decoder_dim"], type=int,
                        help="解码器统一通道数（官方 256，这里 128 更省显存）")
    parser.add_argument("--dropout", default=CONFIG["dropout"], type=float, help="解码器 dropout")
    parser.add_argument("--drop-path", default=CONFIG["drop_path"], type=float,
                        help="主干随机深度上限（官方分割训练用 0.1）")
    parser.add_argument("--backbone-weights", default=CONFIG["backbone_weights"], type=str,
                        help="MiT 主干 ImageNet 预训练权重；留空则按 --variant 自动找 "
                             "weights/mit_{variant}_imagenet.pth")
    parser.add_argument("--weights", default=CONFIG["weights"], type=str,
                        help="整模型权重（热启动场景）；留空表示从主干初始化。"
                             "注意整模型权重与当前 --num-classes 必须一致。路径请用正斜杠")

    # ---- 数据 ----
    parser.add_argument("--data-path", default=CONFIG["data_path"], type=str,
                        help="WUH 数据集根目录（内含 train|val 两个子目录，各含 image 与 label）")
    parser.add_argument("--input-shape", default=CONFIG["input_shape"], type=int,
                        help="网络输入边长，必须是 32 的整数倍")

    # ---- 训练 ----
    parser.add_argument("--device", default=CONFIG["device"], type=str)
    parser.add_argument("--batch-size", default=CONFIG["batch_size"], type=int)
    parser.add_argument("--epochs", default=CONFIG["epochs"], type=int,
                        help="训练轮数。每轮把训练集所有图各取一次")
    parser.add_argument("--workers", default=CONFIG["workers"], type=int, help="Windows 下建议 0")
    parser.add_argument("--lr", default=CONFIG["lr"], type=float,
                        help="基础学习率（解码器用满值，主干按 layer_decay 逐层衰减）")
    parser.add_argument("--min-lr-ratio", default=CONFIG["min_lr_ratio"], type=float,
                        help="最小学习率 = lr × 该比例")
    parser.add_argument("--wd", "--weight-decay", default=CONFIG["wd"], type=float, dest="wd",
                        help="解耦权重衰减（Transformer 常用 0.01，比 CNN 的 1e-4 大）")
    parser.add_argument("--layer-decay", default=CONFIG["layer_decay"], type=float,
                        help="逐层学习率衰减系数（1.0 表示关闭，退化成统一学习率）")
    parser.add_argument("--beta1", default=CONFIG["beta1"], type=float)
    parser.add_argument("--beta2", default=CONFIG["beta2"], type=float)
    parser.add_argument("--precision", default=CONFIG["precision"], type=str,
                        choices=["bf16", "fp16", "fp32"],
                        help="混合精度模式。bf16 无需 GradScaler，Transformer 首选")
    parser.add_argument("--max-grad-norm", default=CONFIG["max_grad_norm"], type=float,
                        help="梯度裁剪阈值，<=0 表示不裁剪")

    # ---- 损失 ----
    parser.add_argument("--cls-weights", default=CONFIG["cls_weights"], type=str,
                        help='各类别权重，逗号分隔，长度必须等于类别数（含背景）。'
                             'WHU 二分类示例 "1,3"；留空则按训练集像素频率自动计算')
    parser.add_argument("--dice", default=CONFIG["dice"], action=argparse.BooleanOptionalAction,
                        help="是否叠加 Dice Loss（默认开启，与姊妹项目口径一致）")
    parser.add_argument("--focal", default=CONFIG["focal"], action=argparse.BooleanOptionalAction,
                        help="是否用 Focal Loss 代替交叉熵")

    # ---- 其他 ----
    parser.add_argument("--best-metric", default=CONFIG["best_metric"], type=str,
                        choices=["mean_iou", "fg_iou"],
                        help="选优指标。二分类建筑提取用 fg_iou（只算建筑类，避免背景把数值注水）；"
                             "多类任务用 mean_iou")
    parser.add_argument("--seed", default=CONFIG["seed"], type=int)
    parser.add_argument("--deterministic", action="store_true", default=CONFIG["deterministic"],
                        help="启用 cudnn 确定性模式（可逐位复现，但慢 10~20%%）")
    parser.add_argument("--resume", default=CONFIG["resume"], type=str,
                        help="断点文件路径；留空表示从头训练")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 种子要在建数据集之前设好，否则增强的随机序列不可复现
    seed_everything(args.seed, deterministic=args.deterministic)
    start_time = time.time()
    train(args)
    print(f"总耗时: {(time.time() - start_time) / 60:.1f} 分钟")

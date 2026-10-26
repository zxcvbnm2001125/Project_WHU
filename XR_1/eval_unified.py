# -*- coding: utf-8 -*-
"""
四模型统一评测（W3 · 消融 A）

为什么需要这个脚本
------------------
三个上游工程（U-Net / FCN / DeepLabV3+）的 evaluate() 口径互不相同：

  U-Net / FCN  按 batch 求指标再求和平均（单批内的均值，再跨批平均）
  DeepLab      用 StreamSegMetrics 的全局混淆矩阵，全验证集汇总后再算

而验证集 474 张里有 137 张完全没有建筑（fg_ratio = 0），这会放大差异：

  - FCN 的验证 batch 写死为 1，空图的 foreground_iou 因 union==0 直接返回 0.0，
    137/474 = 28.9% 的图逐张记 0 分 → FG IoU 被压到约 0.71 倍
  - U-Net 验证 batch=8，很少有整批全空，但 mean_iou 遇到空图时只剩背景类参与平均，
    单张算出来接近 0.97，反而把这批图抬成高分
  - DeepLab 的全局混淆矩阵里，空图 0/0 被 nanmean 跳过、误检正常计入 FP，口径正确

结论：三份训练日志里的数字不能直接拼成一张对比表。

本脚本的做法
------------
训练代码一行不改（保持上游原版），训练结束后用本脚本对四个模型的 best 权重
在同一个 474 张验证集上重算全部指标，统一用全局混淆矩阵：

  PA / mIoU / FG IoU / mAcc / FWIoU
  参数量
  推理延迟（bs=1）、推理吞吐（bs=8）、推理显存峰值

预处理仍走各工程自己的验证管线（缩放、归一化都不动），只统一指标统计方式。

用法
----
    python eval_unified.py                       # 四个模型全跑，输出 results/summary.md
    python eval_unified.py --models unet fcn     # 只跑指定模型
    python eval_unified.py --model fcn           # 单模型模式（内部子进程用，不输出汇总表）
    python eval_unified.py --allow-random        # 权重缺失时用随机初始化跑通流程（自检用）

为什么拆子进程
--------------
四个工程各自都有顶层 `utils/` 和 `model/` 包，同一个进程里无法同时 import，
所以每个模型在独立子进程里加载，结果落到 results/<model>.json，再由主进程汇总。
"""

# pyright: reportMissingImports=false
# 上面这行不是偷懒：本文件里 model / utils / network 的 import 全是**运行时**的——
# 四个工程的 utils、model 包同名，只能先用 _enter() 切换 cwd 并把工程目录插进 sys.path，
# 再 import，否则最先加载的那个包会把另外三个覆盖掉（同名模块只会初始化一次）。
# Pylance 做静态分析时看不到这条运行时路径，会把那 10 处 import 标成"无法解析"，
# 属于预期现象。这行指令只关掉本文件的缺失导入告警，其他诊断照常。

import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

def infer_deeplab_model(weights_path):
    """从 best_{model}_os{OS}_num{C}.pth 里反推主干名。

    DeepLab 的四种主干（resnet50 / resnet101 / xception16 / xception8 / mobilenet）
    结构不同、参数量差一倍，拿错权重不会报错，只会静默加载一部分层、
    输出接近噪声的结果（实测 FG IoU 会从 0.70 掉到 0.02）。
    """
    name = os.path.basename(weights_path or "")
    m = re.match(r"best_(deeplabv3plus_[a-z0-9]+)_os(\d+)_num\d+\.pth", name)
    return m.group(1) if m else None


def infer_deeplab_output_stride(weights_path):
    """同样从文件名解析输出步幅（os8 / os16）。"""
    name = os.path.basename(weights_path or "")
    m = re.search(r"_os(\d+)_num\d+\.pth", name)
    return int(m.group(1)) if m else 16


def _locate_base():
    """向上找到包含 WUH_sub25 的目录，作为工程根。

    本脚本既可能放在 消融实验/ 下，也可能被挪进 XR_1/ 或 XR_2/ 里；
    用 __file__ 直接推导会在移动后指向错误的数据目录，所以按"数据集锚点"来定位。
    """
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(cur, "WUH_sub25")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.abspath(__file__))


def _default_xr_root(base):
    """默认评测哪个 XR 目录：脚本自己就放在某个 XR_* 里时用它，否则用 XR_1。"""
    here = os.path.dirname(os.path.abspath(__file__))
    marks = ("U-Net", "FCN", "DeepLabV3Plus", "SegFormer")
    if sum(os.path.isdir(os.path.join(here, m)) for m in marks) >= 2:
        return here
    return os.path.join(base, "XR_1")


BASE = _locate_base()                                        # .../消融实验
SUBSET = os.path.join(BASE, "WUH_sub25")                     # 四个模型统一读 WUH 原生子集
XR_DEFAULT = _default_xr_root(BASE)                          # 默认评测目录（可用 --xr-root 覆盖）
RESULT_DIR = os.path.join(BASE, "results")

NUM_CLASSES = 2     # 背景 + 建筑
CROP = 512
# 各划分的样本数（test 由 make_test_subset.py 生成）
SPLIT_N = {"val": 474, "test": 604}

# 各模型的 best 权重文件名。用精确文件名而不是通配，避免误捡 2025-07 遗留的
# VOC 时代产物（U-Net/run/train/exp10 里的 best_model_20.pth 就是 num_classes=20 的旧权重）
def build_specs(xr_root):
    """按工程根目录生成规格表，XR_1（消融 A）与 XR_2（消融 B）共用同一套脚本。"""
    return {
        "unet": {
            "project": os.path.join(xr_root, "U-Net"),
            "best_names": ["best_model_1.pth"],
            "label": "U-Net (ResNet50+unetUp)",
        },
        "fcn": {
            "project": os.path.join(xr_root, "FCN"),
            "best_names": ["best_model_1.pth"],
            "label": "FCN (ResNet50)",
        },
        "deeplab": {
            # 各权重文件按 model 名区分，这里把常见三种都列上
            "project": os.path.join(xr_root, "DeepLabV3Plus"),
            "best_names": ["best_deeplabv3plus_xception16_os16_num2.pth",
                           "best_deeplabv3plus_resnet50_os16_num2.pth",
                           "best_deeplabv3plus_resnet101_os16_num2.pth"],
            "label": "DeepLabV3+ (OS16)",
        },
        "segformer": {
            "project": os.path.join(xr_root, "SegFormer"),
            "best_names": ["best_model_2.pth"],
            "label": "SegFormer (MiT-B3)",
        },
    }
ORDER = ["unet", "fcn", "deeplab", "segformer"]


# ---------------------------------------------------------------- 通用工具
def _enter(proj_dir):
    """切到工程目录并让它的顶层包（utils / model / network）可被 import。"""
    if not os.path.isdir(proj_dir):
        raise SystemExit(f"工程目录不存在：{proj_dir}")
    os.chdir(proj_dir)
    if proj_dir not in sys.path:
        sys.path.insert(0, proj_dir)


def _torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:      # PyTorch < 2.6 没有 weights_only
        return torch.load(path, map_location="cpu")


def _extract_state_dict(obj):
    """兼容三种存法：纯 state_dict / {"model_state":...} / DataParallel 的 module. 前缀。"""
    sd = obj
    if isinstance(obj, dict) and "model_state" in obj:
        sd = obj["model_state"]
    if not isinstance(sd, dict):
        raise RuntimeError(f"无法识别的权重格式：{type(obj)}")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def find_best_weights(proj_dir, names):
    """在 run/train/exp*/weights/ 下找指定文件名，多个命中取最新修改的那个。"""
    root = os.path.join(proj_dir, "run", "train")
    if not os.path.isdir(root):
        return None
    hits = []
    for exp in sorted(os.listdir(root)):
        wdir = os.path.join(root, exp, "weights")
        if not os.path.isdir(wdir):
            continue
        for nm in names:
            p = os.path.join(wdir, nm)
            if os.path.isfile(p):
                hits.append(p)
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


def load_weights(model, path):
    sd = _extract_state_dict(_torch_load(path))
    model_sd = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    skipped = [k for k in sd if k not in loadable]
    missing, _ = model.load_state_dict(loadable, strict=False)
    print(f"[CKPT] {path}")
    print(f"[CKPT] matched={len(loadable)} skipped={len(skipped)} missing={len(missing)}")
    if skipped[:3]:
        print(f"[CKPT] skipped 示例：{skipped[:3]}")
    if skipped or missing:
        print(f"[CKPT][警告] 权重与模型结构不匹配：{len(skipped)} 个权重参数未加载、"
              f"{len(missing)} 个模型参数未初始化 —— 本模型的指标不可信，"
              f"先核对主干/解码头规格（如 --segformer-decoder-dim）")
    return len(loadable)


# ---------------------------------------------------------------- 指标
def _acc_hist(hist, gt, pred):
    """把一张图的 (gt, pred) 累加进混淆矩阵；忽略 <0 或 >=C 的非法标签。"""
    mask = (gt >= 0) & (gt < NUM_CLASSES)
    if not mask.any():
        return
    hist += np.bincount(
        NUM_CLASSES * gt[mask].astype(np.int64) + pred[mask],
        minlength=NUM_CLASSES ** 2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_hist(hist):
    """全局混淆矩阵 → PA / mAcc / mIoU / FWIoU / 建筑 IoU。与 DeepLab 的 StreamSegMetrics 同公式。"""
    total = hist.sum()
    if total == 0:
        return dict(pa=0.0, macc=0.0, miou=0.0, fwiou=0.0, iou_bg=0.0, iou_fg=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = np.diag(hist).sum() / total
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    freq = hist.sum(axis=1) / total
    fwiou = float((freq[freq > 0] * iu[freq > 0]).sum())
    return dict(
        pa=float(pa),
        macc=float(np.nanmean(acc_cls)),
        miou=float(np.nanmean(iu)),
        fwiou=fwiou,
        iou_bg=float(np.nan_to_num(iu[0])),
        iou_fg=float(np.nan_to_num(iu[1])),
    )


# ---------------------------------------------------------------- 各模型的构建
def build_model_and_loader(key, args, device, weights=None):
    """返回 (model, forward_fn, loader, extra_info)。预处理保留各工程原版。"""
    import torch
    from torch.utils.data import DataLoader

    if key == "unet":
        _enter(build_specs(args.xr_root)[key]["project"])
        from model.unet_resnet import Unet
        from utils.dataloader import UnetDataset, unet_dataset_collate

        model = Unet(num_classes=NUM_CLASSES)
        ds = UnetDataset(SUBSET, [CROP, CROP], NUM_CLASSES, augmentation=False, split=args.split)
        loader = DataLoader(ds, batch_size=args.eval_batch, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=unet_dataset_collate)

        def fwd(m, x):
            return m(x)

        def unpack(batch):
            return batch[0], batch[1]      # imgs, pngs（第三项 one-hot 用不到）

        return model, fwd, loader, unpack, dict(preprocess="等比缩放+灰边填充, 512")

    if key == "fcn":
        _enter(build_specs(args.xr_root)[key]["project"])
        from model import fcn_resnet50
        from utils.my_dataset import WUHDataset
        from utils.segmentation_transforms import get_transform

        model = fcn_resnet50(num_classes=NUM_CLASSES)
        ds = WUHDataset(SUBSET, transforms=get_transform(train=False), split=args.split)
        loader = DataLoader(ds, batch_size=args.eval_batch, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=ds.collate_fn)

        def fwd(m, x):
            return m(x)["out"]

        def unpack(batch):
            return batch[0], batch[1]

        return model, fwd, loader, unpack, dict(preprocess="等比缩放到 512×512")

    if key == "deeplab":
        _enter(build_specs(args.xr_root)[key]["project"])
        import network
        from utils import WUHSegmentation
        from utils import ext_transforms as et

        # 主干必须与权重一致：优先级为 --deeplab-model > 从权重文件名反推 > resnet50 兜底
        net_name = args.deeplab_model or infer_deeplab_model(weights) or "deeplabv3plus_resnet50"
        os_val = args.deeplab_output_stride or infer_deeplab_output_stride(weights)
        if not hasattr(network.modeling, net_name):
            raise SystemExit(f"DeepLab 主干名不认识：{net_name}（来自权重文件名或 --deeplab-model）")
        model = getattr(network.modeling, net_name)(num_classes=NUM_CLASSES, output_stride=os_val)
        print(f"[deeplab] 主干 {net_name} | output_stride {os_val}")
        val_tf = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ds = WUHSegmentation(root=SUBSET, image_set=args.split, transform=val_tf)
        loader = DataLoader(ds, batch_size=args.eval_batch, shuffle=False, num_workers=0,
                            pin_memory=True)

        def fwd(m, x):
            return m(x)

        def unpack(batch):
            return batch[0], batch[1]

        return model, fwd, loader, unpack, dict(preprocess="原图 512（不缩放）, ImageNet 归一化")

    if key == "segformer":
        _enter(build_specs(args.xr_root)[key]["project"])
        from model.segformer import SegFormer
        from utils.dataloader import UnetDataset, unet_dataset_collate

        model = SegFormer(variant=args.segformer_variant,
                          num_classes=NUM_CLASSES,
                          decoder_dim=args.segformer_decoder_dim,
                          dropout=0.1,
                          drop_path_rate=0.1,
                          img_size=CROP)
        ds = UnetDataset(SUBSET, [CROP, CROP], NUM_CLASSES, augmentation=False, split=args.split)
        loader = DataLoader(ds, batch_size=args.eval_batch, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=unet_dataset_collate)

        def fwd(m, x):
            return m(x)

        def unpack(batch):
            return batch[0], batch[1]

        return model, fwd, loader, unpack, dict(preprocess="等比缩放+居中填充, 512, ImageNet 归一化")

    raise ValueError(f"未知模型：{key}")


def measure_speed(model, fwd, device, args):
    """固定 512×512 输入测推理延迟与吞吐，与数据集无关，四模型完全同条件。"""
    import torch

    out = {}
    model.eval()
    on_cuda = device.type == "cuda"
    for tag, bs, iters in (("latency_bs1", 1, 30), ("throughput_bs8", 8, 12)):
        if not on_cuda:            # CPU 自检时不做有意义的速度测量，只保证流程能过
            iters = 3
        warmup = 5 if on_cuda else 1
        x = torch.randn(bs, 3, CROP, CROP, device=device)
        with torch.inference_mode():
            for _ in range(warmup):
                fwd(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(iters):
                fwd(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
        out[tag] = dict(
            batch=bs,
            ms_per_batch=dt * 1000,
            ms_per_img=dt * 1000 / bs,
            img_per_s=bs / dt,
            peak_mem_mb=peak,
        )
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- 单模型评测
def run_one(key, args):
    import torch

    spec = build_specs(args.xr_root)[key]
    if key == "segformer":
        # 规格由命令行决定，表里的名字要跟着变，否则跑 b0 也会显示成 B3
        spec = dict(spec, label=f"SegFormer (MiT-{args.segformer_variant.upper()})")
    device = torch.device(args.device)

    # 先把权重定下来再建模型：DeepLab 要根据权重文件名选主干
    weights = args.weights or find_best_weights(spec["project"], spec["best_names"])
    if weights:
        # 必须转绝对路径：build_model_and_loader 里的 _enter() 会 chdir 到工程目录，
        # 之后相对路径就指向别处了（曾导致传入的 --weights 被判为"不存在"）
        weights = os.path.abspath(weights)
    if key == "deeplab":
        net_name = infer_deeplab_model(weights)
        if net_name:
            spec = dict(spec, label=f"DeepLabV3+ ({net_name.replace('deeplabv3plus_', '')}, "
                                    f"OS{infer_deeplab_output_stride(weights)})")

    model, fwd, loader, unpack, extra = build_model_and_loader(key, args, device, weights)
    if weights and os.path.isfile(weights):
        load_weights(model, weights)
        weight_status = weights
    else:
        if not args.allow_random:
            raise SystemExit(
                f"[{key}] 找不到 best 权重（期望 run/train/exp*/weights/{spec['best_names'][0]}）。\n"
                f"        先训练，或显式传 --weights；确认流程可用可加 --allow-random。"
            )
        print(f"[{key}] 未找到权重，使用随机初始化（--allow-random），指标无意义。")
        weight_status = "random_init"

    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{key}] 参数量 {n_params/1e6:.2f} M | 验证集 {len(loader.dataset)} 张")

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_img_fg_iou = []       # 旧口径：按图求 FG IoU、空图记 0，复现 U-Net/FCN 原逻辑
    n_empty = 0

    t0 = time.time()
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            imgs, gt = unpack(batch)
            imgs = imgs.to(device, dtype=torch.float32, non_blocking=True)
            logits = fwd(model, imgs)
            pred = logits.argmax(dim=1).cpu().numpy()
            gt = gt.cpu().numpy()
            for p, g in zip(pred, gt):
                _acc_hist(hist, g, p)
                t = (g == 1)
                if t.sum() == 0:
                    n_empty += 1
                u = np.logical_or(t, p == 1).sum()
                per_img_fg_iou.append(float(np.logical_and(t, p == 1).sum() / u) if u > 0 else 0.0)
            seen += len(gt)
            if args.limit and seen >= args.limit:
                break
    eval_sec = time.time() - t0

    m = metrics_from_hist(hist)
    result = dict(
        key=key,
        label=spec["label"],
        weights=weight_status,
        n_val=len(per_img_fg_iou),
        n_empty=n_empty,
        params_m=n_params / 1e6,
        metrics=m,
        legacy_fg_iou_per_image=float(np.mean(per_img_fg_iou)) if per_img_fg_iou else 0.0,
        eval_seconds=eval_sec,
        preprocess=extra["preprocess"],
        device=str(device),
    )
    result.update(measure_speed(model, fwd, device, args))
    result["confusion_matrix"] = hist.tolist()

    print(f"[{key}] PA={m['pa']:.4f} mIoU={m['miou']:.4f} FG IoU={m['iou_fg']:.4f} "
          f"| 旧口径 FG IoU={result['legacy_fg_iou_per_image']:.4f} | 空图 {n_empty}/{len(loader.dataset)}")
    return result


# ---------------------------------------------------------------- 汇总表
def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def write_summary(results, args):
    os.makedirs(RESULT_DIR, exist_ok=True)
    ok = [r for r in results if "error" not in r]

    lines = []
    lines.append("# W3 消融 A · 四模型对比（统一口径）\n")
    lines.append(f"口径：WUH 固定子集 train 1184 / {args.split} {SPLIT_N.get(args.split, '—')} · 输入 512×512 · "
                 f"bs 8 · 指标为全验证集全局混淆矩阵（与 DeepLab StreamSegMetrics 同公式）\n")
    lines.append(f"评测设备：{args.device}；评测 batch：{args.eval_batch}（不影响混淆矩阵，只影响速度）\n")
    lines.append("只比相对优劣，绝对值不跨子集比较。\n")

    lines.append("## 1 主表\n")
    lines.append("| 模型 | 参数量(M) | PA | mIoU(2类) | 建筑 IoU | 背景 IoU | mAcc | FWIoU | 延迟 bs1(ms) | 吞吐 bs8(img/s) | 推理显存 bs8(MB) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ok:
        m = r["metrics"]
        lines.append("| {} | {:.2f} | {} | {} | {} | {} | {} | {} | {} | {:.1f} | {:.0f} |".format(
            r["label"], r["params_m"], _fmt(m["pa"]), _fmt(m["miou"]), _fmt(m["iou_fg"]),
            _fmt(m["iou_bg"]), _fmt(m["macc"]), _fmt(m["fwiou"]),
            _fmt(r["latency_bs1"]["ms_per_img"], 2), r["throughput_bs8"]["img_per_s"],
            r["throughput_bs8"]["peak_mem_mb"],
        ))
    lines.append("")
    lines.append("注：延迟/吞吐为固定随机张量、单卡、`torch.inference_mode()` 下的实测值，"
                 "含 5 次预热；显存为**推理**峰值，非训练峰值。\n")

    lines.append("## 2 口径核查（本次改用全局混淆矩阵的原因）\n")
    lines.append("| 模型 | 旧口径 FG IoU（按图平均、空图记 0） | 统一口径 FG IoU | 验证集空图数 |")
    lines.append("|---|---|---|---|")
    for r in ok:
        lines.append("| {} | {} | {} | {} / {} |".format(
            r["label"], _fmt(r["legacy_fg_iou_per_image"]), _fmt(r["metrics"]["iou_fg"]),
            r["n_empty"], r["n_val"]))
    lines.append("")
    if ok:
        r0 = ok[0]
        full_n = SPLIT_N.get(args.split, r0["n_val"])
        truncated = "" if r0["n_val"] == full_n else f"（自检模式，本次只评了 {r0['n_val']} 张，比例不具代表性）"
        lines.append(f"本次评测 {r0['n_val']} 张中，{r0['n_empty']} 张完全没有建筑{truncated}。"
                     "FCN 验证 batch 写死为 1，这些图逐张被记为 0 分；"
                     "改用全局混淆矩阵后，空图在 0/0 时被跳过、误检仍正常计入 FP。\n")

    lines.append("## 3 各模型的预处理（保持上游原版，未统一）\n")
    lines.append("| 模型 | 验证期预处理 | 权重 |")
    lines.append("|---|---|---|")
    for r in ok:
        w = r["weights"]
        w = "随机初始化（自检）" if w == "random_init" else os.path.basename(w)
        lines.append(f"| {r['label']} | {r['preprocess']} | {w} |")
    lines.append("")

    bad = [r for r in results if "error" in r]
    if bad:
        lines.append("## 4 未完成\n")
        for r in bad:
            lines.append(f"- {r['key']}：{r['error']}")
        lines.append("")

    md = "\n".join(lines)
    md_path = os.path.join(RESULT_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[输出] {md_path}")

    # CSV（便于直接贴进报告或 Excel）
    csv_path = os.path.join(RESULT_DIR, "summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("model,params_M,PA,mIoU,iou_bg,iou_fg,mAcc,FWIoU,"
                "latency_bs1_ms,throughput_bs8_ips,infer_mem_bs8_MB,"
                "legacy_fg_iou,n_empty\n")
        for r in ok:
            m = r["metrics"]
            f.write(f"{r['label']},{r['params_m']:.3f},{m['pa']:.6f},{m['miou']:.6f},"
                    f"{m['iou_bg']:.6f},{m['iou_fg']:.6f},{m['macc']:.6f},{m['fwiou']:.6f},"
                    f"{r['latency_bs1']['ms_per_img']:.3f},{r['throughput_bs8']['img_per_s']:.2f},"
                    f"{r['throughput_bs8']['peak_mem_mb']:.0f},"
                    f"{r['legacy_fg_iou_per_image']:.6f},{r['n_empty']}\n")
    print(f"[输出] {csv_path}")
    return md_path


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(description="四模型统一评测（W3 消融 A）")
    ap.add_argument("--models", nargs="+", default=ORDER, choices=ORDER,
                    help="要评测的模型，默认四个全跑")
    ap.add_argument("--model", default=None, choices=ORDER,
                    help="单模型模式（内部子进程用）：只算一个模型并写 --json-out")
    ap.add_argument("--json-out", default=None, help="单模型模式的结果输出路径")
    ap.add_argument("--weights", default=None, help="覆盖自动查找的 best 权重路径")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="评测用的划分：val（训练期选优用）/ test（留出集，见 make_test_subset.py）")
    ap.add_argument("--xr-root", default=XR_DEFAULT,
                    help="工程根目录：XR_1（消融 A）或 XR_2（消融 B）")
    ap.add_argument("--device", default="cuda", help="cuda / cpu")
    ap.add_argument("--eval-batch", default=8, type=int,
                    help="评测 batch（不影响全局混淆矩阵结果，只影响速度）")
    ap.add_argument("--segformer-variant", default="b3", choices=["b0", "b1", "b2", "b3"])
    ap.add_argument("--segformer-decoder-dim", default=256, type=int,
                    help="SegFormer 解码头宽度，必须与训练一致。本仓库训练用 256；"
                         "若按 128 建模型会静默丢 12 个 decode_head.linear_layers.* 参数，"
                         "指标会掉到 0.1~0.25 而脚本不报错（2026-10-23 实测踩到）")
    ap.add_argument("--deeplab-model", default="",
                    help="DeepLab 主干名（如 deeplabv3plus_xception16）；留空则从权重文件名反推")
    ap.add_argument("--deeplab-output-stride", default=0, type=int, choices=[0, 8, 16],
                    help="DeepLab 输出步幅；0 表示从权重文件名反推")
    ap.add_argument("--allow-random", action="store_true",
                    help="权重缺失时用随机初始化跑通流程（仅自检，指标无意义）")
    ap.add_argument("--limit", default=0, type=int,
                    help="只评测前 N 张（调试用，正式出表不要加）")
    ap.add_argument("--keep-json", action="store_true", help="保留每个模型的中间 json")
    args = ap.parse_args()

    # ---- 单模型模式：在子进程里跑，避免四个工程的同名 utils/model 包互相污染 ----
    if args.model:
        res = run_one(args.model, args)
        out = args.json_out or os.path.join(RESULT_DIR, f"{args.model}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[输出] {out}")
        return

    # ---- 主进程：逐个起子进程 ----
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("[提示] CUDA 不可用，回落到 CPU 评测（速度指标无参考价值）")
                args.device = "cpu"
        except Exception:
            args.device = "cpu"

    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []
    for key in args.models:
        print(f"\n{'=' * 60}\n>>> {build_specs(args.xr_root)[key]['label']}\n{'=' * 60}")
        json_path = os.path.join(RESULT_DIR, f"{key}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--model", key,
               "--json-out", json_path, "--device", args.device,
               "--xr-root", args.xr_root,
               "--split", args.split,
               "--eval-batch", str(args.eval_batch),
               "--segformer-variant", args.segformer_variant,
               "--segformer-decoder-dim", str(args.segformer_decoder_dim),
               "--deeplab-model", args.deeplab_model,
               "--deeplab-output-stride", str(args.deeplab_output_stride)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.weights:
            cmd += ["--weights", os.path.abspath(args.weights)]
        if args.allow_random:
            cmd += ["--allow-random"]

        proc = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-6:]
            print("\n".join(tail))
            results.append(dict(key=key, error=(proc.stderr or "子进程失败").strip().splitlines()[-1]))
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
        if not args.keep_json:
            os.remove(json_path)

    if any("error" not in r for r in results):
        write_summary(results, args)
    else:
        print("\n四个模型均未跑通，未生成汇总表。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""W3 大图滑窗推理：512 窗口 + 25% 重叠 + 汉宁加权融合，并与"逐瓦片硬拼"对比。

两种模式
--------
    single  单块（演示 / 调试）。自动挑一块，或 --r0/--c0 指定
    batch   全量切块。按瓦片网格逐块跑，每块出一张三联对比图 + 一行指标，最后汇总

为什么要拼大图
--------------
WHU 数据集的瓦片全部是 512×512，对单张瓦片跑滑窗会退化成 1 个窗口（重叠区为空、
融合是恒等映射），什么也验证不了。所以按 WHU 原始切片规则把连续的 N×N 个瓦片拼回
大图，再对比两条推理路径：

    A  逐瓦片推理后硬拼     —— 生产里"切片 → 逐块推理 → 拼接"的朴素做法
    B  大图滑窗 + 加权融合   —— 本脚本要验证的做法

切片规则（实测；batch 模式会对每个 split 自动检测行宽）
    编号 n 与 n+1  水平相邻（右列|左列 MAE 18.4，随机两张 61.4）
    编号 n 与 n+64 竖直相邻（MAE 18.5，其余偏移 48~66）
    train: 4736 = 64 × 74  →  row-major，行宽 64，0-based
    idx = row * ROW_WIDTH + col
    val/test 的行宽未知，用 detect_row_width() 扫描后缓存

用法
----
    # 单块演示（自动挑块）
    python XR_1/sliding_window.py

    # 单块，指定位置
    python XR_1/sliding_window.py --r0 24 --c0 56

    # 全量 train，2×2 切块，逐块出图
    python XR_1/sliding_window.py --mode batch --split train --block 2

    # 三个划分全跑，4×4 切块
    python XR_1/sliding_window.py --mode batch --split all --block 4

    # 只算指标不出图（快很多）
    python XR_1/sliding_window.py --mode batch --block 4 --no-images

    # 断点续跑
    python XR_1/sliding_window.py --mode batch --block 4 --skip-existing

输出（single）
--------------
    triplet.png / seam_zoom.png / metrics.json / summary.md (+ 可选 npy)

输出（batch）
-------------
    blocks/<split>/r024_c056/triplet.png   逐块子目录
    blocks.csv                             一行一块的全部指标（可断点续跑）
    summary.md                             汇总表：均值±标准差，按 split 分组
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

NUM_CLASSES = 2
TILE = 512
TRAIN_ROW_WIDTH = 64                # train: 4736 = 64 × 74（实测；其它 split 自动检测）
WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = r"D:\[C]Remote_Sencing_Projection\Project_WUD\Deeplabv3+\WHU"
UNET_PROJ = os.path.join(WS_ROOT, "XR_1", "U-Net")
# 必须和 UNET_PROJ 下的 Unet 结构对应（340 key）。别指到 Project_WUD\U-Net\run\train\exp6，
# 那是另一种结构（398 key，带 scSE），会静默退化成"全图判背景"。
DEFAULT_WEIGHTS = os.path.join(UNET_PROJ, "run", "train", "exp", "weights", "best_model_1.pth")

IMAGE_DIRNAME = "image"
LABEL_DIRNAME = "label"


# ------------------------------------------------------------------ 数据
def load_mask(path):
    """WHU 标签是 PIL mode='1'（bool 数组），必须转 0/1，不能 convert('L')（True→255）。"""
    arr = np.array(Image.open(path))
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def split_dir(split):
    return os.path.join(DATA_ROOT, split)


_NUM_RE = re.compile(r"(\d+)\s*$")


def index_tiles(split="train"):
    """{网格编号: 文件名stem}。

    train 是纯数字 `123.tif`；val 是 `val_123.tif`；**test 混了两批数据**：
    `2_123.tif`（1702 张 = 23×74，成网格）和 `123.tif`（714 张，另一批 0..713）。
    两批编号会撞车，所以按"前缀投票"取该目录的主导前缀，撞车的丢弃并告警。
    """
    d = os.path.join(split_dir(split), IMAGE_DIRNAME)
    cand = []                     # (编号, stem, 前缀)
    for name in os.listdir(d):
        stem = os.path.splitext(name)[0]
        m = _NUM_RE.search(stem)
        if not m:
            continue              # `val_1028 - 副本.tif` 这类脏文件自动排除
        cand.append((int(m.group(1)), stem, stem[:m.start()]))
    if not cand:
        return {}

    votes = Counter(p for _, _, p in cand)
    dominant = votes.most_common(1)[0][0]
    kept, conflict = {}, 0
    for num, stem, pref in cand:
        if pref != dominant:
            continue
        if num in kept:
            conflict += 1
            continue
        kept[num] = stem

    others = {p or "<纯数字>": v for p, v in votes.items() if p != dominant}
    if others:
        print(f"   [命名] {split}: 主导前缀 '{dominant or '<纯数字>'}' {votes[dominant]} 张，"
              f"忽略其它 {others}；编号冲突 {conflict}")
    return kept


def _edge_strips(path):
    """返回 (top, bottom, left, right) 四条边的一维灰度，用于检测切片网格。"""
    a = np.asarray(Image.open(path).convert("RGB"), np.float32).mean(2)
    return a[0], a[-1], a[:, 0], a[:, -1]


def detect_row_width(split, max_k=200, window=420, cache_path=None):
    """自动检测行宽（每行多少张瓦片）。

    思路：编号 n 与 n+k 若竖直相邻，则 bottom(n) 与 top(n+k) 的差异会显著低于其它 k。
    返回 (row_width, quality)；quality = 最小 MAE / 中位 MAE，越小越可信（<0.5 算明确）。
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            c = json.load(f)
        print(f"   [行宽检测] {split}: 命中缓存 row_width={c['row_width']}")
        return int(c["row_width"]), float(c["quality"]), c.get("note", "cached")

    idx = index_tiles(split)
    keys = sorted(idx)
    if len(keys) < 4:
        raise SystemExit(f"{split} 里没有足够的瓦片")

    # 取一段编号连续的窗口，一次性解码边缘（避免上万个文件读）
    best_run, run_start = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[i - 1] + 1:
            if i - run_start > len(best_run):
                best_run, run_start = keys[run_start:i], run_start
            run_start = i
    run = best_run[:window]
    img_dir = os.path.join(split_dir(split), IMAGE_DIRNAME)
    strips = {}
    for k in run:
        strips[k] = _edge_strips(os.path.join(img_dir, idx[k] + ".tif"))
    print(f"   [行宽检测] {split}: 解码连续编号 {run[0]}..{run[-1]}（{len(run)} 张）")

    def v_mae(k):
        vals = [float(np.abs(strips[n][1] - strips[n + k][0]).mean())
                for n in run if n + k in strips]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    def h_mae(k):
        vals = [float(np.abs(strips[n][3] - strips[n + k][2]).mean())
                for n in run if n + k in strips]
        return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

    base, _ = h_mae(1)
    raw = []
    for k in range(1, min(max_k, len(run) - 1) + 1):
        m, cnt = v_mae(k)
        if m is not None and cnt >= 5:
            raw.append((k, m))
    if not raw:
        raise SystemExit("行宽检测失败：候选 k 全部无足够配对")
    med = float(np.median([m for _, m in raw]))
    n_total = len(idx)
    # 真正的行宽必须整除瓦片总数（整行整列），给不整除的候选加罚分
    scored = sorted(((m * (1.0 if n_total % k == 0 else 1.8), k, m) for k, m in raw))
    _, best_k, best_m = scored[0]
    divides = (n_total % best_k == 0)
    quality = best_m / med if med > 0 else float("inf")
    note = (f"k={best_k} MAE={best_m:.2f} 中位={med:.2f} 比值={quality:.3f} "
            f"整行={'是' if divides else '否'}（{n_total}/{best_k}={n_total / best_k:.2f}）；"
            f"水平相邻基线 MAE={base:.2f}")
    print(f"   [行宽检测] {note}")
    if quality > 0.6 and not divides:
        print("   [行宽检测][警告] 竖直相邻信号不明确，网格可能不可靠，请用 --row-width 手动指定")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(dict(row_width=best_k, quality=quality, note=note), f,
                      ensure_ascii=False, indent=2)
    return int(best_k), float(quality), note


def iter_full_blocks(keys, row_width, rows, cols):
    """枚举所有"完整"的 rows×cols 块（步长 = 块大小，互不重叠）。"""
    keyset = set(keys)
    n_rows = max(keys) // row_width + 1
    skipped = 0
    blocks = []
    for r0 in range(0, n_rows - rows + 1, rows):
        for c0 in range(0, row_width - cols + 1, cols):
            ids = [(r0 + dr) * row_width + (c0 + dc)
                   for dr in range(rows) for dc in range(cols)]
            if all(i in keyset for i in ids):
                blocks.append((r0, c0))
            else:
                skipped += 1
    return blocks, n_rows, skipped


def block_ids(row_width, r0, c0, rows, cols):
    return [[(r0 + dr) * row_width + (c0 + dc) for dc in range(cols)]
            for dr in range(rows)]


def make_mosaic(split, row_width, r0, c0, rows, cols, idx):
    """按 row-major 规则拼出大图与对应 GT。"""
    H, W = rows * TILE, cols * TILE
    img = np.zeros((H, W, 3), np.uint8)
    gt = np.zeros((H, W), np.uint8)
    base = split_dir(split)
    for dr in range(rows):
        for dc in range(cols):
            stem = idx[(r0 + dr) * row_width + (c0 + dc)] + ".tif"
            im = Image.open(os.path.join(base, IMAGE_DIRNAME, stem)).convert("RGB")
            lb = load_mask(os.path.join(base, LABEL_DIRNAME, stem))
            y, x = dr * TILE, dc * TILE
            img[y:y + TILE, x:x + TILE] = np.array(im, np.uint8)
            gt[y:y + TILE, x:x + TILE] = lb
    return img, gt


def pick_block(split, idx, rows, cols, row_width, step=8, lo=0.10, hi=0.35):
    """单块模式：挑一块建筑密度适中、且块内密度有变化的区域（避免整块空图让接缝指标退化）。"""
    cache = {}
    label_dir = os.path.join(split_dir(split), LABEL_DIRNAME)

    def fg(stem):
        if stem not in cache:
            cache[stem] = float(load_mask(os.path.join(label_dir, stem + ".tif")).mean())
        return cache[stem]

    max_row = max(idx) // row_width + 1 - rows
    max_col = row_width - cols
    best = None
    for r0 in range(0, max_row + 1, step):
        for c0 in range(0, max_col + 1, step):
            ids = [(r0 + dr) * row_width + (c0 + dc)
                   for dr in range(rows) for dc in range(cols)]
            if any(i not in idx for i in ids):
                continue
            v = np.array([fg(idx[i]) for i in ids], dtype=np.float64)
            mean = float(v.mean())
            if lo <= mean <= hi:
                score = mean + 0.5 * float(v.std())
                if best is None or score > best["score"]:
                    best = dict(score=score, r0=r0, c0=c0, mean=mean,
                                std=float(v.std()), ids=ids)
    if best is None:
        raise SystemExit("找不到合适的大图块，放宽 pick_block 的 lo/hi 或改用 --r0/--c0")
    return best


# ------------------------------------------------------------------ 模型
def build_unet(weights, device, allow_mismatch=False):
    r"""U-Net 原版模型 + 权重。预处理按工程原版：只做 /255，不做 mean/std 归一化。

    ⚠ 参数必须和 `XR_1/U-Net/model/unet_resnet.py` 里的 Unet 完全对应（340 个 key）。
    拿别的工程的权重（例如 `Project_WUD\U-Net\run\train\exp6`，398 个 key，
    上采样块是 cbr1/cbr2 + scSE 注意力）会静默丢掉 78 个参数、20 个参数随机初始化，
    结果就是**整幅图全判背景、IoU 恰好 0**。所以这里默认直接报错退出。
    """
    import torch

    if UNET_PROJ not in sys.path:
        sys.path.insert(0, UNET_PROJ)
    cwd = os.getcwd()
    os.chdir(UNET_PROJ)                      # 工程内可能有相对路径的导入
    try:
        from model.unet_resnet import Unet
    finally:
        os.chdir(cwd)

    model = Unet(num_classes=NUM_CLASSES)
    try:
        obj = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(weights, map_location="cpu")
    sd = obj.get("model_state", obj) if isinstance(obj, dict) else obj
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    own = model.state_dict()
    loadable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing, _ = model.load_state_dict(loadable, strict=False)
    print(f"[CKPT] {weights}")
    print(f"[CKPT] matched={len(loadable)} skipped={len(sd) - len(loadable)} missing={len(missing)}")
    if len(loadable) < len(own) or len(loadable) < len(sd):
        msg = (f"[CKPT][错误] 权重与模型结构不匹配：权重 {len(sd)} 个 key，模型 {len(own)} 个 key，"
               f"只对上 {len(loadable)} 个")
        if not allow_mismatch:
            raise SystemExit(
                msg + "\n"
                "  这份权重不是用 XR_1/U-Net/model/unet_resnet.py 的 Unet 训出来的。\n"
                "  典型差异：up_concat*.conv1/conv2 + up_conv.* 被换成了 cbr1/cbr2 + scSE 注意力。\n"
                "  强行加载会丢掉那些参数、其余随机初始化 → 全图判背景、IoU 恰好 0。\n"
                "  请用 --weights 指向匹配的权重（XR_1\\U-Net\\run\\train\\exp\\weights\\"
                "best_model_1.pth），或加 --allow-mismatch 明确承担这个后果。")
        print(msg + "（已指定 --allow-mismatch，继续）")

    model.to(device).eval()
    return model


def make_infer(model, device, batch=8):
    """(N,512,512,3) uint8 -> (N,2,512,512) float32 概率。"""
    import torch

    def infer(patches):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(patches), batch):
                x = patches[i:i + batch].astype(np.float32) / 255.0
                t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
                outs.append(torch.softmax(model(t), dim=1).float().cpu().numpy())
        return np.concatenate(outs, 0)

    return infer


# ------------------------------------------------------------------ 两条推理路径
def window_starts(length, win, stride):
    """窗口起点；最后一行/列贴边对齐，避免引入 pad 伪影。"""
    if length <= win:
        return [0]
    pos = list(range(0, length - win + 1, stride))
    if pos[-1] != length - win:
        pos.append(length - win)
    return pos


def blend_weights(win, blend):
    if blend == "hann":
        h = np.hanning(win).astype(np.float32)
        return np.outer(h, h) + 1e-3        # 角落为 0 时防除零
    return np.ones((win, win), np.float32)


def sliding_predict(infer, img, win=TILE, overlap=0.25, blend="hann"):
    """大图滑窗：在概率图上加权累加，最后统一 argmax（绝不在 label 上平均）。"""
    H, W = img.shape[:2]
    stride = max(1, int(round(win * (1.0 - overlap))))
    ys = window_starts(H, win, stride)
    xs = window_starts(W, win, stride)
    coords = [(y, x) for y in ys for x in xs]

    w2 = blend_weights(win, blend)
    psum = np.zeros((NUM_CLASSES, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)

    patches = np.stack([img[y:y + win, x:x + win] for y, x in coords])
    probs = infer(patches)
    for k, (y, x) in enumerate(coords):
        psum[:, y:y + win, x:x + win] += probs[k] * w2
        wsum[y:y + win, x:x + win] += w2

    prob = psum / np.maximum(wsum, 1e-8)
    return prob.argmax(0).astype(np.uint8), prob, dict(
        n_windows=len(coords), stride=stride, ys=ys, xs=xs,
        redundancy=float(len(coords) * win * win / (H * W)))


def tilewise_predict(infer, img, rows, cols, win=TILE):
    """逐瓦片推理后硬拼：位置固定为整数倍，等价于"切片-逐块推理-拼接"。"""
    coords = [(r * win, c * win) for r in range(rows) for c in range(cols)]
    patches = np.stack([img[y:y + win, x:x + win] for y, x in coords])
    probs = infer(patches)
    pred = np.zeros(img.shape[:2], np.uint8)
    for k, (y, x) in enumerate(coords):
        pred[y:y + win, x:x + win] = probs[k].argmax(0)
    return pred, probs


# ------------------------------------------------------------------ 指标
def hist_of(gt, pred, region=None):
    g, p = (gt, pred) if region is None else (gt[region], pred[region])
    m = (g >= 0) & (g < NUM_CLASSES)
    if not m.any():
        return np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    return np.bincount(NUM_CLASSES * g[m].astype(np.int64) + p[m],
                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def fg_iou(h):
    inter = np.diag(h).astype(np.float64)
    union = h.sum(0) + h.sum(1) - inter
    return float(inter[1] / union[1]) if union[1] > 0 else float("nan")


def seam_jump(pred, win=TILE):
    """跨缝跳变率：接缝线两侧相邻像素预测不一致的比例。

    这是"接缝伪影"最直接的量化——逐瓦片硬拼时，接缝两侧来自两个独立的推理窗口，
    预测会在线上突然跳变；滑窗融合时线上像素被多个窗口覆盖，跳变应该显著变小。
    """
    H, W = pred.shape
    vs, hs = [], []
    for x in range(win, W, win):
        vs.append(float(np.abs(pred[:, x].astype(np.int16) -
                               pred[:, x - 1].astype(np.int16)).mean()))
    for y in range(win, H, win):
        hs.append(float(np.abs(pred[y, :].astype(np.int16) -
                               pred[y - 1, :].astype(np.int16)).mean()))
    return dict(vertical=vs, horizontal=hs,
                mean=float(np.mean(vs + hs)) if (vs or hs) else 0.0)


def seam_mask(shape, win, half=2):
    """接缝带：瓦片边界线左右各 half 像素。"""
    H, W = shape
    mask = np.zeros((H, W), bool)
    lines = []
    for x in range(win, W, win):
        mask[:, max(0, x - half):x + half] = True
        lines.append(("x", x))
    for y in range(win, H, win):
        mask[max(0, y - half):y + half, :] = True
        lines.append(("y", y))
    return mask, lines


# ------------------------------------------------------------------ 可视化
def to_rgb(mask, colors=((40, 40, 40), (255, 255, 255))):
    out = np.zeros(mask.shape + (3,), np.uint8)
    for v, c in enumerate(colors):
        out[mask == v] = c
    return out


def error_map(gt, pred):
    out = np.full(gt.shape + (3,), 60, np.uint8)
    out[(gt == 0) & (pred == 0)] = (215, 215, 215)
    out[(gt == 1) & (pred == 1)] = (40, 180, 70)      # TP 绿
    out[(gt == 0) & (pred == 1)] = (220, 50, 50)      # FP 红
    out[(gt == 1) & (pred == 0)] = (50, 90, 230)      # FN 蓝
    return out


def diff_map(a, b):
    out = np.full(a.shape + (3,), 235, np.uint8)
    out[(a == 1) & (b == 0)] = (220, 50, 50)          # 只有逐瓦片认为是前景
    out[(a == 0) & (b == 1)] = (50, 90, 230)          # 只有滑窗认为是前景
    return out


def panel(arr, size, label=None, nearest=False):
    im = Image.fromarray(arr)
    im = im.resize((size, size), Image.NEAREST if nearest else Image.BICUBIC)
    canvas = Image.new("RGB", (size, size + 18), (255, 255, 255))
    canvas.paste(im, (0, 18))
    if label:
        ImageDraw.Draw(canvas).text((4, 4), label, fill=(0, 0, 0))
    return canvas


def img_path(out_dir, name, args):
    return os.path.join(out_dir, name + (".jpg" if args.jpg else ".png"))


def _dump(canvas, path):
    if path.lower().endswith(".jpg"):
        canvas.save(path, quality=92)
    else:
        canvas.save(path)


def save_triplet(path, img, gt, pred_a, pred_b, lines, size=512, title=None):
    seam_vis = np.array(Image.fromarray(pred_b * 255).convert("RGB"))
    for kind, pos in lines:
        if kind == "x":
            seam_vis[:, max(0, pos - 1):pos + 1] = (255, 0, 255)
        else:
            seam_vis[max(0, pos - 1):pos + 1, :] = (255, 0, 255)
    row1 = [panel(img, size, "RGB mosaic"),
            panel(to_rgb(gt), size, "ground truth", True),
            panel(to_rgb(pred_a), size, "A tile-wise stitch", True),
            panel(to_rgb(pred_b), size, "B sliding+blend", True)]
    row2 = [panel(diff_map(pred_a, pred_b), size, "diff A vs B (red=A only, blue=B only)", True),
            panel(error_map(gt, pred_a), size, "A errors (G=TP R=FP B=FN)", True),
            panel(error_map(gt, pred_b), size, "B errors (G=TP R=FP B=FN)", True),
            panel(seam_vis, size, "B with tile seams (magenta)", True)]
    canvas = Image.new("RGB", (size * 4, (size + 18) * 2), (255, 255, 255))
    for i, p in enumerate(row1):
        canvas.paste(p, (i * size, 0))
    for i, p in enumerate(row2):
        canvas.paste(p, (i * size, size + 18))
    if title:
        ImageDraw.Draw(canvas).text((4, 2), title, fill=(0, 0, 0))
    _dump(canvas, path)
    return canvas


def save_seam_zoom(path, img, gt, pred_a, pred_b, center=None, half=192, up=2):
    """接缝放大对比。center 默认取图像中心——对 2×2 是唯一接缝交点，对 N×N 是中间交叉点。"""
    H, W = gt.shape
    if center is None:
        center = (H // 2, W // 2)
    cy, cx = center
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    crop = lambda a: a[y0:y1, x0:x1]
    size = min(y1 - y0, x1 - x0) * up
    panels = [panel(crop(img), size, "RGB (2x)"),
              panel(to_rgb(crop(gt)), size, "GT", True),
              panel(to_rgb(crop(pred_a)), size, "A tile-wise", True),
              panel(to_rgb(crop(pred_b)), size, "B sliding", True)]
    canvas = Image.new("RGB", (size * len(panels), size + 18), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * size, 0))
    _dump(canvas, path)
    return canvas


# ------------------------------------------------------------------ 单块实验
def run_block(infer, split, row_width, r0, c0, rows, cols, args, device, idx):
    """跑一个块的 A/B 对照，返回指标 dict（不落盘）。"""
    import torch

    img, gt = make_mosaic(split, row_width, r0, c0, rows, cols, idx)
    H, W = gt.shape

    t0 = time.perf_counter()
    pred_a, _ = tilewise_predict(infer, img, rows, cols)
    if device == "cuda":
        torch.cuda.synchronize()
    t_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_b, prob_b, info_b = sliding_predict(infer, img, TILE, args.overlap, args.blend)
    if device == "cuda":
        torch.cuda.synchronize()
    t_b = time.perf_counter() - t0

    band, lines = seam_mask(gt.shape, TILE, args.seam_half)
    interior = ~band
    flip = pred_a != pred_b
    jump_a, jump_b = seam_jump(pred_a, TILE), seam_jump(pred_b, TILE)

    res = dict(
        r0=r0, c0=c0, rows=rows, cols=cols, split=split,
        W=int(W), H=int(H), row_width=row_width,
        fg_ratio=float(gt.mean()),
        n_win_a=int(rows * cols), n_win_b=int(info_b["n_windows"]),
        stride=int(info_b["stride"]), overlap=args.overlap, blend=args.blend,
        redundancy=float(info_b["redundancy"]),
        t_a=float(t_a), t_b=float(t_b),
        seam_band_pixel_ratio=float(band.mean()),
        a_all=fg_iou(hist_of(gt, pred_a)),
        b_all=fg_iou(hist_of(gt, pred_b)),
        a_band=fg_iou(hist_of(gt, pred_a, band)),
        b_band=fg_iou(hist_of(gt, pred_b, band)),
        a_interior=fg_iou(hist_of(gt, pred_a, interior)),
        b_interior=fg_iou(hist_of(gt, pred_b, interior)),
        flip_overall=float(flip.mean()),
        flip_in_band=float(flip[band].mean()),
        flip_interior=float(flip[interior].mean()),
        jump_a=float(jump_a["mean"]),
        jump_b=float(jump_b["mean"]),
        agreement_iou=float((np.logical_and(pred_a == 1, pred_b == 1).sum()) /
                            max(1, np.logical_or(pred_a == 1, pred_b == 1).sum())),
        agreement_pixel_acc=float((pred_a == pred_b).mean()),
    )
    res["jump_reduction"] = (res["jump_b"] / res["jump_a"]) if res["jump_a"] > 0 else float("nan")
    res["d_all"] = res["b_all"] - res["a_all"]
    res["d_band"] = res["b_band"] - res["a_band"]
    res["d_interior"] = res["b_interior"] - res["a_interior"]
    res["pred_fg_ratio"] = float((pred_b == 1).mean())
    # 兜底自检：权重若加载错（结构不匹配、或多值预处理不一致），模型会塌缩成"全判背景"，
    # 此时 IoU 恰好 0 而不报错。这里显式喊出来，避免又出现一整批 0 分结果。
    if res["fg_ratio"] > 0.02 and res["pred_fg_ratio"] == 0.0:
        print(f"   [警告] 该块 GT 前景 {res['fg_ratio']:.3f}，但滑窗预测前景为 0 像素——"
              f"模型很可能没加载对（结构不匹配 / 输入归一化不一致），本块指标无意义")
    return res, img, gt, pred_a, pred_b, prob_b, lines


def write_block_artifacts(out_dir, img, gt, pred_a, pred_b, lines, res, args):
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"{res['split']} block r{res['r0']:03d}_c{res['c0']:03d}  "
           f"{res['rows']}x{res['cols']}  fg={res['fg_ratio']:.3f}  "
           f"bandIoU {res['a_band']:.3f}->{res['b_band']:.3f}")
    save_triplet(img_path(out_dir, "triplet", args), img, gt, pred_a, pred_b,
                 lines, size=args.panel_size, title=tag)
    save_seam_zoom(img_path(out_dir, "seam_zoom", args), img, gt, pred_a, pred_b,
                   half=args.seam_zoom_half)
    if args.save_npy:
        np.save(os.path.join(out_dir, "pred_tilewise.npy"), pred_a)
        np.save(os.path.join(out_dir, "pred_sliding.npy"), pred_b)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


CSV_FIELDS = [
    "split", "r0", "c0", "rows", "cols", "fg_ratio", "pred_fg_ratio",
    "a_all", "b_all", "d_all",
    "a_band", "b_band", "d_band",
    "a_interior", "b_interior", "d_interior",
    "jump_a", "jump_b", "jump_reduction",
    "flip_overall", "flip_in_band", "flip_interior",
    "agreement_iou", "agreement_pixel_acc",
    "n_win_a", "n_win_b", "redundancy", "t_a", "t_b",
]


def append_csv(path, res):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: res.get(k) for k in CSV_FIELDS})


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ms(rows, name, fmt="{:.4f}"):
    v = np.array([float(r[name]) for r in rows if r.get(name) not in (None, "")],
                 dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return "—"
    return fmt.format(v.mean()) + " ± " + fmt.format(v.std())


def _table(rows, args, heading=None):
    md = []
    if heading:
        md += [f"## {heading}", ""]
    md += [f"- split=`{args.split}` 块={args.rows}×{args.cols} 块数=**{len(rows)}** "
           f"overlap={args.overlap} blend={args.blend}", "",
           "| 指标 | 均值 ± 标准差 |", "|---|---|",
           f"| 前景 IoU（全图）A | {_ms(rows, 'a_all')} |",
           f"| 前景 IoU（全图）B | {_ms(rows, 'b_all')} |",
           f"| **前景 IoU（接缝带）A** | {_ms(rows, 'a_band')} |",
           f"| **前景 IoU（接缝带）B** | {_ms(rows, 'b_band')} |",
           f"| **接缝带增益 B−A** | {_ms(rows, 'd_band', '{:+.4f}')} |",
           f"| 前景 IoU（非接缝区）A | {_ms(rows, 'a_interior')} |",
           f"| 前景 IoU（非接缝区）B | {_ms(rows, 'b_interior')} |",
           f"| **跨缝跳变率 A** | {_ms(rows, 'jump_a')} |",
           f"| **跨缝跳变率 B** | {_ms(rows, 'jump_b')} |",
           f"| 跳变率比值 B/A | {_ms(rows, 'jump_reduction')} |",
           f"| 逐像素一致率 A/B | {_ms(rows, 'agreement_pixel_acc')} |", ""]
    d = np.array([float(r["d_band"]) for r in rows if r.get("d_band") not in (None, "")])
    d = d[~np.isnan(d)]
    if d.size:
        md.append(f"- **{int((d > 0).sum())}/{d.size}** 块的接缝带 IoU 被滑窗改善"
                  f"（{(d > 0).mean():.1%}），最差 {d.min():+.4f}，最好 {d.max():+.4f}")
    j = np.array([float(r["jump_reduction"]) for r in rows
                  if r.get("jump_reduction") not in (None, "")])
    j = j[~np.isnan(j)]
    if j.size:
        md.append(f"- 跨缝跳变率平均降至 {j.mean():.1%}（中位数 {np.median(j):.1%}）")
    return "\n".join(md) + "\n"


def aggregate_md(rows, args, elapsed=None, per_split=None):
    md = ["# 全量滑窗推断汇总", ""]
    if elapsed is not None:
        md.append(f"- 总耗时 {elapsed:.1f} s")
    md.append("")
    if per_split:
        for sp in per_split:
            sub = [r for r in rows if r["split"] == sp]
            if sub:
                md.append(_table(sub, args, heading=f"{sp}（{len(sub)} 块）"))
    md.append(_table(rows, args, heading="全部合并"))
    return "\n".join(md) + "\n"


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser(description="W3 大图滑窗推理验证（单块 / 全量批量）")
    ap.add_argument("--mode", choices=["single", "batch"], default="single")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--rows", type=int, default=None, help="块行数（瓦片数）")
    ap.add_argument("--cols", type=int, default=None, help="块列数（瓦片数）")
    ap.add_argument("--block", type=int, default=None, help="N×N 块，等价于 --rows N --cols N")
    ap.add_argument("--r0", type=int, default=None, help="块起始行（给了就不自动挑块）")
    ap.add_argument("--c0", type=int, default=None, help="块起始列")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--blend", choices=["hann", "mean"], default="hann")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="权重与模型结构不匹配时也强行继续（结果几乎必然是全图判背景，慎用）")
    ap.add_argument("--batch", type=int, default=8, help="推理时的 GPU batch 大小")
    ap.add_argument("--seam-half", type=int, default=2, help="接缝带半宽（像素）")
    ap.add_argument("--row-width", type=int, default=None, help="手动指定行宽，跳过自动检测")
    ap.add_argument("--limit", type=int, default=None, help="batch 模式每个 split 最多跑多少块")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已有 triplet.png 的块")
    ap.add_argument("--no-images", action="store_true", help="只算指标不出图（快很多）")
    ap.add_argument("--save-npy", action="store_true", help="额外保存预测 npy")
    ap.add_argument("--panel-size", type=int, default=512, help="每格缩略图边长（像素）")
    ap.add_argument("--jpg", action="store_true", help="出 jpg 而不是 png（体积约 1/5）")
    ap.add_argument("--seam-zoom-half", type=int, default=192,
                    help="接缝放大图半边长（原图像素）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.block:
        args.rows = args.cols = args.block
    args.rows = args.rows or 4
    args.cols = args.cols or 4

    if args.out is None:
        args.out = os.path.join(WS_ROOT, "_tools",
                                "sliding" if args.mode == "single" else "sliding_batch")
    os.makedirs(args.out, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    if args.mode == "single" and len(splits) > 1:
        raise SystemExit("single 模式请指定单个 --split")

    print("=" * 78)
    print(f"模式={args.mode}  split={args.split}  块={args.rows}×{args.cols}  设备={device}")
    print("=" * 78)
    print("加载 U-Net")
    model = build_unet(args.weights, device, allow_mismatch=args.allow_mismatch)
    infer = make_infer(model, device, args.batch)

    # ---------------- batch ----------------
    if args.mode == "batch":
        csv_path = os.path.join(args.out, "blocks.csv")
        t_start = time.perf_counter()
        done_splits = []
        for sp in splits:
            idx = index_tiles(sp)
            if not idx:
                print(f"[跳过] {sp} 下没有瓦片")
                continue
            if args.row_width:
                row_width = args.row_width
                print(f"   [行宽] {sp} 手动指定 {row_width}")
            elif sp == "train":
                row_width = TRAIN_ROW_WIDTH
                print(f"   [行宽] {sp} 已知 {row_width}（4736 = 64 × 74）")
            else:
                row_width, qual, note = detect_row_width(
                    sp, cache_path=os.path.join(args.out, f"_rowwidth_{sp}.json"))

            blocks, n_rows, skipped = iter_full_blocks(idx, row_width, args.rows, args.cols)
            if not blocks:
                print(f"[跳过] {sp}：行宽 {row_width} 下找不到完整块")
                continue
            cover = args.rows * args.cols * len(blocks)
            print(f"\n[{sp}] 瓦片 {len(idx)}  网格 {row_width}列×{n_rows}行  "
                  f"完整块 {len(blocks)} 个（覆盖 {cover}/{len(idx)} 张）"
                  f" 不完整块 {skipped} 个已跳过")
            todo = blocks[:args.limit] if args.limit else blocks
            n_ok = 0
            for bi, (r0, c0) in enumerate(todo):
                sub = os.path.join(args.out, "blocks", sp, f"r{r0:03d}_c{c0:03d}")
                if args.skip_existing and os.path.exists(img_path(sub, "triplet", args)):
                    continue
                res, img, gt, pred_a, pred_b, prob_b, lines = run_block(
                    infer, sp, row_width, r0, c0, args.rows, args.cols, args, device, idx)
                if not args.no_images:
                    write_block_artifacts(sub, img, gt, pred_a, pred_b, lines, res, args)
                append_csv(csv_path, res)
                n_ok += 1
                eta = (time.perf_counter() - t_start) / max(1, n_ok) * (len(todo) - bi - 1)
                print(f"[{sp} {bi + 1:4d}/{len(todo):4d}] r{r0:03d}_c{c0:03d} "
                      f"fg={res['fg_ratio']:.3f} | IoU {res['a_all']:.4f}->{res['b_all']:.4f} "
                      f"| band {res['a_band']:.4f}->{res['b_band']:.4f} ({res['d_band']:+.4f}) "
                      f"| jump {res['jump_a']:.4f}->{res['jump_b']:.4f} "
                      f"| {res['t_a'] + res['t_b']:.2f}s  ETA {eta / 60:.1f}min", flush=True)
            done_splits.append(sp)

        elapsed = time.perf_counter() - t_start
        all_rows = read_csv_rows(csv_path)
        md = aggregate_md(all_rows, args, elapsed, per_split=done_splits or None)
        with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print("=" * 78)
        print(md)
        print(f"输出目录 {args.out}")
        return

    # ---------------- single ----------------
    sp = splits[0]
    idx = index_tiles(sp)
    if not idx:
        raise SystemExit(f"{sp} 下没有瓦片")
    if args.row_width:
        row_width = args.row_width
    elif sp == "train":
        row_width = TRAIN_ROW_WIDTH
        print(f"   [行宽] {sp} 已知 {row_width}（4736 = 64 × 74）")
    else:
        row_width, _, _ = detect_row_width(
            sp, cache_path=os.path.join(args.out, f"_rowwidth_{sp}.json"))

    if args.r0 is not None and args.c0 is not None:
        r0, c0 = args.r0, args.c0
        ids = [i for row in block_ids(row_width, r0, c0, args.rows, args.cols) for i in row]
        miss = [i for i in ids if i not in idx]
        if miss:
            raise SystemExit(f"块内缺少瓦片编号 {miss[:8]}…")
        fg = float(np.mean([load_mask(os.path.join(
            split_dir(sp), LABEL_DIRNAME, idx[i] + ".tif")).mean() for i in ids]))
        print(f"   指定块 r{r0}_c{c0}，前景占比 {fg:.4f}")
    else:
        block = pick_block(sp, idx, args.rows, args.cols, row_width)
        r0, c0 = block["r0"], block["c0"]
        print(f"   自动挑块 r{r0}..{r0 + args.rows - 1} c{c0}..{c0 + args.cols - 1}，"
              f"前景占比 mean={block['mean']:.4f} std={block['std']:.4f}")

    # 退化测试：单张 512 瓦片，滑窗必须等价于直接推理
    img0, _ = make_mosaic(sp, row_width, r0, c0, 1, 1, idx)
    single = img0[:TILE, :TILE]
    pred_deg, _, info_deg = sliding_predict(infer, single, TILE, args.overlap, args.blend)
    pred_direct = infer(single[None])[0].argmax(0).astype(np.uint8)
    degenerate_ok = bool(np.array_equal(pred_deg, pred_direct))
    print(f"   退化测试：窗口数={info_deg['n_windows']}（应为 1）  逐像素一致={degenerate_ok}")

    res, img, gt, pred_a, pred_b, prob_b, lines = run_block(
        infer, sp, row_width, r0, c0, args.rows, args.cols, args, device, idx)
    res["degenerate_test"] = dict(n_windows=int(info_deg["n_windows"]),
                                  pixel_identical=degenerate_ok)

    for k in ("a_all", "b_all", "a_band", "b_band", "a_interior", "b_interior"):
        print(f"   FG IoU {k:12s} = {res[k]:.4f}")
    print(f"   翻转率 overall={res['flip_overall']:.5f} band={res['flip_in_band']:.5f} "
          f"interior={res['flip_interior']:.5f}")
    print(f"   A/B 一致率 {res['agreement_pixel_acc']:.5f}  IoU一致性 {res['agreement_iou']:.4f}")
    print(f"   跨缝跳变率 A={res['jump_a']:.4f} B={res['jump_b']:.4f}"
          f"（降低 {1 - res['jump_reduction']:.1%}）")
    print(f"   窗口 A={res['n_win_a']} B={res['n_win_b']} 冗余={res['redundancy']:.3f} "
          f"耗时 {res['t_a']:.2f}s / {res['t_b']:.2f}s")

    if not args.no_images:
        write_block_artifacts(args.out, img, gt, pred_a, pred_b, lines, res, args)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(aggregate_md([{k: v for k, v in res.items() if k in CSV_FIELDS}], args))
    print(f"   输出目录 {args.out}")
    print("=" * 78)
    print("完成")


if __name__ == "__main__":
    main()

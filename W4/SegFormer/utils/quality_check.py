#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""结果质检 —— W4 第②项（v1.3 新增）

对批量推理的产物做**自动异常检查**，输出一份人可读的报告 + 一份机可读的异常清单。
**模型无关**：只读 <out>/manifest.csv 和 <out>/masks/，不加载任何权重、不依赖 torch，
所以 U-Net / SegFormer / 以后换任何模型都能直接复用同一份脚本。

为什么要做这个：
    推理跑完不等于交付。真正会翻车的是"某几张图全黑""某张图糊成一整块""某条拼接缝
    那一列全是建筑"这类问题——它们不会抛异常，只会静默地把精度拉低。质检就是把它们
    在交付前抓出来。

四项主检查（对应计划书 W4 第②项）：
    1. 空图率        预测全空的图占比，逐图列出
    2. 面积分布      前景占比的稳健离群点（中位数 + IQR / MAD，不用均值标准差）
    3. 超大连通域    单个连通域面积超过全图比例阈值 -> 粘连、模型塌陷、拼接错位
    4. 边界贴边率    连通域触到图像边缘的面积占比 -> 拼接接缝、大范围误检

三项对账（不查不知道，查了才发现）：
    5. mask 值域/尺寸   是否真的是 uint8 {0,1}、尺寸对不对
    6. manifest 与磁盘  清单说有、磁盘没有（或反之）
    7. GT 对照          有 --gt 时额外算逐图 IoU / 漏检 / 误检 / 整体偏向

用法：
    :: 只做无 GT 的四项检查
    python utils\quality_check.py --out run\predict\batch_test

    :: 有 GT 时再叠上 IoU 对照（GT 文件名与输入图同 stem）
    python utils\quality_check.py --out run\predict\batch_test ^
        --gt ..\..\WUH_sub25\test\label

    :: 自定义阈值 + 只看前 30 条明细
    python utils\quality_check.py --out ... --big-cc 0.2 --border-frac 0.01 --top 30

产物：
    <out>/qc_report.md        人读：总览 + 每项检查明细
    <out>/qc_anomalies.csv    机读：一行一条异常（check, severity, key, value, threshold, note, src）
    <out>/qc_summary.json     汇总数值，方便以后画图/对比不同模型

退出码：0 = 跑完（有没有异常都返回 0）；2 = 输入不合法（manifest 不存在等）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_pct(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return ("%." + str(nd) + "f%%") % (100.0 * x)


def fmt_num(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return ("%." + str(nd) + "f") % x


def save_text_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def save_rows_atomic(path, fields, rows):
    tmp = path + ".tmp"
    # utf-8-sig：带 BOM，Excel 双击打开才不会把中文显示成乱码
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 读图 / 统计
# --------------------------------------------------------------------------- #
def load_mask_binary(path):
    """读掩膜 -> (uint8 0/1 数组, 备注)。备注非空表示遇到了需要报告的情况。

    兼容四种存法：
      - PIL mode '1'（1 位）      -> True 视作前景
      - L / I / I;16 单通道       -> >0 视作前景（0/255 与 0/1 都对）
      - P 调色板                  -> 用调色板还原成 RGB 后按"非全黑"判定
      - RGB / RGBA 彩色叠加图     -> 非全黑视作前景（并提示这不能算严格指标）
    """
    note = ""
    im = Image.open(path)
    mode = im.mode
    if mode == "1":
        arr = np.array(im, dtype=bool).astype(np.uint8)
    elif mode in ("L", "I", "I;16", "I;16B", "F"):
        a = np.array(im)
        a = np.nan_to_num(a.astype(np.float64))
        arr = (a > 0).astype(np.uint8)
        uniq = np.unique(a)
        if uniq.size > 2:
            note = "取值超过 2 种（%d 种），已按 >0 二值化" % uniq.size
        elif uniq.size == 2 and not set(uniq.tolist()) <= {0.0, 1.0}:
            note = "取值为 %s，已按 >0 二值化（疑似 0/255 存法）" % uniq.tolist()
    elif mode == "P":
        rgb = np.array(im.convert("RGB"))
        arr = (rgb.max(axis=2) > 0).astype(np.uint8)
        note = "调色板图，已按非黑判定前景"
    elif mode in ("RGB", "RGBA"):
        rgb = np.array(im.convert("RGB"))
        arr = (rgb.max(axis=2) > 0).astype(np.uint8)
        note = "彩色图当掩膜，已按非黑判定前景（不建议用于严格指标）"
    else:
        arr = (np.array(im.convert("L")) > 0).astype(np.uint8)
        note = "未知模式 %s，已转 L 后二值化" % mode
    return arr, note


def component_stats(mask):
    """连通域统计（8 邻接）。

    返回 dict:
      n_cc         连通域个数
      max_cc_ratio 最大连通域面积 / 全图面积
      border_ratio 触边连通域面积合计 / 全图面积
      n_border     触边连通域个数
      small_ratio  面积 < min_cc 的连通域个数 / 总个数（碎斑比例，诊断用）
    """
    h, w = mask.shape
    total = float(h * w)
    out = dict(n_cc=0, max_cc_ratio=0.0, border_ratio=0.0, n_border=0)
    if mask.max() == 0:
        return out
    if not _HAS_SCIPY:
        # 没有 scipy 就退化成"整幅当成一个连通域"，至少还能查超大与贴边
        out["n_cc"] = 1
        out["max_cc_ratio"] = float(mask.sum()) / total
        touch = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
        out["n_border"] = 1 if touch else 0
        out["border_ratio"] = float(mask.sum()) / total if touch else 0.0
        return out
    lab, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return out
    areas = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    areas[0] = 0.0
    border_ids = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    border_ids = border_ids[border_ids > 0]
    out["n_cc"] = int(n)
    out["max_cc_ratio"] = float(areas.max()) / total
    out["n_border"] = int(border_ids.size)
    out["border_ratio"] = float(areas[border_ids].sum()) / total if border_ids.size else 0.0
    return out


def robust_bounds(values, k=3.0):
    """稳健上下界：优先 median ± k*1.4826*MAD，MAD 退化时用 IQR，再退化用 1/99 分位。"""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size < 8:
        return None, None
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad > 1e-12:
        return med - k * mad, med + k * mad
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    if iqr > 1e-12:
        return float(q1) - k * 0.7413 * iqr, float(q3) + k * 0.7413 * iqr
    lo, hi = np.percentile(v, [1, 99])
    return float(lo), float(hi)


def iou_binary(pred, gt):
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    if union == 0:
        return None, 0.0, 0.0
    fp = float(np.logical_and(pred, ~gt.astype(bool)).sum())
    fn = float(np.logical_and(~pred.astype(bool), gt.astype(bool)).sum())
    return inter / union, fp / union, fn / union


# --------------------------------------------------------------------------- #
# 输入发现
# --------------------------------------------------------------------------- #
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def index_dir_by_stem(root):
    """建 stem -> 绝对路径 的索引（递归）。stem = 相对路径去扩展名，与 manifest 的 key 对齐。"""
    idx = {}
    root = os.path.abspath(root)
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(IMAGE_EXT):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            stem = os.path.splitext(rel)[0].replace("\\", "/")
            idx[stem] = full
            idx.setdefault(os.path.splitext(fn)[0], full)  # 扁平目录再挂一个纯文件名键
    return idx


def read_manifest(out_dir):
    path = os.path.join(out_dir, "manifest.csv")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 默认路径：从本文件所在位置向上找含 WUH_sub25 的目录，不把绝对路径写死
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data_root(start, name="WUH_sub25", max_up=6):
    """向上找数据根目录（含 WUH_sub25 的那层）；找不到返回 None。"""
    d = start
    for _ in range(max_up):
        if os.path.isdir(os.path.join(d, name)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


_DATA_ROOT = _find_data_root(_HERE)
DEFAULT_OUT = os.path.join(_HERE, "run", "predict", "batch")
DEFAULT_GT = (os.path.join(_DATA_ROOT, "WUH_sub25", "val", "label")
              if _DATA_ROOT else None)


def _pick_gt_by_manifest(rows, data_root):
    """--gt 没显式指定时，用 manifest 里的 key 去 val/test/train 三个标签目录里试，
    挑对得上的那个；对不上任何一半就返回 None（宁可跳过 GT 对照，也不要静默错配）。"""
    if not data_root:
        return None
    keys = set(r.get("key") for r in rows if r.get("key"))
    if not keys:
        return None
    best, best_n = None, 0
    for split in ("val", "test", "train"):
        d = os.path.join(data_root, "WUH_sub25", split, "label")
        if not os.path.isdir(d):
            continue
        idx = index_dir_by_stem(d)
        n = sum(1 for k in keys if k in idx)
        if n > best_n:
            best, best_n = d, n
    if best and best_n * 2 >= len(keys):
        return best
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="对批量推理产物做结果质检（空图率/面积分布/超大连通域/贴边率/GT对照）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # 注意：--out 不能同时写 required=True 和 default=…，argparse 只认 required，默认值会被忽略
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="batch_infer.py 的输出目录（含 manifest.csv 与 masks/）；"
                        "默认 <工程>/run/predict/batch")
    p.add_argument("--gt", default=DEFAULT_GT,
                   help="可选：GT 标签目录；默认自动找 <数据根>/WUH_sub25/val/label，"
                        "找不到就不做 GT 对照")
    p.add_argument("--top", default=15, type=int, help="每项检查在报告里最多列几条明细")
    # 阈值：默认值只是起点，正式用之前先看一遍分布再定（报告里会打印实际分布）
    p.add_argument("--empty-px", default=8, type=int, help="前景像素 <= 此值算空图")
    p.add_argument("--big-cc", default=0.25, type=float, help="单连通域面积占全图比例 > 此值算超大连通域")
    p.add_argument("--border-frac", default=0.15, type=float,
                   help="触边连通域面积合计占比 > 此值算异常（默认 0.15：按 80 张 val 瓦片实测校准，"
                        "P50 7.2%% / P90 29.0%%；512 瓦片上建筑被切开是常态，只有成片贴边才是接缝信号）")
    p.add_argument("--z", default=3.0, type=float, help="面积离群的稳健倍数（median ± z*MAD）")
    p.add_argument("--small-cc", default=20, type=int, help="像素数小于此值的连通域算碎斑（仅诊断）")
    p.add_argument("--iou-bad", default=0.30, type=float, help="逐图 IoU < 此值记 fatal")
    p.add_argument("--iou-warn", default=0.50, type=float, help="逐图 IoU < 此值记 warn")
    p.add_argument("--fg-shift", default=0.05, type=float, help="预测与 GT 的平均前景占比偏差超过此值告警")
    p.add_argument("--limit", default=0, type=int, help="只检查前 N 张（调试用）")
    p.add_argument("--report", default=None, help="报告路径，默认 <out>/qc_report.md")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out)
    rows = read_manifest(out_dir)
    if rows is None:
        print("[错误] 找不到 %s" % os.path.join(out_dir, "manifest.csv"))
        print("       --out 要指向 batch_infer.py 的输出目录")
        return 2
    if not rows:
        print("[错误] manifest.csv 是空的，说明还没有任何图跑成功")
        return 2
    # manifest 是追加日志：同一张图重跑一次就多一行。按 key 去重，只留最后一行（最新产物），
    # 否则重跑过的图会被重复统计、把空图率和面积分布带偏。
    _dedup = {}
    for _r in rows:
        _dedup[_r.get("key") or ""] = _r
    n_dup = len(rows) - len(_dedup)
    rows = list(_dedup.values())
    if n_dup:
        print("[质检] manifest 有 %d 行重复（同一张图重跑过），已按 key 只保留最新一行" % n_dup)
    if args.limit > 0:
        rows = rows[: args.limit]

    # --gt 用默认值时，按 manifest 的 key 自动挑对得上的标签目录；
    # 否则很容易出现"跑了 test 的 604 张、却拿 val 的标签去算 IoU"这种静默错配
    if args.gt and DEFAULT_GT and os.path.abspath(args.gt) == os.path.abspath(DEFAULT_GT):
        picked = _pick_gt_by_manifest(rows, _DATA_ROOT)
        if picked is None:
            print("[警告] --gt 未指定，默认标签目录与 manifest 对不上，已跳过 GT 对照；"
                  "要对照请显式传 --gt <标签目录>")
            args.gt = None
        elif os.path.abspath(picked) != os.path.abspath(args.gt):
            print("[质检] --gt 未指定，按 manifest 自动选用标签目录：%s" % picked)
            args.gt = picked

    gt_idx = index_dir_by_stem(args.gt) if args.gt else None
    if args.gt and not gt_idx:
        print("[警告] --gt 目录里没有找到任何图片：%s" % args.gt)

    print("[质检] 输出目录 %s" % out_dir)
    print("[质检] manifest %d 行 | GT %s" % (len(rows), args.gt if args.gt else "未提供"))
    if not _HAS_SCIPY:
        print("[警告] 没装 scipy，连通域只能粗算（整幅当一个域）")

    t0 = time.time()
    recs = []
    anomalies = []
    notes = []

    def add(check, severity, key, value, threshold, note, src=""):
        anomalies.append(dict(check=check, severity=severity, key=key,
                              value=value, threshold=threshold, note=note, src=src))

    n_missing = 0
    for i, row in enumerate(rows, 1):
        key = row.get("key") or ""
        src = row.get("src") or ""
        mask_rel = (row.get("mask") or "").strip()
        rec = dict(key=key, src=src, ok=False)

        if not mask_rel:
            n_missing += 1
            add("orphan", "warn", key, "manifest 无 mask 列", "-", "该行没记录掩膜路径", src)
            recs.append(rec)
            continue
        mask_path = mask_rel if os.path.isabs(mask_rel) else os.path.join(out_dir, mask_rel)
        if not os.path.isfile(mask_path):
            n_missing += 1
            add("orphan", "fatal", key, "掩膜文件缺失", mask_rel, "manifest 说有、磁盘上没有", src)
            recs.append(rec)
            continue

        try:
            arr, mnote = load_mask_binary(mask_path)
        except Exception as e:
            add("mask_bad", "fatal", key, "%s: %s" % (type(e).__name__, e), "-", "掩膜读不开", src)
            recs.append(rec)
            continue
        if mnote:
            notes.append("%s: %s" % (key, mnote))

        h, w = arr.shape
        fg_px = int(arr.sum())
        fg_ratio = fg_px / float(h * w)
        cc = component_stats(arr)
        rec.update(dict(ok=True, H=h, W=w, fg_px=fg_px, fg_ratio=fg_ratio, **cc))

        # 与 manifest 记录对账
        try:
            mh, mw = int(row.get("H") or -1), int(row.get("W") or -1)
            if (mh, mw) != (h, w) and mh > 0:
                add("shape_mismatch", "warn", key, "%dx%d" % (h, w), "%dx%d" % (mh, mw),
                    "掩膜实际尺寸与 manifest 不一致", src)
        except Exception:
            pass

        # --- 检查 1：空图 ---（判级放在 GT 对照之后：有 GT 才能区分「真负」与「整幅漏检」）
        rec["is_empty"] = fg_px <= args.empty_px

        # --- 检查 3：超大连通域 ---
        if cc["max_cc_ratio"] > args.big_cc:
            add("big_cc", "fatal", key, fmt_num(cc["max_cc_ratio"], 4), args.big_cc,
                "最大连通域占全图 %.1f%%（粘连/塌陷/接缝错位）" % (100 * cc["max_cc_ratio"]), src)

        # --- 检查 4：贴边 ---
        if cc["border_ratio"] > args.border_frac:
            sev = "fatal" if cc["border_ratio"] > 0.5 else "warn"
            add("border", sev, key, fmt_num(cc["border_ratio"], 4), args.border_frac,
                "%d 个连通域触边，占全图 %.1f%%" % (cc["n_border"], 100 * cc["border_ratio"]), src)

        # --- 检查 7：GT 对照 ---
        if gt_idx:
            gt_path = gt_idx.get(key) or gt_idx.get(os.path.splitext(os.path.basename(key))[0])
            if gt_path:
                try:
                    gt, _ = load_mask_binary(gt_path)
                except Exception as e:
                    add("mask_bad", "warn", key, "%s: %s" % (type(e).__name__, e), "-", "GT 读不开", gt_path)
                    gt = None
                if gt is not None:
                    if gt.shape != arr.shape:
                        add("shape_mismatch", "warn", key, "%dx%d" % (h, w), "%dx%d" % gt.shape,
                            "预测与 GT 尺寸不一致，跳过 IoU", src)
                    else:
                        iou, fp_r, fn_r = iou_binary(arr, gt)
                        gt_fg = float(gt.mean())
                        rec.update(dict(gt_fg_ratio=gt_fg, iou=iou, fp_ratio=fp_r, fn_ratio=fn_r))
                        if iou is None:
                            pass  # 两边都空，指标无定义
                        else:
                            if iou < args.iou_bad:
                                add("gt_iou", "fatal", key, fmt_num(iou, 4), args.iou_bad,
                                    "IoU 过低（FP %.1f%% / FN %.1f%%）" % (100 * fp_r, 100 * fn_r), src)
                            elif iou < args.iou_warn:
                                add("gt_iou", "warn", key, fmt_num(iou, 4), args.iou_warn,
                                    "IoU 偏低（FP %.1f%% / FN %.1f%%）" % (100 * fp_r, 100 * fn_r), src)
                            if rec["is_empty"] and gt_fg > args.empty_px / float(gt.size):
                                add("gt_missing", "fatal", key, fmt_num(gt_fg, 4), "-",
                                    "GT 有建筑但预测全空（整幅漏检）", src)
                            if (not rec["is_empty"]) and gt_fg * gt.size <= args.empty_px:
                                add("gt_extra", "warn", key, fmt_num(fg_ratio, 4), "-",
                                    "GT 全空但预测出建筑（整幅误检）", src)

        # --- 检查 1：空图判级（有 GT 时区分「真负」与「整幅漏检」）---
        if rec["is_empty"]:
            gt_fg = rec.get("gt_fg_ratio")
            if gt_fg is not None and gt_fg * arr.size <= args.empty_px:
                rec["empty_true_neg"] = True   # GT 也是空的：真负，不算异常
            else:
                add("empty", "warn", key, fg_px, args.empty_px,
                    "预测全空（未提供 GT，无法区分真负与漏检）" if gt_fg is None
                    else "预测全空且 GT 有建筑（整幅漏检）", src)

        recs.append(rec)
        if i % 200 == 0 or i == len(rows):
            print("[质检] %d/%d" % (i, len(rows)))

    # ---------------- 汇总 ----------------
    good = [r for r in recs if r.get("ok")]
    n = len(good)
    fg = np.array([r["fg_ratio"] for r in good], dtype=np.float64) if n else np.array([])
    # 面积离群只在「非空图」上判定：空图占比恒为 0，会把中位数与 MAD 一起拖垮
    fg_nonempty = np.array([r["fg_ratio"] for r in good if not r.get("is_empty")], dtype=np.float64)
    lo, hi = robust_bounds(fg_nonempty.tolist(), k=args.z) if fg_nonempty.size >= 8 else (None, None)
    if lo is not None and lo < 0:
        lo = 0.0

    # 检查 2：面积分布离群
    if lo is not None:
        for r in good:
            if r.get("is_empty"):
                continue
            if r["fg_ratio"] > hi:
                add("area", "warn", r["key"], fmt_num(r["fg_ratio"], 4), fmt_num(hi, 4),
                    "前景占比显著高于同批（疑似整片误检）", r["src"])
            elif r["fg_ratio"] < lo:
                add("area", "warn", r["key"], fmt_num(r["fg_ratio"], 4), fmt_num(lo, 4),
                    "前景占比显著低于同批（疑似漏检）", r["src"])

    n_empty = sum(1 for r in good if r.get("is_empty"))
    n_big = sum(1 for r in good if r["max_cc_ratio"] > args.big_cc)
    n_border = sum(1 for r in good if r["border_ratio"] > args.border_frac)
    ious = [r.get("iou") for r in good if r.get("iou") is not None]
    shifts = [r["fg_ratio"] - r["gt_fg_ratio"] for r in good if r.get("gt_fg_ratio") is not None]
    mean_shift = float(np.mean(shifts)) if shifts else None

    if mean_shift is not None and abs(mean_shift) > args.fg_shift:
        add("gt_bias", "warn", "<全体>", fmt_num(mean_shift, 4), args.fg_shift,
            "预测前景占比整体%s %.1f 个百分点（系统性%s报）"
            % ("偏多" if mean_shift > 0 else "偏少", 100 * abs(mean_shift),
               "多" if mean_shift > 0 else "漏"),
            "%d 张" % len(shifts))

    summary = dict(
        out_dir=out_dir, gt=args.gt, n_manifest=len(rows), n_checked=n, n_mask_missing=n_missing,
        empty_rate=(n_empty / n if n else None), n_empty=n_empty,
        n_empty_true_neg=sum(1 for r in good if r.get("empty_true_neg")),
        fg_mean=float(fg.mean()) if n else None, fg_median=float(np.median(fg)) if n else None,
        fg_p10=float(np.percentile(fg, 10)) if n else None,
        fg_p90=float(np.percentile(fg, 90)) if n else None,
        fg_outlier_lo=lo, fg_outlier_hi=hi,
        big_cc_rate=(n_big / n if n else None), n_big=n_big,
        border_rate=(n_border / n if n else None), n_border=n_border,
        iou_median=float(np.median(ious)) if ious else None,
        iou_p10=float(np.percentile(ious, 10)) if ious else None,
        fg_bias_mean=mean_shift,
        n_anomaly=len(anomalies),
        n_fatal=sum(1 for a in anomalies if a["severity"] == "fatal"),
        n_warn=sum(1 for a in anomalies if a["severity"] == "warn"),
        thresholds=dict(empty_px=args.empty_px, big_cc=args.big_cc, border_frac=args.border_frac,
                        z=args.z, iou_bad=args.iou_bad, iou_warn=args.iou_warn,
                        fg_shift=args.fg_shift),
        elapsed_s=round(time.time() - t0, 1), ts=now_str())

    # ---------------- 写产物 ----------------
    rep_path = args.report or os.path.join(out_dir, "qc_report.md")
    anom_path = os.path.join(out_dir, "qc_anomalies.csv")
    json_path = os.path.join(out_dir, "qc_summary.json")

    save_rows_atomic(anom_path, ["check", "severity", "key", "value", "threshold", "note", "src"], anomalies)
    save_text_atomic(json_path, json.dumps(summary, ensure_ascii=False, indent=2))
    save_text_atomic(rep_path, render_report(args, summary, recs, good, anomalies, notes))

    # ---------------- 控制台 ----------------
    print("-" * 74)
    print("[质检] 已检查 %d 张 | 空图 %d（%s）| 超大连通域 %d（%s）| 贴边 %d（%s）"
          % (n, n_empty, fmt_pct(summary["empty_rate"]), n_big, fmt_pct(summary["big_cc_rate"]),
             n_border, fmt_pct(summary["border_rate"])))
    if ious:
        print("[质检] GT 对照：IoU 中位 %s、P10 %s、前景占比偏差 %s"
              % (fmt_num(summary["iou_median"]), fmt_num(summary["iou_p10"]), fmt_num(mean_shift, 4)))
    print("[质检] 异常 %d 条（fatal %d / warn %d）"
          % (len(anomalies), summary["n_fatal"], summary["n_warn"]))
    print("[质检] 报告   %s" % rep_path)
    print("[质检] 清单   %s" % anom_path)
    print("[质检] 汇总   %s" % json_path)
    print("[质检] 用时 %ss" % summary["elapsed_s"])
    return 0


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def _table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_report(args, s, recs, good, anomalies, notes):
    L = []
    L.append("# 结果质检报告")
    L.append("")
    L.append("> 生成时间 %s ｜ 输出目录 `%s` ｜ 用时 %ss" % (s["ts"], s["out_dir"], s["elapsed_s"]))
    L.append(">")
    L.append("> 本脚本**不加载模型**，只检查已经落盘的掩膜，所以同一份报告可以跨模型对比。")
    L.append("")

    L.append("## 0. 结论")
    L.append("")
    verdict = []
    if s["n_fatal"]:
        verdict.append("**有 %d 条致命异常，交付前必须逐条看**（清单见 `qc_anomalies.csv` 的 `severity=fatal`）" % s["n_fatal"])
    else:
        verdict.append("没有致命异常")
    n_tn = s.get("n_empty_true_neg") or 0
    real_empty = (s["n_empty"] - n_tn) / float(s["n_checked"]) if s["n_checked"] else 0.0
    if s["n_empty"] and n_tn:
        verdict.append("空图 %d 张里有 %d 张 GT 也是空的（真负），真正需要看的是剩下 %d 张"
                       % (s["n_empty"], n_tn, s["n_empty"] - n_tn))
    if real_empty > 0.05:
        verdict.append("空图率 %s 偏高，先确认这些图是真的没有建筑、还是整幅漏检" % fmt_pct(real_empty))
    if s["border_rate"] is not None and s["border_rate"] > 0.15:
        verdict.append("贴边异常占 %s，优先怀疑拼接接缝（滑窗融合）而不是模型能力" % fmt_pct(s["border_rate"]))
    for v in verdict:
        L.append("- " + v)
    L.append("")

    _sec = [0]

    def next_sec(title):
        """章节号按实际输出顺序自增：某项检查没有异常时不会留下空档。"""
        _sec[0] += 1
        L.append("## %d. %s" % (_sec[0], title))

    next_sec("总览")
    L.append("")
    L.append(_table(["指标", "数值", "说明"], [
        ["manifest 行数", s["n_manifest"], "清单声称跑完的张数"],
        ["实际检查", s["n_checked"], "掩膜能读开并统计的张数"],
        ["掩膜缺失", s["n_mask_missing"], "清单有、磁盘没有 —— 非 0 就是续跑判据出了问题"],
        ["空图率", "%s (%d 张)" % (fmt_pct(s["empty_rate"]), s["n_empty"]),
         "前景像素 ≤ %d；其中 %d 张 GT 也为空（真负，不算异常）"
         % (s["thresholds"]["empty_px"], s.get("n_empty_true_neg") or 0)],
        ["前景占比 中位 / 均值", "%s / %s" % (fmt_pct(s["fg_median"]), fmt_pct(s["fg_mean"])), "P10–P90: %s – %s" % (fmt_pct(s["fg_p10"]), fmt_pct(s["fg_p90"]))],
        ["面积离群区间",
         ("%s – %s" % (fmt_pct(s["fg_outlier_lo"]), fmt_pct(s["fg_outlier_hi"])))
         if s.get("fg_outlier_lo") is not None else "样本不足（< 8 张），未判定",
         "median ± %.1f×MAD" % s["thresholds"]["z"]],
        ["超大连通域", "%s (%d 张)" % (fmt_pct(s["big_cc_rate"]), s["n_big"]), "单域 > %s 全图" % fmt_pct(s["thresholds"]["big_cc"], 0)],
        ["贴边异常", "%s (%d 张)" % (fmt_pct(s["border_rate"]), s["n_border"]), "触边域合计 > %s 全图" % fmt_pct(s["thresholds"]["border_frac"], 0)],
        ["GT IoU 中位 / P10", "%s / %s" % (fmt_num(s["iou_median"]), fmt_num(s["iou_p10"])), "仅 --gt 时" ],
        ["前景占比偏差", fmt_num(s["fg_bias_mean"], 4), "正=偏多报，负=偏漏报（仅 --gt 时）"],
        ["异常条数", "%d（fatal %d / warn %d）" % (s["n_anomaly"], s["n_fatal"], s["n_warn"]), "见 qc_anomalies.csv"],
    ]))
    L.append("")

    # 分布
    fg = np.array([r["fg_ratio"] for r in good], dtype=np.float64) if good else np.array([])
    if fg.size:
        L.append("### 前景占比分布")
        L.append("")
        edges = [0, 0.001, 0.01, 0.05, 0.15, 0.30, 0.50, 0.80, 1.0001]
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            c = int(((fg >= a) & (fg < b)).sum())
            if c:
                rows.append(["%s – %s" % (fmt_pct(a, 1), fmt_pct(b, 1)), c, fmt_pct(c / float(fg.size), 1)])
        L.append(_table(["区间", "张数", "占比"], rows))
        L.append("")

    # 每项检查明细（标题不带编号，编号由 next_sec 按实际输出顺序给）
    order = [("empty", "空图", "预测前景像素 ≤ 阈值的图"),
             ("area", "面积分布离群", "前景占比落在 median ± z×MAD 之外的图"),
             ("big_cc", "超大连通域", "单个 8 邻接连通域面积占全图比例过高：粘连、塌陷、或拼接错位"),
             ("border", "贴边异常", "触边连通域面积合计占比过高：优先怀疑滑窗融合的接缝"),
             ("gt_iou", "GT 对照：IoU 偏低", "逐图 IoU 低于阈值的图"),
             ("gt_missing", "整幅漏检", "GT 有建筑、预测全空"),
             ("gt_extra", "整幅误检", "GT 全空、预测却有建筑"),
             ("gt_bias", "整体偏向", "预测与 GT 的前景占比均值差"),
             ("mask_bad", "掩膜读不开 / 值域异常", "文件损坏或存法不对"),
             ("orphan", "清单与磁盘不一致", "manifest 说有、磁盘没有（或没有 mask 列）"),
             ("shape_mismatch", "尺寸不一致", "掩膜尺寸与 manifest / GT 对不上")]
    for check, title, desc in order:
        items = [a for a in anomalies if a["check"] == check]
        if not items:
            continue
        items.sort(key=lambda a: (a["severity"] != "fatal", a["key"]))
        next_sec(title)
        L.append("")
        L.append("%s（%d 条）" % (desc, len(items)))
        L.append("")
        rows = []
        for a in items[: args.top]:
            rows.append([a["severity"], "`%s`" % a["key"], a["value"], a["threshold"], a["note"]])
        L.append(_table(["级别", "key", "实测", "阈值", "说明"], rows))
        if len(items) > args.top:
            L.append("")
            L.append("… 其余 %d 条见 `qc_anomalies.csv`" % (len(items) - args.top))
        L.append("")

    # 最差样本
    if good:
        by_iou = [r for r in good if r.get("iou") is not None]
        if by_iou:
            by_iou.sort(key=lambda r: r["iou"])
            next_sec("最差 10 张（按逐图 IoU）")
            L.append("")
            L.append(_table(["key", "IoU", "预测前景", "GT 前景", "FP 占并集", "FN 占并集", "连通域数"],
                            [["`%s`" % r["key"], fmt_num(r["iou"]), fmt_pct(r["fg_ratio"]),
                              fmt_pct(r.get("gt_fg_ratio")), fmt_pct(r["fp_ratio"], 1),
                              fmt_pct(r["fn_ratio"], 1), r["n_cc"]] for r in by_iou[:10]]))
            L.append("")
        by_cc = sorted(good, key=lambda r: -r["max_cc_ratio"])[:10]
        if by_cc and by_cc[0]["max_cc_ratio"] > 0:
            next_sec("最大连通域 Top 10")
            L.append("")
            L.append(_table(["key", "最大域占全图", "连通域数", "触边域数", "前景占比"],
                            [["`%s`" % r["key"], fmt_pct(r["max_cc_ratio"], 1), r["n_cc"],
                              r["n_border"], fmt_pct(r["fg_ratio"])] for r in by_cc]))
            L.append("")

    if notes:
        next_sec("掩膜存法提醒")
        L.append("")
        L.append("下列掩膜的位深/存法与「严格 uint8 0/1」不完全一致，已按 >0 二值化后统计：")
        L.append("")
        for t in notes[:20]:
            L.append("- %s" % t)
        if len(notes) > 20:
            L.append("- … 共 %d 条" % len(notes))
        L.append("")

    next_sec("阈值与口径")
    L.append("")
    L.append(_table(["参数", "本次取值", "怎么定"], [
        ["`--empty-px`", s["thresholds"]["empty_px"], "小于几个像素就算空图；瓦片 512² 时 8 px 约等于 0.003%"],
        ["`--big-cc`", s["thresholds"]["big_cc"], "单栋楼很少超过全图 25%；超过基本是粘连或塌陷"],
        ["`--border-frac`", s["thresholds"]["border_frac"], "去掉正常压边的小楼，只看成片的贴边区域"],
        ["`--z`", s["thresholds"]["z"], "面积离群的稳健倍数，用 MAD 不用标准差（抗离群）"],
        ["`--iou-bad` / `--iou-warn`", "%s / %s" % (s["thresholds"]["iou_bad"], s["thresholds"]["iou_warn"]), "逐图 IoU 的两级警戒线"],
        ["`--fg-shift`", s["thresholds"]["fg_shift"], "整体偏向告警阈值（5 个百分点）"],
    ]))
    L.append("")
    L.append("**连通域口径**：8 邻接；`border_ratio` 只统计「触到图像最外一圈像素」的连通域。")
    L.append("")
    L.append("**这些阈值是起点，不是真理**：正式对外报数前，先用本报告打印的分布看一眼，再决定卡在哪。")
    L.append("面试被问到「你的异常判定阈值怎么来的」，正确回答是「先统计这批数据的分布，再按 IQR/MAD 定离群线，"
             "最后人工复核前 N 张确认没有误杀」，而不是「我拍了个 0.25」。")
    L.append("")
    next_sec("复现命令")
    L.append("")
    L.append("```powershell")
    cmd = "python utils\\quality_check.py --out \"%s\"" % os.path.basename(s["out_dir"])
    if s["gt"]:
        cmd += " --gt \"%s\"" % s["gt"]
    L.append(cmd)
    L.append("```")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())

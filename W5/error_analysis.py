# -*- coding: utf-8 -*-
"""W5 错误分析：把 test 集上的误差拆成"可归因的类型"，并挑出典型案例出图。

设计思路
--------
不看"哪张图 IoU 最低"，而是看"错在哪种形态上"。分三步：

1) 像素级分解（二分类，ground truth 是硬的）
       TP = pred ∩ gt
       FN = gt ∩ ¬pred      —— 该找到没找到（漏检）
       FP = pred ∩ ¬gt      —— 不该找到却找到了（误检/外扩/粘连）

   再把 FN/FP 按"离 GT 边界的距离"再切一刀：
       FN 在 GT 腐蚀核内  → **物体级漏检**（整栋/整块没出来）
       FN 在 GT 边界带内  → **边界带漏检**（形状在，但内缩了）
       FP 距 GT ≤ BAND    → **边界外扩**（形状在，但胖了）
       FP 距 GT ≤ NEAR    → **近邻粘连**（两栋之间被填死）
       FP 距 GT > NEAR    → **孤立误检**（凭空多出来一块）

2) 形态学归因（只对"物体级漏检"和"孤立误检"的连通域做）
       面积 / 最小外接矩形的长宽比 / 凸包实心度 / 平均亮度
       → 判成：小目标漏检 / 条状(围墙·长条建筑)漏检 / 大块漏检 / 阴影暗顶 /
               线性误检(道路·田埂) / 大面积误检(裸地·硬化地面) / 高亮误检 / 一般

3) 边界质量单独量化（BF-score）
       预测边界与 GT 边界在 ±BAND 像素内互匹配的比例 → BF1
       用来回答"边界模糊到底有多糊"，而不是靠眼睛看

这些规则是**启发式**，阈值全部可以在文件末尾 MANUAL 里改；报告里会如实写明
"自动归因 + 人工复核 N 例"，不假装是全自动的语义理解。

用法
----
    cd W5
    D:\\miniconda3\\envs\\unet\\python.exe error_analysis.py
    # 指定预测目录 / 只看前 100 张 / 换 tolerance
    ... error_analysis.py --pred <目录> --limit 100 --band 5

输出（默认 ./out/）
----
    error_stats.json        全部聚合数字（报告的数据源）
    error_images.csv        逐图一行
    error_components.csv    逐连通域一行（可拿去再统计）
    summary.md              自动生成的数字汇总（中文表格）
    cases/<类型>_<stem>.png  典型案例六格图
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, OrderedDict

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as cc_label, distance_transform_edt

# ---------------------------------------------------------------------------
# 类型名（顺序即报告里的呈现顺序）
# ---------------------------------------------------------------------------
FN_BAND = "边界带漏检(内缩)"
FN_SMALL = "物体漏检-小目标"
FN_THIN = "物体漏检-条状(围墙/长条)"
FN_BIG = "物体漏检-大块"
FN_DARK = "物体漏检-阴影/暗顶"
FN_GEN = "物体漏检-一般"
FP_BAND = "误检-边界外扩"
FP_NEAR = "误检-近邻粘连"
FP_LINE = "误检-线性(道路/田埂)"
FP_DARK = "误检-阴影"
FP_BIG = "误检-大面积(裸地/硬化)"
FP_BRIGHT = "误检-高亮(白屋顶/裸岩)"
FP_GEN = "误检-一般"
FN_NOISE = "漏检-碎屑(<MIN_CC)"
FP_NOISE = "误检-噪点(<MIN_CC)"

TYPE_ORDER = [
    FN_BAND, FN_SMALL, FN_THIN, FN_BIG, FN_DARK, FN_GEN, FN_NOISE,
    FP_BAND, FP_NEAR, FP_LINE, FP_DARK, FP_BIG, FP_BRIGHT, FP_GEN, FP_NOISE,
]

CN2ASCII = {
    FN_BAND: "fn_band", FN_SMALL: "fn_small", FN_THIN: "fn_thin", FN_BIG: "fn_big",
    FN_DARK: "fn_dark", FN_GEN: "fn_general", FN_NOISE: "fn_noise",
    FP_BAND: "fp_band", FP_NEAR: "fp_near", FP_LINE: "fp_line", FP_DARK: "fp_dark",
    FP_BIG: "fp_big", FP_BRIGHT: "fp_bright", FP_GEN: "fp_general", FP_NOISE: "fp_noise",
}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


# ---------------------------------------------------------------------------
# 基础 IO（用 PIL 读，避开 cv2.imread 对非 ASCII 路径的限制）
# ---------------------------------------------------------------------------
def read_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: str) -> np.ndarray:
    """标签/预测统一成 0/1 的 uint8。

    注意：WHU 标签是 PIL mode='1'（bool），绝不能 convert('L')——那会变成 0/255。
    这里直接 np.asarray 拿原始值再 >0，0/1 与 0/255 两种存法都能吃。
    """
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return (a > 0).astype(np.uint8)


def resolve_mask_dir(d: str) -> str:
    """batch_infer 的输出是 <out>/masks/*_mask.tif，这里自动往里钻一层。"""
    if d and os.path.isdir(os.path.join(d, "masks")):
        return os.path.join(d, "masks")
    return d


def find_mask(pred_dir: str, stem: str) -> str | None:
    for suf in ("_mask.png", "_mask.tif", "_label.tif", ".png", ".tif"):
        p = os.path.join(pred_dir, stem + suf)
        if os.path.exists(p):
            return p
    # 后备：目录里找以 stem 开头的第一个图
    for name in sorted(os.listdir(pred_dir)):
        if name.startswith(stem) and name.lower().endswith((".png", ".tif", ".tiff")):
            return os.path.join(pred_dir, name)
    return None


# ---------------------------------------------------------------------------
# 形态学工具
# ---------------------------------------------------------------------------
def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def boundary_of(m: np.ndarray) -> np.ndarray:
    """1 像素宽的轮廓（内部边界）。"""
    if m.max() == 0:
        return np.zeros_like(m)
    return (m - cv2.erode(m, disk(1))).astype(np.uint8)


def comp_features(comp: np.ndarray, v_channel: np.ndarray) -> dict:
    ys, xs = np.nonzero(comp)
    area = int(len(xs))
    pts = np.column_stack([xs, ys]).astype(np.int32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    maj, mino = float(max(w, h)), float(max(min(w, h), 1e-6))
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else float(area)
    v = v_channel[comp > 0]
    return dict(
        area=area,
        w=float(w), h=float(h),
        elong=maj / mino,
        min_side=mino,
        solidity=area / hull_area if hull_area > 0 else 1.0,
        v_med=float(np.median(v)) if v.size else 0.0,
        cx=float(xs.mean()), cy=float(ys.mean()),
        x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()),
    )


def classify_fn_obj(f: dict, cfg: dict) -> str:
    """物体级漏检的形态归因（优先级：暗 > 条状 > 小 > 大 > 一般）。"""
    if f["v_med"] < cfg["dark_v"]:
        return FN_DARK
    if f["elong"] >= cfg["elong"] and f["min_side"] <= cfg["thin_side"]:
        return FN_THIN
    if f["area"] < cfg["small_area"]:
        return FN_SMALL
    if f["area"] >= cfg["big_area"]:
        return FN_BIG
    return FN_GEN


def classify_fp_iso(f: dict, cfg: dict) -> str:
    """孤立误检的形态归因（优先级：线性 > 暗 > 大面积 > 高亮 > 一般）。"""
    if f["elong"] >= cfg["elong"] and f["min_side"] <= cfg["thin_side"]:
        return FP_LINE
    if f["v_med"] < cfg["dark_v"]:
        return FP_DARK
    if f["area"] >= cfg["big_area"]:
        return FP_BIG
    if f["v_med"] >= cfg["bright_v"] and f["solidity"] >= 0.85:
        return FP_BRIGHT
    return FP_GEN


def bfs(pred: np.ndarray, gt: np.ndarray, band: int) -> tuple[float, float, float]:
    """边界 F1（Boundary F-score）：预测轮廓与 GT 轮廓在 ±band 内互匹配。"""
    pb, gb = boundary_of(pred), boundary_of(gt)
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0, 0.0, 0.0
    d_gt = distance_transform_edt(gb == 0)
    d_pr = distance_transform_edt(pb == 0)
    p = float((d_gt[pb > 0] <= band).mean())
    r = float((d_pr[gb > 0] <= band).mean())
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# 单图分析
# ---------------------------------------------------------------------------
def analyze_one(stem: str, pred: np.ndarray, gt: np.ndarray, rgb: np.ndarray,
                cfg: dict) -> tuple[dict, list]:
    band, near = cfg["band"], cfg["near"]
    v = rgb.max(axis=2).astype(np.float32)          # HSV 的 V 通道（等价于 max(R,G,B)）

    gt_len = int(gt.sum())
    pred_len = int(pred.sum())
    inter = int(np.logical_and(gt, pred).sum())
    union = gt_len + pred_len - inter
    iou = inter / union if union else 1.0
    prec = inter / pred_len if pred_len else 1.0
    rec = inter / gt_len if gt_len else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    _, _, bf1 = bfs(pred, gt, band)

    n_gt_cc = int(cc_label(gt)[1]) if gt_len else 0
    n_pred_cc = int(cc_label(pred)[1]) if pred_len else 0

    # ---- 像素级分解 ----
    fn = np.logical_and(gt == 1, pred == 0)
    fp = np.logical_and(pred == 1, gt == 0)
    gt_core = cv2.erode(gt, disk(band)) if gt_len else gt
    fn_obj = np.logical_and(fn, gt_core == 1)
    fn_bnd = np.logical_and(fn, gt_core == 0)

    d_gt = distance_transform_edt(gt == 0) if gt_len else np.full(gt.shape, 1e6, np.float32)
    fp_band = np.logical_and(fp, d_gt <= band)
    fp_near = np.logical_and(fp, np.logical_and(d_gt > band, d_gt <= near))
    fp_iso = np.logical_and(fp, d_gt > near)

    comps = []
    type_px = Counter()
    type_cc = Counter()

    type_px[FN_BAND] += int(fn_bnd.sum())

    def harvest(mask: np.ndarray, kind: str, classifier) -> None:
        lbl, n = cc_label(mask)
        if n == 0:
            return
        seen = 0
        for i in range(1, n + 1):
            comp = (lbl == i)
            area = int(comp.sum())
            if area < cfg["min_cc"]:
                type_px[FN_NOISE if kind == "fn" else FP_NOISE] += area
                type_cc[FN_NOISE if kind == "fn" else FP_NOISE] += 1
                continue
            f = comp_features(comp, v)
            t = classifier(f, cfg) if classifier else kind
            type_px[t] += area
            type_cc[t] += 1
            seen += 1
            if seen <= cfg["max_comps_keep"]:
                comps.append(dict(stem=stem, kind=kind, type=t, **f))
        return

    harvest(fn_obj, "fn", classify_fn_obj)
    harvest(fp_iso, "fp", classify_fp_iso)

    type_px[FP_BAND] += int(fp_band.sum())
    type_px[FP_NEAR] += int(fp_near.sum())

    row = dict(
        stem=stem, gt_fg=gt_len, pred_fg=pred_len, iou=iou, prec=prec, rec=rec,
        f1=f1, bf1=bf1, n_gt_cc=n_gt_cc, n_pred_cc=n_pred_cc,
        fn_px=int(fn.sum()), fp_px=int(fp.sum()),
        fn_bnd=int(fn_bnd.sum()), fn_obj=int(fn_obj.sum()),
        fp_band=int(fp_band.sum()), fp_near=int(fp_near.sum()),
        fp_iso=int(fp_iso.sum()),
        v_mean=float(v.mean()), v_med=float(np.median(v)),
        is_empty=(gt_len == 0),
        empty_false_alarm=int(pred_len) if gt_len == 0 else 0,
    )
    return row, comps, type_px, type_cc


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------
def get_font(size: int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _to_img(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def panel_gt(m: np.ndarray) -> Image.Image:
    return _to_img(np.where(m > 0, 255, 0).astype(np.uint8)).convert("RGB")


def panel_err(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    out = np.full(gt.shape + (3,), 40, np.uint8)
    tp = np.logical_and(gt == 1, pred == 1)
    fn = np.logical_and(gt == 1, pred == 0)
    fp = np.logical_and(pred == 1, gt == 0)
    out[tp] = (200, 200, 200)
    out[fn] = (235, 70, 70)
    out[fp] = (70, 130, 235)
    return _to_img(out)


def panel_outline(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    out = rgb.copy()
    g_b = boundary_of(gt) > 0
    p_b = boundary_of(pred) > 0
    out[g_b] = (60, 230, 60)      # GT 轮廓：绿
    out[p_b] = (255, 60, 220)     # 预测轮廓：洋红
    return _to_img(out)


SHORT = {
    FN_SMALL: "漏·小目标", FN_THIN: "漏·条状", FN_BIG: "漏·大块", FN_DARK: "漏·暗顶",
    FN_GEN: "漏·一般", FN_NOISE: "漏·碎屑",
    FP_LINE: "多·线性", FP_DARK: "多·阴影", FP_BIG: "多·大面积", FP_BRIGHT: "多·高亮",
    FP_GEN: "多·一般", FP_NOISE: "多·噪点",
}


def panel_types(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                comps: list, cfg: dict) -> Image.Image:
    """错误连通域归因视图：先把 FN/FP 半透明染色，再框出每个被归因的连通域。"""
    fn = np.logical_and(gt == 1, pred == 0)
    fp = np.logical_and(pred == 1, gt == 0)
    out = rgb.astype(np.float32)
    out[fn] = out[fn] * 0.45 + np.array([255, 60, 60], np.float32) * 0.55
    out[fp] = out[fp] * 0.45 + np.array([80, 170, 255], np.float32) * 0.55
    img = _to_img(out)
    d = ImageDraw.Draw(img)
    font = get_font(15)
    for c in comps:
        col = (255, 60, 60) if c["kind"] == "fn" else (80, 170, 255)
        d.rectangle([c["x0"], c["y0"], c["x1"], c["y1"]], outline=col, width=2)
        tag = SHORT.get(c["type"], c["type"][:6])
        ty = c["y0"] - 17 if c["y0"] >= 18 else c["y1"] + 2
        d.text((c["x0"] + 2, ty), tag, fill=(255, 255, 255), font=font)
    return img


def make_figure(path: str, stem: str, rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                row: dict, comps: list, cfg: dict, subtitle: str) -> None:
    panels = [
        (panel_outline(rgb, gt, pred), "① 原图 + GT轮廓(绿) / 预测轮廓(洋红)"),
        (panel_gt(gt), "② GT"),
        (panel_gt(pred), "③ 预测"),
        (panel_err(gt, pred), "④ 错误图: TP灰 FN红 FP蓝"),
        (panel_types(rgb, gt, pred, comps, cfg), "⑤ 错误连通域归因(半透明染色+框+类型)"),
        (_to_img(rgb), "⑥ 原图"),
    ]
    h, w = gt.shape
    cols, rows = 3, 2
    cap_h, title_h = 26, 62
    fig = Image.new("RGB", (w * cols + 8 * (cols + 1), title_h + (h + cap_h) * rows + 8), (24, 24, 24))
    d = ImageDraw.Draw(fig)
    f_title, f_sub, f_cap = get_font(20), get_font(15), get_font(14)
    d.text((8, 8), f"{stem}   IoU={row['iou']:.4f}  F1={row['f1']:.4f}  BF1={row['bf1']:.4f}", fill=(255, 255, 255), font=f_title)
    d.text((8, 36), subtitle[:150], fill=(180, 220, 255), font=f_sub)
    for i, (im, cap) in enumerate(panels):
        r, c = divmod(i, cols)
        x = 8 + c * (w + 8)
        y = title_h + r * (h + cap_h)
        fig.paste(im.resize((w, h)), (x, y))
        d.text((x, y + h + 4), cap, fill=(200, 200, 200), font=f_cap)
    fig.save(path)


# ---------------------------------------------------------------------------
# 聚合 & 报告片段
# ---------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_table(headers: list, rows: list) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="W5 错误分析：误差分解 + 形态归因 + 案例出图")
    ap.add_argument("--pred", required=True, help="预测 mask 目录（*_mask.png，0/1）")
    ap.add_argument("--pred2", default=None, help="第二个模型的预测目录（可选，做互补性对比）")
    ap.add_argument("--image", default=None, help="影像目录")
    ap.add_argument("--label", default=None, help="标签目录")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--limit", default=0, type=int, help=">0 时只跑前 N 张")
    ap.add_argument("--band", default=3, type=int, help="边界带宽度（像素）")
    ap.add_argument("--near", default=15, type=int, help="近邻判定距离（像素）")
    ap.add_argument("--small-area", default=300, type=int, help="小目标面积上限（像素）")
    ap.add_argument("--big-area", default=5000, type=int, help="大块面积阈值（像素）")
    ap.add_argument("--elong", default=4.0, type=float, help="条状判定长宽比")
    ap.add_argument("--thin-side", default=12.0, type=float, help="条状判定短边上限")
    ap.add_argument("--dark-v", default=70.0, type=float, help="暗/阴影亮度上限（0-255）")
    ap.add_argument("--bright-v", default=195.0, type=float, help="高亮亮度下限")
    ap.add_argument("--min-cc", default=30, type=int, help="小于此面积记为碎屑/噪点")
    ap.add_argument("--max-cases", default=18, type=int, help="出多少个案例图")
    ap.add_argument("--max-comps-keep", default=60, type=int, help="每张图最多保留多少连通域明细")
    args = ap.parse_args()

    cfg = dict(band=args.band, near=args.near, small_area=args.small_area,
               big_area=args.big_area, elong=args.elong, thin_side=args.thin_side,
               dark_v=args.dark_v, bright_v=args.bright_v, min_cc=args.min_cc,
               max_comps_keep=args.max_comps_keep)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.pred = resolve_mask_dir(args.pred)
    if args.pred2:
        args.pred2 = resolve_mask_dir(args.pred2)
    image_dir = args.image or os.path.join(base, "WUH_sub25", "test", "image")
    label_dir = args.label or os.path.join(base, "WUH_sub25", "test", "label")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    cases_dir = os.path.join(out_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(label_dir))
             if f.lower().endswith((".tif", ".tiff", ".png"))]
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f"[输入] 预测 {args.pred}")
    print(f"[输入] 影像 {image_dir} / 标签 {label_dir}")
    print(f"[输入] 共 {len(stems)} 张，band={cfg['band']} near={cfg['near']}")

    # 全局混淆矩阵（两模型各一份）
    def empty_cm():
        return dict(tp=0, fp=0, fn=0, tn=0)

    cm1, cm2 = empty_cm(), (empty_cm() if args.pred2 else None)
    rows = []
    all_comps = []
    type_px = Counter()
    type_cc = Counter()
    fn_px_1 = {}
    fn_px_2 = {}
    skipped = []

    for i, stem in enumerate(stems, 1):
        g_p = os.path.join(label_dir, stem + ".tif")
        if not os.path.exists(g_p):
            alt = [f for f in os.listdir(label_dir) if os.path.splitext(f)[0] == stem]
            if not alt:
                skipped.append(stem)
                continue
            g_p = os.path.join(label_dir, alt[0])
        i_p = None
        for suf in (".tif", ".tiff", ".png"):
            cand = os.path.join(image_dir, stem + suf)
            if os.path.exists(cand):
                i_p = cand
                break
        m_p = find_mask(args.pred, stem)
        if i_p is None or m_p is None:
            skipped.append(stem)
            continue

        gt = read_mask(g_p)
        pred = read_mask(m_p)
        rgb = read_rgb(i_p)
        row, comps, tpx, tcc = analyze_one(stem, pred, gt, rgb, cfg)
        row["has_pred2"] = bool(args.pred2)
        if args.pred2:
            m2 = find_mask(args.pred2, stem)
            if m2:
                pred2 = read_mask(m2)
                inter2 = int(np.logical_and(gt, pred2).sum())
                row["iou2"] = inter2 / (int(gt.sum()) + int(pred2.sum()) - inter2) if (int(gt.sum()) + int(pred2.sum()) - inter2) else 1.0
                fn2 = np.logical_and(gt == 1, pred2 == 0)
                fn_px_2[stem] = int(fn2.sum())
                cm2["tp"] += inter2
                cm2["fp"] += int(np.logical_and(pred2 == 1, gt == 0).sum())
                cm2["fn"] += int(fn2.sum())
                cm2["tn"] += int(gt.size) - inter2 - int(np.logical_and(pred2 == 1, gt == 0).sum()) - int(fn2.sum())
            else:
                row["iou2"] = None
        rows.append(row)
        all_comps.extend(comps)
        type_px.update(tpx)
        type_cc.update(tcc)
        fn_px_1[stem] = row["fn_px"]
        cm1["tp"] += int(np.logical_and(gt, pred).sum())
        cm1["fp"] += row["fp_px"]
        cm1["fn"] += row["fn_px"]
        cm1["tn"] += int(gt.size) - int(np.logical_and(gt, pred).sum()) - row["fp_px"] - row["fn_px"]

        if i % 100 == 0 or i == len(stems):
            print(f"[{i}/{len(stems)}] 已处理，跳过 {len(skipped)}")

    if not rows:
        print("[错误] 没有任何一张图被成功处理，检查 --pred / --label 路径", file=sys.stderr)
        return 1

    # ---------------- 全局指标 ----------------
    def cm_metrics(cm: dict) -> dict:
        tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
        total = tp + fp + fn + tn
        iou_fg = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, pixel_acc=(tp + tn) / total if total else 0.0,
                    miou=(iou_fg + iou_bg) / 2, iou_fg=iou_fg, iou_bg=iou_bg,
                    precision=prec, recall=rec, f1=f1)

    gm1 = cm_metrics(cm1)
    gm2 = cm_metrics(cm2) if cm2 else None

    ious = np.array([r["iou"] for r in rows], dtype=np.float64)
    bf1s = np.array([r["bf1"] for r in rows], dtype=np.float64)
    non_empty = [r for r in rows if not r["is_empty"]]
    empty = [r for r in rows if r["is_empty"]]

    stats = dict(
        n_images=len(rows),
        n_skipped=len(skipped),
        config=cfg,
        pred_dir=os.path.abspath(args.pred),
        pred2_dir=os.path.abspath(args.pred2) if args.pred2 else None,
        global_metrics=gm1,
        global_metrics2=gm2,
        iou=dict(mean=float(ious.mean()), median=float(np.median(ious)),
                 p10=float(np.percentile(ious, 10)), p90=float(np.percentile(ious, 90)),
                 n_zero=int((ious == 0).sum()), n_above_09=int((ious > 0.9).sum())),
        bf1=dict(mean=float(bf1s.mean()), median=float(np.median(bf1s))),
        empty_images=dict(n=len(empty),
                          false_alarm=sum(1 for r in empty if r["empty_false_alarm"] > 0),
                          false_alarm_px=sum(r["empty_false_alarm"] for r in empty)),
        type_px=dict(type_px), type_cc=dict(type_cc),
    )

    # 域差异：按影像平均亮度分 4 档看 IoU
    if non_empty:
        vm = np.array([r["v_mean"] for r in non_empty])
        qs = np.percentile(vm, [25, 50, 75])
        bins = OrderedDict([("暗(<=P25)", []), ("偏暗(P25-P50)", []), ("偏亮(P50-P75)", []), ("亮(>P75)", [])])
        for r in non_empty:
            k = "暗(<=P25)" if r["v_mean"] <= qs[0] else ("偏暗(P25-P50)" if r["v_mean"] <= qs[1]
                 else ("偏亮(P50-P75)" if r["v_mean"] <= qs[2] else "亮(>P75)"))
            bins[k].append(r["iou"])
        stats["domain_by_brightness"] = {k: dict(n=len(v), iou_mean=float(np.mean(v))) for k, v in bins.items() if v}
        # 按建筑密度分档
        db = OrderedDict([("空(0)", []), ("低(0-2%)", []), ("中(2-8%)", []), ("较高(8-20%)", []), ("高(>20%)", [])])
        for r in rows:
            ratio = r["gt_fg"] / (512 * 512)
            k = ("空(0)" if r["gt_fg"] == 0 else "低(0-2%)" if ratio <= 0.02
                 else "中(2-8%)" if ratio <= 0.08 else "较高(8-20%)" if ratio <= 0.20 else "高(>20%)")
            db[k].append(r["iou"])
        stats["by_density"] = {k: dict(n=len(v), iou_mean=float(np.mean(v))) for k, v in db.items() if v}

    # 两模型互补性
    if args.pred2 and fn_px_1 and fn_px_2:
        common = sum(min(fn_px_1.get(s, 0), fn_px_2.get(s, 0)) for s in fn_px_1)
        union_fn = sum(max(fn_px_1.get(s, 0), fn_px_2.get(s, 0)) for s in fn_px_1)
        stats["cross_model"] = dict(
            common_fn_px=common, union_fn_px=union_fn,
            common_ratio=common / union_fn if union_fn else 0.0,
            iou_mean_1=float(np.mean([r["iou"] for r in rows])),
            iou_mean_2=float(np.mean([r["iou2"] for r in rows if r.get("iou2") is not None])),
            n_both_gt_found=sum(1 for r in rows if r["iou"] > 0 or (r.get("iou2") or 0) > 0),
        )

    # ---------------- 写文件 ----------------
    with open(os.path.join(out_dir, "error_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    cols = ["stem", "gt_fg", "pred_fg", "iou", "prec", "rec", "f1", "bf1",
            "n_gt_cc", "n_pred_cc", "fn_px", "fp_px", "fn_bnd", "fn_obj",
            "fp_band", "fp_near", "fp_iso", "v_mean", "is_empty", "empty_false_alarm"]
    if args.pred2:
        cols.append("iou2")
    with open(os.path.join(out_dir, "error_images.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if all_comps:
        ccols = ["stem", "kind", "type", "area", "w", "h", "elong", "min_side",
                 "solidity", "v_med", "cx", "cy", "x0", "y0", "x1", "y1"]
        with open(os.path.join(out_dir, "error_components.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ccols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_comps)

    # ---------------- 典型案例 ----------------
    by_type = {}
    for c in all_comps:
        by_type.setdefault(c["type"], []).append(c)
    picked, used_stems = [], set()
    # 轮转挑：每类轮流取 1 个，保证类型覆盖均衡（而不是前几类吃光名额）
    pool = {t: sorted(by_type.get(t, []), key=lambda c: -c["area"]) for t in TYPE_ORDER if by_type.get(t)}
    while len(picked) < args.max_cases and pool:
        progressed = False
        for t in list(pool):
            if len(picked) >= args.max_cases:
                break
            lst = pool[t]
            while lst:
                c = lst.pop(0)
                if c["stem"] in used_stems:
                    continue
                picked.append((t, c))
                used_stems.add(c["stem"])
                progressed = True
                break
            if not lst:
                pool.pop(t, None)
        if not progressed:
            break
    # 补充：IoU=0 的"整张失败"样本（真实建筑全部漏掉那种），单独补位
    zero_cases = [r for r in sorted(rows, key=lambda r: r["iou"]) if r["iou"] == 0.0]
    for r in zero_cases:
        if len(picked) >= args.max_cases:
            break
        if r["stem"] in used_stems:
            continue
        picked.append(("零重叠(IoU=0)", dict(stem=r["stem"], kind="zero", type="零重叠(IoU=0)",
                                            area=r["fn_px"], elong=0.0, solidity=0.0,
                                            v_med=r["v_med"], x0=0, y0=0, x1=511, y1=511,
                                            cx=256.0, cy=256.0, w=512.0, h=512.0,
                                            min_side=512.0)))
        used_stems.add(r["stem"])

    row_by_stem = {r["stem"]: r for r in rows}
    case_meta = []
    for t, c in picked:
        stem = c["stem"]
        r = row_by_stem[stem]
        gt = read_mask(os.path.join(label_dir, stem + ".tif"))
        pred = read_mask(find_mask(args.pred, stem))
        rgb = read_rgb(os.path.join(image_dir, stem + ".tif"))
        comps_here = [x for x in all_comps if x["stem"] == stem]
        sub = (f"主因: {t} | 该连通域 area={c['area']}px 长宽比={c['elong']:.1f} "
               f"实心度={c['solidity']:.2f} 亮度中位={c['v_med']:.0f} | FN={r['fn_px']} FP={r['fp_px']}")
        fn_out = os.path.join(cases_dir, f"{CN2ASCII.get(t, 'case')}_{stem}.png")
        make_figure(fn_out, stem, rgb, gt, pred, r, comps_here, cfg, sub)
        case_meta.append(dict(stem=stem, type=t, file=os.path.relpath(fn_out, out_dir),
                              area=c["area"], elong=round(c["elong"], 2),
                              solidity=round(c["solidity"], 3), v_med=round(c["v_med"], 1),
                              gt_fg=r["gt_fg"], fn_px=r["fn_px"], fp_px=r["fp_px"],
                              iou=round(r["iou"], 4)))
    stats["cases"] = case_meta
    with open(os.path.join(out_dir, "error_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---------------- summary.md ----------------
    total_px = sum(type_px.values()) or 1
    md = []
    md.append("# 错误分析自动汇总\n")
    md.append(f"- 预测来源：`{stats['pred_dir']}`")
    if args.pred2:
        md.append(f"- 对照来源：`{stats['pred2_dir']}`")
    md.append(f"- 样本：{len(rows)} 张（跳过 {len(skipped)}）")
    md.append(f"- 阈值：band={cfg['band']}px near={cfg['near']}px small={cfg['small_area']}px "
              f"big={cfg['big_area']}px elong={cfg['elong']} thin={cfg['thin_side']}px "
              f"dark_v={cfg['dark_v']} min_cc={cfg['min_cc']}px\n")

    md.append("## 1 全局指标（test 全量像素级混淆矩阵）\n")
    rr = [[k, f"{gm1[k]:.4f}" if isinstance(gm1[k], float) else gm1[k]] for k in
          ("pixel_acc", "iou_fg", "iou_bg", "miou", "precision", "recall", "f1", "tp", "fp", "fn", "tn")]
    md.append(fmt_table(["指标（主模型）", "值"], rr))
    if gm2:
        rr2 = [[k, f"{gm2[k]:.4f}" if isinstance(gm2[k], float) else gm2[k]] for k in
               ("pixel_acc", "iou_fg", "iou_bg", "miou", "precision", "recall", "f1")]
        md.append("\n" + fmt_table(["指标（对照模型）", "值"], rr2))
    md.append("")

    md.append("## 2 逐图分布\n")
    md.append(fmt_table(["项", "值"], [
        ["IoU 均值", f"{stats['iou']['mean']:.4f}"],
        ["IoU 中位", f"{stats['iou']['median']:.4f}"],
        ["IoU P10", f"{stats['iou']['p10']:.4f}"],
        ["IoU P90", f"{stats['iou']['p90']:.4f}"],
        ["IoU=0 的图", stats["iou"]["n_zero"]],
        ["IoU>0.9 的图", stats["iou"]["n_above_09"]],
        ["BF1 均值", f"{stats['bf1']['mean']:.4f}"],
        ["空白图张数", stats["empty_images"]["n"]],
        ["空白图误报张数", stats["empty_images"]["false_alarm"]],
        ["空白图误报总像素", stats["empty_images"]["false_alarm_px"]],
    ]))
    md.append("")

    md.append("## 3 误差类型分解（像素口径）\n")
    rows3 = []
    for t in TYPE_ORDER:
        px = type_px.get(t, 0)
        if px == 0:
            continue
        rows3.append([t, px, pct(px / total_px), type_cc.get(t, 0)])
    md.append(fmt_table(["类型", "像素数", "占全部误差", "连通域数"], rows3))
    md.append("")

    if "domain_by_brightness" in stats:
        md.append("## 4 域差异：按影像亮度分档\n")
        md.append(fmt_table(["亮度档", "张数", "IoU 均值"],
                            [[k, v["n"], f"{v['iou_mean']:.4f}"] for k, v in stats["domain_by_brightness"].items()]))
        md.append("")
    if "by_density" in stats:
        md.append("## 5 按建筑密度分档\n")
        md.append(fmt_table(["密度档", "张数", "IoU 均值"],
                            [[k, v["n"], f"{v['iou_mean']:.4f}"] for k, v in stats["by_density"].items()]))
        md.append("")
    if "cross_model" in stats:
        cmx = stats["cross_model"]
        md.append("## 6 两模型互补性\n")
        md.append(fmt_table(["项", "值"], [
            ["共同漏检像素", cmx["common_fn_px"]],
            ["任一模型漏检像素（并集）", cmx["union_fn_px"]],
            ["共同漏检占并集", pct(cmx["common_ratio"])],
            ["模型1 逐图 IoU 均值", f"{cmx['iou_mean_1']:.4f}"],
            ["模型2 逐图 IoU 均值", f"{cmx['iou_mean_2']:.4f}"],
        ]))
        md.append("")

    md.append("## 7 典型案例\n")
    md.append(fmt_table(["类型", "样本", "IoU", "连通域面积", "长宽比", "亮度中位"],
                        [[c["type"], c["stem"], f"{c['iou']:.4f}", c["area"], f"{c['elong']:.1f}", c["v_med"]]
                         for c in case_meta]))
    md.append(f"\n案例图在 `cases/`，共 {len(case_meta)} 张。")

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"\n[完成] 输出 {out_dir}")
    print(f"[完成] IoU(fg) 全局 {gm1['iou_fg']:.4f} | 逐图均值 {stats['iou']['mean']:.4f} | BF1 {stats['bf1']['mean']:.4f}")
    print(f"[完成] 误差类型 {len([t for t in TYPE_ORDER if type_px.get(t, 0) > 0])} 类，案例图 {len(case_meta)} 张")
    return 0


# ---------------------------------------------------------------------------
# 手动配置区（仿 predict.py 风格：改这里，或命令行覆盖，命令行优先）
# 用法示例：MANUAL = {"pred": r"..\_tools\w5\pred_unet", "limit": 0}
# ---------------------------------------------------------------------------
MANUAL = {}


if __name__ == "__main__":
    if MANUAL:
        argv = sys.argv[1:]
        for k, v in MANUAL.items():
            key = "--" + k.replace("_", "-")
            if key not in argv:
                argv.extend([key, str(v)])
        sys.argv = [sys.argv[0]] + argv
    raise SystemExit(main())

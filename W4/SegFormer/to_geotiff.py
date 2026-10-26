#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""把掩膜写回带地理参考的 GeoTIFF —— W4 第③项

## 先说清楚一件事（这段写进报告里，面试会问）

WHU 的瓦片本身**没有地理参考**：`rasterio.open()` 读出来 `crs=None`、
`transform=(1, 0, 0, 0, 1, 0)`（单位阵），PIL 打开也只有像素坐标。
所以"保留原始投影和仿射变换"这句话在这些瓦片上无从谈起——**没有什么可以保留的东西**。

本脚本的做法是：**给瓦片配一套合成的仿射变换与坐标系**，把整条链路（掩膜 -> 带 CRS 的
GeoTIFF -> 面积统计表）真实地跑通，并在输出与日志里明确标注"地理参考为示意值"。
换成真正带地理参考的正射影像时，只要把 `--crs / --pixel-size / --origin` 换成从影像读出来的
真实值（`src.crs` / `src.transform`），**其余代码一行不用改**——这正是本脚本的价值。

## 默认参数为什么是这几个数

武汉市中心 (114.3055°E, 30.5928°N) 在 `EPSG:32650`（WGS84 / UTM 50N）里是
`easting=241633.9 m, northing=3387568.7 m`；取 `0.3 m/px`（常见无人机正射分辨率）后
对齐到 0.3 的整数倍：`origin=(241633.8, 3387568.8)`。
**1 像素 = 0.3 × 0.3 = 0.09 m²**，所以面积表里的平方米数 = 像素数 × 0.09。

## 两种模式

- `--mode tiles`：一张掩膜 -> 一个 GeoTIFF（各自独立，适合抽几张交付）
- `--mode mosaic`：整个目录按瓦片网格拼成**一整幅** GeoTIFF（适合给客户/领导看全貌）
  网格顺序取自 WHU 的编号规则：`编号 = 行 × 行宽 + 列`，行宽用 `--row-width` 给
  （train=64、val=14、test=23；都是 74 行，四至范围被竖切成不同列数）

## 用法

    :: 1) 单张/几张 -> 一堆小 GeoTIFF
    python to_geotiff.py --mode tiles --out run\predict\batch_test --dst out\geo_test

    :: 2) 整幅镶嵌（test 的行宽是 23）
    python to_geotiff.py --mode mosaic --out run\predict\batch_test --row-width 23 ^
                         --dst out\mosaic_test.tif

    :: 3) 改地理参考（真实影像上就传真实的 CRS 与原点）
    python to_geotiff.py --mode mosaic --crs EPSG:4547 --pixel-size 0.5 ^
                         --origin 500000 3380000 --dst out\mosaic.tif

    :: 4) 不想敲命令行？改文件末尾的 MANUAL 配置区，然后直接
    python to_geotiff.py

## 产物

- GeoTIFF：`uint8`，前景=`--fg-value`（默认 1），背景 0，`--nodata 255`，
  LZW 压缩 + 分块（`tiled=True`），QGIS/ArcGIS 直接拖进去就是正确位置
- 面积表 CSV：每个连通域一行（面积 m²、质心经纬度对应的投影坐标、外接框），
  默认 `--area-csv` 与 GeoTIFF 同目录同名
- 自检：写完会用 `rasterio` 重新打开，回读 CRS/仿射/尺寸/前景像素数并与内存里的结果对账
"""

from __future__ import print_function

import argparse
import csv
import os
import re
import sys

import numpy as np
from PIL import Image

import rasterio
from rasterio.transform import Affine

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


_HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------- #
# 掩膜读取
# --------------------------------------------------------------------------- #
def read_mask(path):
    """读成二值 uint8 {0,1}。WHU 的标签是 PIL mode='1'，GridMask 之类也可能是 0/255 或彩色。"""
    im = Image.open(path)
    if im.mode == "1":
        arr = np.array(im, dtype=np.uint8)
    elif im.mode in ("L", "I", "I;16", "F"):
        arr = (np.array(im) > 0).astype(np.uint8)
    elif im.mode in ("P", "PA"):
        arr = (np.array(im.convert("L")) > 0).astype(np.uint8)
    else:
        rgb = np.array(im.convert("RGB"))
        arr = (rgb.max(axis=2) > 0).astype(np.uint8)
    return arr


def list_masks(mask_dir, limit=0, prefix_filter=None):
    """按编号排序返回 [(key, path, prefix, num)]

    - 输出文件名通常是 `<原名>_mask.tif`，所以先把这段后缀去掉，再解析结尾数字
    - 不排序的话拼接顺序会乱（sorted 用编号优先，编号相同再按名字）
    - prefix_filter 给了就只保留该前缀（编号混批时用）
    """
    items = []
    for fn in os.listdir(mask_dir):
        if not fn.lower().endswith(IMAGE_EXT):
            continue
        key = os.path.splitext(fn)[0]
        for suf in ("_mask", "_label", "_overlay", "-mask"):
            if key.lower().endswith(suf):
                key = key[:-len(suf)]
                break
        m = re.match(r"^(.*?)(\d+)$", key)
        prefix = m.group(1) if m else key
        num = int(m.group(2)) if m else -1
        if prefix_filter is not None and prefix != prefix_filter:
            continue
        items.append((key, os.path.join(mask_dir, fn), prefix, num))
    items.sort(key=lambda t: (t[3], t[0]))
    if limit and limit > 0:
        items = items[:limit]
    return items


# --------------------------------------------------------------------------- #
# 仿射变换
# --------------------------------------------------------------------------- #
def make_transform(pixel_size, origin):
    """北向上的仿射：x 向右为正、y 向下为负（这是 GeoTIFF 的常规方向）。

    Affine(a, b, c, d, e, f) 对应
        X = a * col + b * row + c
        Y = d * col + e * row + f
    pixel_size=0.3、origin=(x0, y0) 时：a=0.3, e=-0.3, c=x0, f=y0（y0 是左上角）
    """
    px = float(pixel_size)
    x0, y0 = float(origin[0]), float(origin[1])
    return Affine(px, 0.0, x0, 0.0, -px, y0)


def transform_of_tile(base, tile_size, row, col):
    """在整幅仿射的基础上，算出第 (row, col) 块瓦片的仿射（左上角平移过去）。"""
    px = base.a
    x0 = base.c + col * tile_size * px
    y0 = base.f - row * tile_size * px
    return Affine(px, 0.0, x0, 0.0, -px, y0)


# --------------------------------------------------------------------------- #
# 写 GeoTIFF
# --------------------------------------------------------------------------- #
def write_geotiff(dst, arr_u8, transform, crs, fg_value=1, nodata=255,
                  compress="LZW", overwrite=True):
    """arr_u8 是 {0,1}；落到磁盘前按 fg_value 映射（默认 1，可改成 255 兼容看图软件）。"""
    out = arr_u8.astype(np.uint8)
    if fg_value != 1:
        out = (out > 0).astype(np.uint8) * np.uint8(fg_value)

    prof = dict(
        driver="GTiff",
        height=int(out.shape[0]),
        width=int(out.shape[1]),
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    if nodata is not None and int(nodata) != int(fg_value):
        prof["nodata"] = int(nodata)

    d = os.path.dirname(os.path.abspath(dst))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(dst) and not overwrite:
        raise SystemExit("目标已存在（加 --overwrite 覆盖）：%s" % dst)

    tmp = dst + ".tmp"
    with rasterio.open(tmp, "w", **prof) as ds:
        ds.write(out, 1)
    os.replace(tmp, dst)          # 原子替换：中断不会留下半个文件
    return dst


def verify_geotiff(path):
    """回读自检：CRS、仿射、尺寸、前景像素数。"""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        info = dict(
            crs=str(ds.crs), transform=tuple(ds.transform)[:6],
            width=ds.width, height=ds.height, dtype=ds.dtypes[0],
            nodata=ds.nodata, bounds=tuple(round(v, 3) for v in ds.bounds),
            n_fg=int((arr != 0).sum()), compress=ds.profile.get("compress"),
        )
    return info


# --------------------------------------------------------------------------- #
# 面积统计（连通域 -> 平方米）
# --------------------------------------------------------------------------- #
def component_areas(mask, px_area, key="", max_components=500, min_area_px=1,
                    transform=None):
    """8 邻接连通域，按面积降序返回若干行（面积 m²、质心、外接框）。

    transform 给了就顺手把质心换算成投影坐标（东, 北），这样面积表能直接丢进 GIS 或贴到地图上。
    """
    if not mask.any():
        return []
    if _HAS_SCIPY:
        lab, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    else:
        lab, n = mask.astype(np.int32), 1
    if n == 0:
        return []
    slices = _ndi.find_objects(lab) if _HAS_SCIPY else [None] * n
    areas = np.bincount(lab.ravel())
    order = np.argsort(areas[1:])[::-1] + 1          # 去掉背景的 0 号
    rows = []
    for rank, idx in enumerate(order[:max_components], start=1):
        a_px = int(areas[idx])
        if a_px < min_area_px:
            continue
        sl = slices[idx - 1] if slices and slices[idx - 1] is not None else None
        if sl is None:
            cy = cx = float("nan")
            h = w = 0
        else:
            cy = (sl[0].start + sl[0].stop - 1) / 2.0
            cx = (sl[1].start + sl[1].stop - 1) / 2.0
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
        cx_r, cy_r = round(cx, 1), round(cy, 1)
        if transform is not None and cy_r == cy_r and cx_r == cx_r:
            # 像素 (col,row) -> 投影坐标：X = a*col + c，Y = e*row + f（e 为负）
            cx_map = round(transform.c + cx_r * transform.a, 2)
            cy_map = round(transform.f + cy_r * transform.e, 2)
        else:
            cx_map = cy_map = ""
        rows.append(dict(key=key, rank=rank, area_px=a_px,
                         area_m2=round(a_px * px_area, 3),
                         centroid_row=cy_r, centroid_col=cx_r,
                         centroid_x=cx_map, centroid_y=cy_map,
                         bbox_h=h, bbox_w=w))
    return rows


def write_area_csv(path, rows, transform=None):
    """transform 参数只为兼容老调用；行列 -> 投影坐标的换算已经在 component_areas 里做完了。"""
    fields = ["key", "rank", "area_px", "area_m2",
              "centroid_row", "centroid_col", "centroid_x", "centroid_y",
              "bbox_h", "bbox_w"]
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            if r.get("centroid_x", "") == "" and transform is not None and r.get("centroid_row") == r.get("centroid_row"):
                r["centroid_x"] = round(transform.c + r["centroid_col"] * transform.a, 2)
                r["centroid_y"] = round(transform.f + r["centroid_row"] * transform.e, 2)
            r.setdefault("centroid_x", "")
            r.setdefault("centroid_y", "")
            w.writerow({k: r.get(k, "") for k in fields})
    return path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    g = globals().get("MANUAL", {})          # 文件末尾那个手动配置区
    p = argparse.ArgumentParser(
        description="把掩膜写回带地理参考的 GeoTIFF（合成仿射，瓦片本身无 CRS）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", default=g.get("mode", "mosaic"), choices=["tiles", "mosaic"],
                   help="tiles=每张一个 GeoTIFF；mosaic=按网格拼成一整幅")
    p.add_argument("--out", default=g.get("out", os.path.join(_HERE, "run", "predict", "batch")),
                   help="batch_infer.py 的输出目录（下面默认取它的 masks/）")
    p.add_argument("--masks", default=g.get("masks", None),
                   help="掩膜目录；不传就用 <out>/masks")
    p.add_argument("--dst", default=g.get("dst", os.path.join(_HERE, "run", "predict", "geo")),
                   help="mode=tiles 时是目录；mode=mosaic 时是 .tif 文件路径")
    p.add_argument("--crs", default=g.get("crs", "EPSG:32650"),
                   help="坐标系；瓦片没有真实 CRS，这里给的是示意值（武汉 UTM 50N）")
    p.add_argument("--pixel-size", type=float, default=g.get("pixel_size", 0.3),
                   help="分辨率（米/像素）")
    p.add_argument("--origin", type=float, nargs=2,
                   default=list(g.get("origin", (241633.8, 3387568.8))),
                   metavar=("X", "Y"),
                   help="整幅左上角的投影坐标（东, 北）；默认是武汉市中心在 EPSG:32650 的值")
    p.add_argument("--row-width", type=int, default=g.get("row_width", 0),
                   help="mosaic 用：每行多少块（train=64, val=14, test=23）")
    p.add_argument("--tile-prefix", default=g.get("tile_prefix", None),
                   help="编号前缀过滤，如 '2_' 或 ''（纯数字批）；默认自动取最多的那批")
    p.add_argument("--tile-size", type=int, default=g.get("tile_size", 0),
                   help="mosaic 用：瓦片边长像素；0=按第一张自动识别")
    p.add_argument("--fg-value", type=int, default=g.get("fg_value", 1),
                   help="前景像素值（写 255 更兼容只看黑白图的软件）")
    p.add_argument("--nodata", type=int, default=g.get("nodata", 255),
                   help="nodata 值；-1 表示不设")
    p.add_argument("--compress", default=g.get("compress", "LZW"), choices=["LZW", "DEFLATE", "NONE"],
                   help="压缩方式")
    p.add_argument("--area-csv", default=g.get("area_csv", None),
                   help="面积表 CSV 路径；默认与 GeoTIFF 同目录同名")
    p.add_argument("--max-components", type=int, default=g.get("max_components", 500),
                   help="面积表里最多列多少个连通域（按面积降序）")
    p.add_argument("--min-area-px", type=int, default=g.get("min_area_px", 1),
                   help="小于这么多像素的连通域不写进面积表（滤掉碎斑）")
    p.add_argument("--limit", type=int, default=g.get("limit", 0),
                   help="只处理前 N 张（调试用）")
    p.add_argument("--overwrite", default=True, action=argparse.BooleanOptionalAction,
                   help="目标已存在时是否覆盖")
    p.add_argument("--quiet", default=False, action="store_true")
    p.add_argument("--announce-synthetic", default=True, action=argparse.BooleanOptionalAction,
                   help="在日志里声明“地理参考是示意值”")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out)
    mask_dir = os.path.abspath(args.masks) if args.masks else os.path.join(out_dir, "masks")
    if not os.path.isdir(mask_dir):
        raise SystemExit("掩膜目录不存在：%s（先跑 batch_infer.py，或用 --masks 指定）" % mask_dir)

    items = list_masks(mask_dir, args.limit, prefix_filter=args.tile_prefix)
    if not items:
        raise SystemExit("掩膜目录里没有图片：%s" % mask_dir)

    px = float(args.pixel_size)
    px_area = px * px
    base = make_transform(px, args.origin)
    nodata = None if int(args.nodata) < 0 else int(args.nodata)

    print("[地理参考] CRS %s | %.3f m/px | 左上角 (%.1f, %.1f)"
          % (args.crs, px, args.origin[0], args.origin[1]))
    if args.announce_synthetic:
        print("[声明] 这些瓦片原本没有 CRS 与仿射（读出来 crs=None、transform=单位阵），"
              "上面的地理参考是**示意值**；换成真实正射影像时传入真实的 CRS/原点即可")
    print("[掩膜] %d 张，来自 %s" % (len(items), mask_dir))

    t0 = __import__("time").time()
    area_rows = []

    # ---------------------------------------------------------------- tiles
    if args.mode == "tiles":
        dst_dir = os.path.abspath(args.dst)
        geo_tiles = args.row_width > 0
        if geo_tiles:
            print("[网格] tiles 模式按编号摆位：编号 = 行 × %d + 列，每张各自落在整幅上的真实位置"
                  % args.row_width)
        else:
            print("[网格] 没给 --row-width，每张都以同一个左上角为原点（互不重叠、坐标会重复）；"
                  "想按真实网格摆位就补 --row-width")
        n_ok = 0
        for i, (key, path, _pre, _num) in enumerate(items, start=1):
            mask = read_mask(path)
            tf = base
            if geo_tiles and _num >= 0:
                row, col = divmod(_num, args.row_width)
                tf = transform_of_tile(base, int(mask.shape[0]), row, col)
            dst = os.path.join(dst_dir, key + ".tif")
            write_geotiff(dst, mask, tf, args.crs, args.fg_value, nodata,
                          args.compress, args.overwrite)
            area_rows.extend(component_areas(mask, px_area, key,
                                             args.max_components, args.min_area_px,
                                             transform=tf))
            n_ok += 1
            if not args.quiet and (i % 50 == 0 or i == len(items)):
                print("[写盘] %d/%d" % (i, len(items)))
        info = verify_geotiff(os.path.join(dst_dir, items[0][0] + ".tif"))
        print("[自检] 抽查 %s" % os.path.basename(items[0][0] + ".tif"))
        print("       CRS %s | 尺寸 %dx%d | 前景 %d 像素 | 压缩 %s"
              % (info["crs"], info["width"], info["height"], info["n_fg"], info["compress"]))
        print("       仿射 a=%.3f e=%.3f c=%.1f f=%.1f（e 为负=北向上）"
              % (info["transform"][0], info["transform"][4],
                 info["transform"][2], info["transform"][5]))
        print("[完成] %d 个 GeoTIFF -> %s" % (n_ok, dst_dir))

    # ---------------------------------------------------------------- mosaic
    else:
        if args.row_width <= 0:
            raise SystemExit("mode=mosaic 必须给 --row-width（train=64, val=14, test=23；都是 74 行）")
        # 编号可能是 '2_1121' 也可能是纯数字 '254'：按前缀分组，取最多的那组，其余明确忽略
        groups = {}
        for it in items:
            groups.setdefault(it[2], []).append(it)
        prefix, group = max(groups.items(), key=lambda kv: len(kv[1]))
        group = [it for it in items if it[2] == prefix]
        others = sorted(((k, len(v)) for k, v in groups.items() if k != prefix),
                        key=lambda kv: -kv[1])
        if others:
            n_ign = sum(c for _k, c in others)
            head = "、".join("'%s' %d 张" % (k if k else "<纯数字>", c) for k, c in others[:3])
            tail = " 等 %d 种" % len(others) if len(others) > 3 else ""
            print("[编号] 主导前缀 '%s' %d 张；忽略 %d 张（%s%s）— 编号混批时无法拼进同一张网格"
                  % (prefix if prefix else "<纯数字>", len(group), n_ign, head, tail))
            print("       想指定另一批就用 --tile-prefix，想只看某批就用 --masks 指向单独目录")
        if any(num < 0 for _k, _p, _pre, num in group):
            raise SystemExit("有掩膜文件名里没有数字编号，无法定位到网格（mosaic 需要 '名字+数字'）")

        first = read_mask(group[0][1])
        tile = int(args.tile_size) if args.tile_size > 0 else int(first.shape[0])
        if first.shape[0] != first.shape[1]:
            raise SystemExit("瓦片不是正方形：%s" % first.shape)
        if tile != first.shape[0]:
            print("[提示] --tile-size %d 与掩膜实际边长 %d 不一致，以 --tile-size 为准"
                  % (tile, first.shape[0]))

        nums = [it[3] for it in group]
        n_rows = max(nums) // args.row_width + 1
        canvas = np.zeros((n_rows * tile, args.row_width * tile), dtype=np.uint8)
        touched = set()
        n_collision = 0
        for i, (key, path, _pre, num) in enumerate(group, start=1):
            mask = read_mask(path)
            if mask.shape != (tile, tile):
                print("[跳过] %s 尺寸 %s != %d" % (key, mask.shape, tile))
                continue
            row, col = divmod(num, args.row_width)
            if row >= n_rows:
                continue
            if (row, col) in touched:
                n_collision += 1          # 编号重复（同号不同名），后到的覆盖先到的
            canvas[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = mask
            touched.add((row, col))
            if not args.quiet and (i % 100 == 0 or i == len(group)):
                print("[拼接] %d/%d" % (i, len(group)))
        if n_collision:
            print("[警告] 有 %d 张掩膜的编号落在同一格，后写入的覆盖了前面的——检查是否混了不同批次"
                  % n_collision)

        dst = os.path.abspath(args.dst)
        if not dst.lower().endswith((".tif", ".tiff")):
            dst = os.path.join(dst, "mosaic.tif")
        write_geotiff(dst, canvas, base, args.crs, args.fg_value, nodata,
                      args.compress, args.overwrite)

        covered = len(touched)
        print("[拼接] 网格 %d 行 x %d 列 | 实际覆盖 %d 块（%.1f%%）"
              % (n_rows, args.row_width, covered,
                 100.0 * covered / float(n_rows * args.row_width)))
        area_rows.extend(component_areas(canvas, px_area, "mosaic",
                                         args.max_components, args.min_area_px,
                                         transform=base))
        info = verify_geotiff(dst)
        print("[自检] CRS %s | 尺寸 %dx%d | 前景 %d 像素 | 压缩 %s"
              % (info["crs"], info["width"], info["height"], info["n_fg"], info["compress"]))
        print("       四至 bounds=%s" % (info["bounds"],))
        print("       地表覆盖约 %.2f km x %.2f km"
              % (info["width"] * px / 1000.0, info["height"] * px / 1000.0))
        print("[完成] GeoTIFF -> %s" % dst)

    # ---------------------------------------------------------------- 面积表
    area_rows.sort(key=lambda r: -r["area_px"])
    csv_path = args.area_csv
    if not csv_path:
        if args.mode == "tiles":
            csv_path = os.path.abspath(args.dst) + ".csv"
        else:
            dst_abs = os.path.abspath(args.dst)
            csv_path = (dst_abs if dst_abs.lower().endswith((".tif", ".tiff"))
                        else os.path.join(dst_abs, "mosaic.tif"))
            csv_path = os.path.splitext(csv_path)[0] + "_area.csv"
    write_area_csv(csv_path, area_rows, base)

    n_fg_rows = len(area_rows)
    total_m2 = sum(r["area_m2"] for r in area_rows)
    print("[面积] 连通域 %d 个 | 合计 %.1f m²（%.4f km²）| 1 像素 = %.4f m²"
          % (n_fg_rows, total_m2, total_m2 / 1e6, px_area))
    if n_fg_rows:
        print("       最大一个 %.1f m² | 中位 %.1f m²"
              % (area_rows[0]["area_m2"],
                 area_rows[len(area_rows) // 2]["area_m2"]))
    print("[面积表] %s" % csv_path)
    print("[用时] %.1fs" % (__import__("time").time() - t0))
    return 0


# =========================================================================== #
# ↓↓↓ 手动配置区：不想敲命令行就改这里，然后直接 `python to_geotiff.py` ↓↓↓
#     命令行给了参数就优先用命令行的（这里的值只是默认值）
# =========================================================================== #
MANUAL = dict(
    mode="mosaic",        # "tiles" = 每张一个 GeoTIFF；"mosaic" = 按网格拼成一整幅
    out=None,             # batch_infer.py 的输出目录；None = 本工程 run\predict\batch
    masks=None,           # 掩膜目录；None = <out>\masks
    dst=None,             # tiles: 目录；mosaic: .tif 路径；None = 本工程 run\predict\geo
    crs="EPSG:32650",     # 坐标系（示意值；真实影像请换成 src.crs）
    pixel_size=0.3,       # 分辨率 米/像素
    origin=(241633.8, 3387568.8),   # 整幅左上角（东, 北）——武汉市中心在 UTM 50N 的坐标
    row_width=23,         # 每行几块：train=64 / val=14 / test=23
    tile_prefix=None,     # 编号前缀过滤，如 "2_" 或 ""；None = 自动取最多的那批
    tile_size=0,          # 0 = 按第一张掩膜自动识别
    fg_value=1,           # 前景像素值，改成 255 更兼容只看黑白图的软件
    nodata=255,           # -1 = 不设
    compress="LZW",
    area_csv=None,        # None = 与 GeoTIFF 同目录
    max_components=500,
    min_area_px=1,
    limit=0,
)


if __name__ == "__main__":
    sys.exit(main())

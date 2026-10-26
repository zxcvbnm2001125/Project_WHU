# -*- coding: utf-8 -*-
"""U-Net 单图 / 小目录预测（薄壳版）。

和 batch_infer.py 的分工
------------------------
- 本文件：看几张图对不对。逐张推理，打印每张的耗时和前景占比，输出索引掩膜 + 叠加图。
- batch_infer.py：跑全量、断了能续、留清单，给下游质检 / 写 GeoTIFF 用。

两者共用**同一套预处理和后处理**：本文件的推理全部委托给
`batch_infer.UnetAdapter`（prep / infer / post / overlay / colorize），
所以同一张图在"单图模式"和"批量模式"下得到的结果逐像素一致。

相对最初的 predict.py 修掉的四处口径问题
----------------------------------------
1. 概率图先 `cv2.resize(INTER_LINEAR)` 放大再 argmax（原 predict.py:86-89）——
   类别索引不能线性插值，会插出不存在的类别。现在在网络分辨率上 argmax，
   再用最近邻缩回原图尺寸。
2. `--mix_type` 写成 `default=True, action='store_true'`（原 predict.py:162），
   永远为真、关不掉。现在是 `--mix-type / --no-mix-type`。
3. 只存"原图 + 掩膜"混合图（原 predict.py:92-108），拿去做指标或转 GeoJSON 都不行。
   现在默认存 uint8 索引掩膜（0 背景 / 1 建筑），叠加图另存 `*_overlay.png`。
4. 每次运行都新建 `run/predict/expN`（原 predict.py:113）——单图调试时目录越堆越多，
   也没法重复覆盖同一份结果。现在可以用 `--out` 指定固定目录（不传则保持原行为）。

用法
----
    python predict.py                                    # 默认看 WUH_sub25/val/image 全部
    python predict.py --data_path path/to/img.tif        # 单张
    python predict.py --data_path path/to/imgs --max-images 5   # 只看 5 张
    python predict.py --out run/predict/preview          # 固定输出目录
    python predict.py --no-mix-type                      # 只要索引掩膜，不要叠加图
    python predict.py --input-shape 480                  # 换输入边长（必须是 32 的倍数）
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from batch_infer import (                                             # noqa: E402
    UnetAdapter, find_latest_weights, list_images,
)
from utils.create_exp_folder import create_val_exp_folder              # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                            "WUH_sub25", "val", "image")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="U-Net 预测（单张 / 小目录，薄壳版）")
    p.add_argument("--data_path", "--data-path", dest="data_path", default=DEFAULT_DATA,
                   help="图片或文件夹（递归）；默认 WUH_sub25/val/image")
    p.add_argument("--weights", default=None,
                   help="权重；不传则自动取 run/train/*/weights 下最新的 best_model_*.pth")
    p.add_argument("--num-classes", default=1, type=int,
                   help="前景类数（不含背景）；WHU 建筑二分类填 1")
    p.add_argument("--input-shape", "--input_shape", dest="input_shape", default=512, type=int,
                   help="网络输入边长，必须能被 32 整除（训练用的是 512）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--alpha", default=0.5, type=float, help="叠加图里掩膜的透明度（0~1）")
    p.add_argument("--mix-type", "--mix_type", dest="mix_type", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="是否另外保存原图+掩膜的叠加图（默认保存）")
    p.add_argument("--out", default=None,
                   help="输出目录；不传则按原行为每次新建 run/predict/expN")
    p.add_argument("--max-images", default=0, type=int, help=">0 时只跑前 N 张，用于抽检")
    return p.parse_args(argv)


def collect_images(data_path):
    if os.path.isdir(data_path):
        return [p for p in list_images([data_path])
                if os.path.splitext(p)[1].lower() in IMG_EXTS]
    if os.path.isfile(data_path):
        return [data_path]
    raise SystemExit("输入路径不存在：%s" % data_path)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda 但当前环境没有可用 GPU")

    weights = args.weights or find_latest_weights()
    if not os.path.isfile(weights):
        raise SystemExit("权重不存在：%s" % weights)

    srcs = collect_images(args.data_path)
    if args.max_images and args.max_images > 0:
        srcs = srcs[:args.max_images]
    if not srcs:
        raise SystemExit("没有找到可用的图片：%s" % args.data_path)

    # 建 adapter 就等于建了模型 + 做了权重覆盖率检查（覆盖率不足会直接报错退出）
    adapter = UnetAdapter(weights=os.path.abspath(weights), device=device,
                          input_shape=args.input_shape, num_classes=args.num_classes,
                          alpha=args.alpha)

    out_dir = args.out or create_val_exp_folder()
    os.makedirs(out_dir, exist_ok=True)
    print("[输出] %s（%d 张 | 索引掩膜%s）"
          % (out_dir, len(srcs), " + 叠加图" if args.mix_type else ""))

    fg_values, t0 = [], time.time()
    for i, path in enumerate(srcs, 1):
        t_img = time.time()
        prep = adapter.prep(path)
        mask = adapter.post(adapter.infer(prep["x"][None])[0], prep["meta"])

        stem = os.path.splitext(os.path.basename(path))[0]
        Image.fromarray(mask).save(os.path.join(out_dir, "%s_mask.png" % stem))
        if args.mix_type:
            Image.fromarray(adapter.overlay(mask, path)).save(
                os.path.join(out_dir, "%s_overlay.png" % stem))

        fg = float((mask > 0).mean())
        fg_values.append(fg)
        print("[%3d/%3d] %-24s %dx%d | 前景 %5.2f%% | %6.1f ms"
              % (i, len(srcs), os.path.basename(path), mask.shape[1], mask.shape[0],
                 fg * 100, (time.time() - t_img) * 1000))

    dt = time.time() - t0
    print("[汇总] %d 张 | 总耗时 %.1fs | 平均 %.0f ms/张 | 前景占比均值 %.2f%%"
          % (len(srcs), dt, dt * 1000 / len(srcs), 100 * float(np.mean(fg_values))))
    print("[汇总] 目录 %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

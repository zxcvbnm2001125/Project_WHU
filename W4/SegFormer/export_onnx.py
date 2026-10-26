# -*- coding: utf-8 -*-
"""SegFormer 导出 ONNX + 一致性校验 + 延迟基准。

和 batch_infer.py / to_geotiff.py 的分工
---------------------------------------
- batch_infer.py ：训练好的权重跑批量推理，出掩膜。
- to_geotiff.py  ：掩膜写回带坐标系的大图。
- export_onnx.py ：把模型交出去（ONNX），并证明"交出去的和这里跑的一样"。

导出与校验的引擎在同级 `utils/onnx_export.py`（两个工程共用、逐字节相同），
本文件只做 SegFormer 特有的三件事：
  1. 建模型（规格从权重旁边的 `train_config.txt` 自动读：variant / decoder_dim /
     num_classes / input_shape，复用 batch_infer 的 Adapter，含双向覆盖率校验）
  2. 提供"原图 -> 0~255 CHW 张量"的预处理（与验证期同一套几何：等比缩放 + 灰边填充）
  3. 提供原推理管线（prep -> infer -> post），用来核对导出结果的口径

用法
----
    python export_onnx.py                          # 自动找最新权重，跑默认 val 子集
    python export_onnx.py --limit-verify 20        # 多校几张
    python export_onnx.py --fp16                   # 顺带测 GPU FP16 延迟
    python export_onnx.py --weights run/train/exp1/weights/best_model_2.pth
    python export_onnx.py --no-bake                # 不把 ImageNet 归一化烘进图

产物（默认落在 run/onnx/）
-------------------------
    segformer_<variant>.onnx           导出的模型
    onnx_export_report.md              人看的报告：结论/契约/校验/延迟/限制
    onnx_export_report.json            机器读的同一份数据

注意
----
- SegFormer 训练期做过 ImageNet 标准化（`utils/utils.py:98`），所以导出的图里
  烘的是 `(x/255 - mean) / std`；U-Net 只做 `x/255`，两个模型**不能共用同一段
  部署预处理**，这点在报告里也标了。
- 权重文件名里的类别数是"前景类别数"（`best_model_2.pth` 指 num_classes=1 + 背景），
  实际建模型时内部 +1，别被文件名绕进去。
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# onnx / onnxruntime 被装在非标准目录时（离线机、沙箱），用环境变量指过来：
#   set ONNX_LIBS=D:\path\to\pylibs
_LIBS = os.environ.get("ONNX_LIBS", "")
if _LIBS and os.path.isdir(_LIBS) and _LIBS not in sys.path:
    sys.path.insert(0, _LIBS)

from batch_infer import SegFormerAdapter                             # noqa: E402
from predict import find_latest_weights                              # noqa: E402
from utils.onnx_export import run_export                             # noqa: E402
from utils.utils import cvtColor                                     # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(_HERE))                      # -> 消融实验/
DEFAULT_INPUT = os.path.join(_ROOT, "WUH_sub25", "val", "image")
DEFAULT_GT = os.path.join(_ROOT, "WUH_sub25", "val", "label")
DEFAULT_OUT = os.path.join(_HERE, "run", "onnx")


# --------------------------------------------------------------------------- 预处理
def make_raw_loader(side):
    """原图 -> [3, side, side] float32，像素范围 0~255（**不归一化**）。

    几何处理与验证期完全一致（predict.py:222-246）：
    等比缩放 + 居中灰边 (128,128,128) 填充 + BICUBIC 重采样。
    """

    def load(path):
        image = cvtColor(Image.open(path))
        iw, ih = image.size
        scale = min(side / float(iw), side / float(ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        canvas.paste(image.resize((nw, nh), Image.BICUBIC),
                     ((side - nw) // 2, (side - nh) // 2))
        arr = np.array(canvas, dtype=np.float64)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)

    return load


def make_pipeline(adapter):
    """原推理管线（batch_infer 用的那条路），用来对口径。"""

    def predict(path):
        item = adapter.prep(path)
        small = adapter.infer([item["x"]])[0]
        return adapter.post(small, item["meta"])

    return predict


# --------------------------------------------------------------------------- 入口
def parse_args(argv=None):
    g = globals().get("MANUAL", {})          # 文件末尾那个手动配置区
    p = argparse.ArgumentParser(
        description="SegFormer 导出 ONNX 并校验一致性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default=g.get("weights", None),
                   help="权重路径；不填 = 自动找 run/train/exp*/weights 下最新的 best_model_*.pth")
    p.add_argument("--input", default=g.get("input") or DEFAULT_INPUT,
                   help="校验用图片目录")
    p.add_argument("--gt-dir", default=g.get("gt_dir") or DEFAULT_GT,
                   help="真值掩膜目录")
    p.add_argument("--gt", default=True, action=argparse.BooleanOptionalAction,
                   help="是否用真值算 IoU 对齐（关掉就只做数值一致性）")
    p.add_argument("--out", default=g.get("out") or DEFAULT_OUT,
                   help="产物目录")
    p.add_argument("--input-shape", type=int, default=g.get("input_shape", 0),
                   help="输入边长；0 = 从 train_config.txt 自动读（推荐），否则必须能被 32 整除")
    p.add_argument("--variant", default=g.get("variant", None),
                   help="MiT 规格 b1/b2/b3；不填 = 从 train_config.txt 自动读")
    p.add_argument("--decoder-dim", type=int, default=g.get("decoder_dim", 0),
                   help="解码器宽度；0 = 自动读（本项目训练用 256）")
    p.add_argument("--num-classes", type=int, default=g.get("num_classes", 0),
                   help="前景类别数；0 = 自动读（WHU 建筑二分类为 1）")
    p.add_argument("--opset", type=int, default=g.get("opset", 17),
                   help="ONNX opset 版本")
    p.add_argument("--limit-verify", type=int, default=g.get("limit_verify", 12),
                   help="一致性校验用几张图")
    p.add_argument("--batch", type=int, default=g.get("batch", 8),
                   help="吞吐基准用的 batch")
    p.add_argument("--iters", type=int, default=g.get("iters", 30),
                   help="每个基准跑几次（取中位数）")
    p.add_argument("--dynamic-hw", default=g.get("dynamic_hw", False),
                   action=argparse.BooleanOptionalAction,
                   help="把 H/W 也设成动态维（默认只动态 batch）")
    p.add_argument("--bake", default=g.get("bake", True),
                   action=argparse.BooleanOptionalAction,
                   help="把 ImageNet 归一化烘进计算图（默认开）")
    p.add_argument("--fp16", default=g.get("fp16", True),
                   action=argparse.BooleanOptionalAction,
                   help="顺带测 GPU FP16 延迟（需要 CUDA）")
    p.add_argument("--device", default=g.get("device", "cuda"),
                   help="cuda / cpu")
    p.add_argument("--ort-libs", default=g.get("ort_libs", None),
                   help="额外塞进 sys.path 的目录，用来指定某一份 onnxruntime"
                        "（例如解压出来的 onnxruntime-gpu）")
    p.add_argument("--ort-device", default=g.get("ort_device", "auto"),
                   choices=["auto", "gpu", "cpu"],
                   help="ORT 用哪个 EP；auto = 有 CUDA EP 就用")
    p.add_argument("--ort-fp16", default=g.get("ort_fp16", True),
                   action=argparse.BooleanOptionalAction,
                   help="顺便把图转一份 FP16 在 CUDA EP 上测（需要 GPU 版 ORT）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] 没有可用 GPU，退回 CPU（延迟数字会慢一个量级）")
        device = "cpu"

    weights = args.weights or find_latest_weights()
    print("[权重] %s" % weights)
    adapter = SegFormerAdapter(
        weights=weights, device=torch.device(device),
        variant=args.variant,
        decoder_dim=(args.decoder_dim or None),
        num_classes=(args.num_classes or None),
        input_shape=(args.input_shape or None),
    )
    side = int(adapter.side)
    model_name = "segformer_%s" % str(adapter.variant).lower()

    norm = "imagenet" if args.bake else "none"
    norm_desc = ("(x / 255 - mean) / std  [ImageNet]" if args.bake
                 else "不做（输入须已归一化）")

    payload = run_export(
        adapter=adapter,
        raw_loader=make_raw_loader(side),
        pipeline_predict=make_pipeline(adapter),
        model_name=model_name,
        out_dir=os.path.abspath(args.out),
        side=side,
        norm=norm,
        norm_desc=norm_desc,
        opset=args.opset,
        dynamic_batch=True,
        dynamic_hw=args.dynamic_hw,
        n_verify=args.limit_verify,
        iters=args.iters,
        batch=args.batch,
        fp16=args.fp16,
        ort_fp16=args.ort_fp16,
        ort_libs=args.ort_libs,
        ort_device=args.ort_device,
        input_dir=args.input,
        gt_dir=(args.gt_dir if args.gt else None),
        weights=weights,
    )
    v = payload["verify"]
    print("[验收] argmax 不一致像素 %d | 数据 %d 张 | 产物 %s"
          % (v["argmax_diff_px"], v["n_files"], payload["onnx"]))
    return 0


# =========================================================================== #
# ↓↓↓ 手动配置区：不想敲命令行就改这里，然后直接 `python export_onnx.py` ↓↓↓
#     命令行给了参数就优先用命令行的（这里的值只是默认值）
# =========================================================================== #
MANUAL = dict(
    weights=r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\W4\SegFormer\run\train\exp\weights\best_model_2.pth",          # None = 自动找 run/train/exp*/weights 下最新的 best_model_*.pth
    input=r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25\val\image",            # None = WUH_sub25/val/image
    gt_dir=r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25\val\label",           # None = WUH_sub25/val/label（为空则不算 IoU，只做数值一致性）
    out=None,              # None = 本工程 run/onnx
    input_shape=0,         # 0 = 从 train_config.txt 自动读（推荐）；否则须能被 32 整除
    variant=None,          # None = 自动读 b1/b2/b3
    decoder_dim=0,         # 0 = 自动读（本项目训练用 256）
    num_classes=0,         # 0 = 自动读（WHU 建筑二分类 = 1）
    opset=17,
    limit_verify=12,       # 一致性校验用几张图
    batch=8,               # 吞吐基准的 batch
    iters=30,
    dynamic_hw=False,      # True = H/W 也动态（换尺寸部署才需要，需自行验证精度）
    bake=True,             # True = 把 ImageNet 归一化烘进图，部署端不用再做
    fp16=True,             # True = 顺便测 GPU FP16 延迟（无 CUDA 会自动跳过）
    device="cuda",
)


if __name__ == "__main__":
    sys.exit(main())

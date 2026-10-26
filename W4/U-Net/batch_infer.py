# -*- coding: utf-8 -*-
"""U-Net 批量推理（断点续跑版）。

和 predict.py 的分工
--------------------
- predict.py：保持原样，交互式看少量图用，输出"原图+掩膜"的混合图。
- batch_infer.py（本文件）：批量入口。一次跑完整个目录，支持中断续跑、
  失败重试、清单落盘、吞吐统计。推理内核与 batch_runner.py 分离。

相对 predict.py 修正的四处口径问题（都属于"能跑但结果不对"的隐形问题）
--------------------------------------------------------------------------
1. **输入尺寸**：predict.py 把 input_shape 写死成 [480, 480]（predict.py:55），
   而训练用的是 512（train.py:146）。尺寸与训练不一致会明显掉点。
   本脚本默认 512，可用 --input-shape 覆盖（必须能被 32 整除）。
2. **掩膜内容**：predict.py 只存"原图 + 掩膜的混合图"（predict.py:92-108），
   拿去做指标或转 GeoJSON 都不行。本脚本默认存 uint8 索引图（0 背景 / 1 建筑），
   叠加图另存到 overlays/。想要 predict.py 那种彩色图用 --mask-mode color。
3. **缩回原图的插值**：predict.py 先把"概率图"双线性放大再 argmax
   （predict.py:86-89）。类别索引不能线性插值（会插出不存在的类别），
   本脚本在网络分辨率上取 argmax，再用最近邻缩回原尺寸，
   与 SegFormer 工程 predict.py:245 的口径一致。
4. **批量化 / 可续跑**：predict.py 逐张 forward、无清单、中断即从头再来。
   本脚本按 --batch-size 成批前向，每张图落盘后才写 manifest，断了能接着跑。

用法
----
    python batch_infer.py                         # 自动选最新权重，跑默认输入目录
    python batch_infer.py --limit 20              # 先抽 20 张冒烟
    python batch_infer.py --input D:/data/images  # 换成自己的目录（递归）
    python batch_infer.py                         # 中断后再执行同一条命令 = 续跑
    python batch_infer.py --retry-failed          # 只重跑上次失败的那几张
    python batch_infer.py --no-resume             # 从头全部重跑
    python batch_infer.py --weights run/train/exp1/weights/best_model_1.pth
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

from model.unet_resnet import Unet                                    # noqa: E402
from utils.utils import cvtColor, preprocess_input, resize_image      # noqa: E402
try:                                                                  # 标准布局：utils/batch_runner.py
    from utils.batch_runner import (                                  # noqa: E402
        BatchRunner, fmt_dur, list_images, sha1_of_file,
    )
except ModuleNotFoundError:  # 兼容 batch_runner.py 被挪到工程根目录的布局
    from batch_runner import (                                        # noqa: E402
        BatchRunner, fmt_dur, list_images, sha1_of_file,
    )

# 类别配色：与 SegFormer / DeepLabV3+ 工程一致，便于四个模型并排看
# 0 = 背景（黑），1 = 建筑（暗红）
PALETTE = np.array([[0, 0, 0], [128, 0, 0]], dtype=np.uint8)

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "WUH_sub25", "test", "image")
DEFAULT_OUT = os.path.join(_HERE, "run", "predict", "batch")


# --------------------------------------------------------------------------- Adapter
class UnetAdapter(object):
    """把 U-Net 的预处理/后处理封成引擎要的四个方法。

    预处理严格对齐训练期（train.py -> UnetDataset.__getitem__）：
    等比缩放 + 居中灰边填充到 input_shape，再 /255，不做 ImageNet 标准化。
    """

    def __init__(self, weights, device, input_shape=512, num_classes=1,
                 alpha=0.5, use_amp=False):
        if input_shape % 32 != 0:
            raise SystemExit(
                "--input-shape=%d 必须是 32 的整数倍（ResNet50 主干共下采样 32 倍），"
                "最近可用值：%d 或 %d"
                % (input_shape, input_shape // 32 * 32, (input_shape // 32 + 1) * 32))
        self.weights = weights
        self.device = device
        self.side = int(input_shape)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.use_amp = bool(use_amp)
        self.model = self._build()

    # ---------------------------------------------------------------- 建模
    def _build(self):
        model = Unet(num_classes=self.num_classes + 1)

        ckpt = torch.load(self.weights, map_location="cpu")
        for key in ("state_dict", "model", "net"):       # 兼容被外层字典包过一层
            if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
        ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}

        own = model.state_dict()
        hit = {k: v for k, v in ckpt.items()
               if k in own and tuple(own[k].shape) == tuple(v.shape)}
        miss_model = [k for k in own if k not in hit]
        miss_ckpt = [k for k in ckpt if k not in hit]
        cover = len(hit) / max(len(own), 1)

        # 覆盖率不足就明说，别让它"能加载多少算多少"然后输出噪声（这类静默失败最难查）
        if cover < 0.95 or miss_model:
            raise SystemExit(
                "权重与模型结构不匹配：对上 %d/%d 个参数（覆盖 %.0f%%）。\n"
                "  模型缺: %s\n  权重多: %s\n"
                "  请确认 --num-classes 与训练时一致（WHU 建筑二分类填 1）。"
                % (len(hit), len(own), cover * 100,
                   miss_model[:4] or "无", miss_ckpt[:4] or "无"))

        model.load_state_dict(hit, strict=False)
        model.eval().to(self.device)
        print("[CKPT] %s" % self.weights)
        print("[CKPT] 加载 %d/%d 个参数（覆盖 %.0f%%）| 类别 %d（含背景）"
              % (len(hit), len(own), cover * 100, self.num_classes + 1))
        return model

    # ---------------------------------------------------------------- 前处理
    def prep(self, path):
        image = cvtColor(Image.open(path))
        iw, ih = image.size
        canvas, nw, nh = resize_image(image, (self.side, self.side))
        top = (self.side - nh) // 2
        left = (self.side - nw) // 2

        arr = preprocess_input(np.array(canvas, np.float64))     # -> 0~1
        x = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return {"x": x, "meta": {"orig": (iw, ih), "crop": (top, left, nh, nw)}}

    # ---------------------------------------------------------------- 推理
    @torch.inference_mode()
    def infer(self, xs):
        tensor = torch.from_numpy(np.ascontiguousarray(xs)).to(self.device)
        if self.use_amp and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                logits = self.model(tensor)
        else:
            logits = self.model(tensor)
        pred = logits.float().argmax(dim=1).to(torch.uint8).cpu().numpy()
        return [pred[i] for i in range(pred.shape[0])]

    def on_oom(self):
        """显存不足时的自救钩子：把缓存块还给驱动，再让引擎对半拆批重试。"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 后处理
    def post(self, mask_small, meta):
        top, left, nh, nw = meta["crop"]
        iw, ih = meta["orig"]
        core = mask_small[top:top + nh, left:left + nw]
        if core.size == 0:                        # 极端小图兜底
            core = mask_small
        if core.shape[1] != iw or core.shape[0] != ih:
            core = np.array(Image.fromarray(core).resize((iw, ih), Image.NEAREST))
        return core.astype(np.uint8)

    def overlay(self, mask, path):
        rgb = np.array(cvtColor(Image.open(path)).convert("RGB"))
        color = self.colorize(mask)
        out = rgb.astype(np.float32) * (1 - self.alpha) + color.astype(np.float32) * self.alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    def colorize(self, mask):
        idx = np.clip(np.asarray(mask, dtype=np.int64), 0, len(PALETTE) - 1)
        return PALETTE[idx]


# --------------------------------------------------------------------------- 权重定位
def find_latest_weights(root=None, verbose=True):
    """在 run/train/<exp>/weights/ 下找最新的 best_model_*.pth。

    不用 glob：工程路径含 `[C]`，方括号会被当成字符类，静默匹配不到。
    """
    root = root or os.path.join(_HERE, "run", "train")
    if not os.path.isdir(root):
        raise SystemExit("找不到权重目录：%s" % root)
    hits = []
    for exp in sorted(os.listdir(root)):
        wdir = os.path.join(root, exp, "weights")
        if not os.path.isdir(wdir):
            continue
        for name in sorted(os.listdir(wdir)):
            if name.startswith("best_model_") and name.lower().endswith(".pth"):
                hits.append(os.path.join(wdir, name))
    if not hits:
        raise SystemExit("%s 下没有 best_model_*.pth，先用 --weights 指定" % root)
    hits.sort(key=os.path.getmtime, reverse=True)
    if verbose:
        print("[自动] 候选权重 %d 个，取最新：%s" % (len(hits), hits[0]))
        for p in hits[1:4]:
            print("        次新：%s" % p)
    return hits[0]


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="U-Net 批量推理（断点续跑）：一次跑完整个目录，断了能接着跑",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=[DEFAULT_INPUT], nargs="+", metavar="PATH",
                   help="图片、目录（递归）或文本清单；可以连着写多个")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="输出目录。**续跑依赖它稳定**，默认 run/predict/batch")
    p.add_argument("--weights", default=None,
                   help="权重路径；不给就取 run/train/*/weights 下最新的 best_model_*.pth")
    p.add_argument("--input-shape", default=512, type=int,
                   help="网络输入边长，必须 32 的整数倍；训练用的是 512")
    p.add_argument("--num-classes", default=1, type=int,
                   help="前景类数（不含背景），WHU 建筑填 1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", default=8, type=int, help="一次前向几张")
    p.add_argument("--prefetch", default=4, type=int, help="读图线程数")
    p.add_argument("--limit", default=0, type=int, help=">0 时只跑前 N 张，冒烟用")
    p.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                   help="断点续跑：跳过已完成且产物完整的图")
    p.add_argument("--retry-failed", default=False, action="store_true",
                   help="只重跑 failed.csv 里的图")
    p.add_argument("--force-continue", default=False, action="store_true",
                   help="任务指纹（权重/尺寸/类别）与输出目录里的记录不一致时仍继续")
    p.add_argument("--save-mask", default=True, action=argparse.BooleanOptionalAction,
                   help="保存掩膜到 masks/")
    p.add_argument("--save-overlay", default=True, action=argparse.BooleanOptionalAction,
                   help="保存叠加图到 overlays/")
    p.add_argument("--mask-mode", default="index", choices=["index", "color"],
                   help="index=uint8 索引图（可评估）；color=彩色图（同 predict.py）")
    p.add_argument("--alpha", default=0.5, type=float, help="叠加图里掩膜的透明度")
    p.add_argument("--log-every", default=20, type=int, help="每 N 张打一行进度")
    p.add_argument("--amp", default=False, action="store_true",
                   help="半精度前向（更快，边界略有差异）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 --device cuda 但当前环境没有可用 GPU")
    print("[环境] torch %s | device %s | %s"
          % (torch.__version__, device,
             torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / float(1024 ** 3)
        print("[显存] 总 %.1f GB | 当前空闲 %.1f GB | 本次 --batch-size %d"
              % (total_b / float(1024 ** 3), free_gb, args.batch_size))
        if free_gb < 4.0:
            print("[提示] 空闲显存不到 4 GB：512 输入下先把 --batch-size 降到 4 试；"
                  "设太大会让显卡把显存溢出到内存，系统内存冲到 90% 以上且吞吐骤降")

    weights = args.weights or find_latest_weights()
    if not os.path.isfile(weights):
        raise SystemExit("权重不存在：%s" % weights)
    weights = os.path.abspath(weights)

    # nargs="+" 的 default 必须是 list；写成一个字符串时 argparse 会原样返回，
    # list("D:\...") 会被拆成 ['D', ':', '\', ...] 逐字符当路径 → "路径不存在：D"
    inputs = args.input if isinstance(args.input, (list, tuple)) else [args.input]
    srcs = list_images(inputs)
    if args.limit and args.limit > 0:
        srcs = srcs[:args.limit]
    print("[输入] %d 张（来自 %s）" % (len(srcs), ", ".join(os.path.abspath(p) for p in inputs[:3])))

    print("[指纹] 计算权重 sha1 ...")
    t_sig = time.time()
    digest = sha1_of_file(weights)
    print("[指纹] %s (%d MB, 用时 %s)"
          % (digest[:16], os.path.getsize(weights) // (1 << 20), fmt_dur(time.time() - t_sig)))

    adapter = UnetAdapter(weights=weights, device=device,
                          input_shape=args.input_shape, num_classes=args.num_classes,
                          alpha=args.alpha, use_amp=args.amp)

    signature = dict(
        task="batch_infer", model="unet", norm="unit",
        weights=weights, weights_sha1=digest, weights_bytes=os.path.getsize(weights),
        input_shape=args.input_shape, num_classes=args.num_classes,
        mask_mode=args.mask_mode, save_mask=args.save_mask,
        save_overlay=args.save_overlay, alpha=args.alpha,
        inputs=[os.path.abspath(p) for p in inputs],
    )

    runner = BatchRunner(
        out_dir=args.out, adapter=adapter,
        batch_size=args.batch_size, prefetch=args.prefetch,
        resume=args.resume, retry_failed=args.retry_failed,
        force_continue=args.force_continue, save_mask=args.save_mask,
        save_overlay=args.save_overlay, mask_mode=args.mask_mode,
        alpha=args.alpha, log_every=args.log_every)
    runner.prepare(signature)
    summary = runner.run(srcs)
    if device.type == "cuda":
        print("[显存] 本次峰值（张量已分配）%.0f MB"
              % (torch.cuda.max_memory_allocated() / float(1024 ** 2)))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

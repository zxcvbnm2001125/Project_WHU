# -*- coding: utf-8 -*-
"""SegFormer 批量推理（断点续跑版）。

和 predict.py 的分工
--------------------
- predict.py：保持原样，交互式看少量图用（逐张推理 + tqdm + 每次新建 run/predict/expN）。
- batch_infer.py（本文件）：批量入口。批量化前向、中断续跑、失败重试、清单落盘。

本脚本**复用** predict.py 里已经验证过的几件东西，不复制粘贴：
    find_latest_weights()  权重自动定位（用 listdir，不用 glob，避开路径里的 [C]）
    read_train_config()    从 train_config.txt 读回 variant / decoder_dim / num_classes
    load_model()           建模型 + 部分加载 + 双向覆盖率校验（规格填错直接报错）
    build_palette() / colorize() / overlay()   配色与叠加

相对 predict.py 的三处改动
--------------------------
1. **批量化**：predict.py 一次 forward 一张（predict.py:241）。这里按 --batch-size 叠批，
   512x512 的瓦片在 8 GB 显存上 bs=8 很宽松。
2. **索引掩膜**：predict.py 的 masks/ 存的是彩色图（predict.py:287），不能直接拿去算
   指标或转 GeoJSON。这里默认存 uint8 索引图，--mask-mode color 可切回彩色。
3. **可续跑**：predict.py 每跑一次就 create_val_exp_folder() 新建一个 expN
   （predict.py:267），断点续跑没有稳定目录。这里用固定的 --out，
   每张图落盘后才写 manifest，重跑自动跳过已完成的。

为什么训练用 bf16、这里用 fp32
------------------------------
train_config.txt 里 precision=bf16 是训练加速用的。推理改成 fp32 只是数值上略有差异
（同一权重的 argmax 结果通常完全一致），换来的是不依赖显卡对 bf16 的支持。

用法
----
    python batch_infer.py                          # 自动找最新权重 + 读 train_config
    python batch_infer.py --limit 20               # 冒烟
    python batch_infer.py --input D:/data/images   # 换输入（可多个）
    python batch_infer.py                          # 中断后重跑同一条命令 = 续跑
    python batch_infer.py --retry-failed           # 只重跑上次失败的
    python batch_infer.py --weights run/train/exp1/weights/best_model_2.pth --variant b2
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from predict import (                                                 # noqa: E402
    build_palette, colorize, find_latest_weights, load_model,
    overlay as blend_overlay, read_train_config,
)
from utils.utils import cvtColor, preprocess_input                    # noqa: E402
try:                                                                  # 标准布局：utils/batch_runner.py
    from utils.batch_runner import (                                  # noqa: E402
        BatchRunner, fmt_dur, list_images, sha1_of_file,
    )
except ModuleNotFoundError:  # 兼容 batch_runner.py 被挪到工程根目录的布局
    from batch_runner import (                                        # noqa: E402
        BatchRunner, fmt_dur, list_images, sha1_of_file,
    )

DEFAULT_INPUT = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "WUH_sub25", "test", "image")
DEFAULT_OUT = os.path.join(_HERE, "run", "predict", "batch")


# --------------------------------------------------------------------------- Adapter
class SegFormerAdapter(object):
    """预处理严格对齐验证期（predict.py:222-246）：

    等比缩放 + 居中灰边填充到 input_shape -> ImageNet 归一化 ->
    前向（模型内部已把 logits 插值回输入分辨率，见 model/segformer.py:181）-> argmax ->
    裁掉灰边 -> 最近邻缩回原尺寸。
    """

    def __init__(self, weights, device, variant=None, decoder_dim=None,
                 num_classes=None, input_shape=None, alpha=0.5):
        cfg = read_train_config(weights)
        self.weights = weights
        self.device = device
        self.alpha = float(alpha)

        # 优先级：命令行 > train_config.txt > 兜底默认值（与 predict.py:125-131 一致）
        self.variant = variant or cfg.get("variant") or "b3"
        self.decoder_dim = int(decoder_dim or cfg.get("decoder_dim") or 256)
        self.num_classes = int(num_classes or cfg.get("num_classes") or 1)
        self.side = int(input_shape or cfg.get("input_shape") or 512)

        if cfg:
            print("[自动] 训练配置 %s" % os.path.join(
                os.path.dirname(os.path.dirname(weights)), "train_config.txt"))
        if variant is not None and cfg.get("variant") and variant != cfg["variant"]:
            print("[警告] 你指定 variant=%s，但该权重训练时是 variant=%s。"
                  "\n       除非确实要跨规格加载，否则删掉 --variant 让脚本自动读取。"
                  % (variant, cfg["variant"]))
        print("[Model] SegFormer-%s | decoder_dim %d | 类别 %d（含背景）| 输入 %d"
              % (self.variant.upper(), self.decoder_dim,
                 self.num_classes + 1, self.side))

        ns = SimpleNamespace(weights=weights, variant=self.variant,
                             num_classes=self.num_classes,
                             decoder_dim=self.decoder_dim,
                             input_shape=self.side)
        self.model = load_model(ns, device)          # 内含双向覆盖率校验
        self.palette = build_palette(self.num_classes + 1)

    # ---------------------------------------------------------------- 前处理
    def prep(self, path):
        image = cvtColor(Image.open(path))
        iw, ih = image.size
        side = self.side
        scale = min(side / iw, side / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))

        canvas = Image.new("RGB", (side, side), (128, 128, 128))
        left, top = (side - nw) // 2, (side - nh) // 2
        canvas.paste(image.resize((nw, nh), Image.BICUBIC), (left, top))

        data = preprocess_input(np.array(canvas, np.float64), "imagenet")
        x = np.transpose(data, (2, 0, 1)).astype(np.float32)
        return {"x": x, "meta": {"orig": (iw, ih), "crop": (top, left, nh, nw)}}

    # ---------------------------------------------------------------- 推理
    @torch.inference_mode()
    def infer(self, xs):
        tensor = torch.from_numpy(np.ascontiguousarray(xs)).to(self.device)
        logits = self.model(tensor)                  # [N, C, side, side]
        # argmax(softmax(x)) == argmax(x)，省掉 softmax 只为省时间，结果不变
        pred = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
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
        if core.size == 0:
            core = mask_small
        if core.shape[1] != iw or core.shape[0] != ih:
            core = np.array(Image.fromarray(core).resize((iw, ih), Image.NEAREST))
        return core.astype(np.uint8)

    def overlay(self, mask, path):
        rgb = np.array(cvtColor(Image.open(path)).convert("RGB"))
        return blend_overlay(rgb, self.colorize(mask), self.alpha)

    def colorize(self, mask):
        return colorize(np.asarray(mask, dtype=np.uint8), self.palette)


# --------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SegFormer 批量推理（断点续跑）：一次跑完整个目录，断了能接着跑",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=[DEFAULT_INPUT], nargs="+", metavar="PATH",
                   help="图片、目录（递归）或文本清单；可以连着写多个")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="输出目录。**续跑依赖它稳定**，默认 run/predict/batch")
    p.add_argument("--weights", default=None,
                   help="权重路径；不给就取 run/train/*/weights 下最新的 best_model_*.pth")
    # 以下四项默认 None：不传就从权重的 train_config.txt 读，读不到再用兜底值
    p.add_argument("--variant", default=None, choices=["b0", "b1", "b2", "b3"],
                   help="必须与训练该权重时一致；不传则自动读取")
    p.add_argument("--decoder-dim", default=None, type=int,
                   help="必须与训练时一致；不传则自动读取")
    p.add_argument("--num-classes", default=None, type=int,
                   help="前景类数（不含背景）；不传则自动读取")
    p.add_argument("--input-shape", default=None, type=int,
                   help="网络输入边长；不传则自动读取")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", default=8, type=int, help="一次前向几张")
    p.add_argument("--prefetch", default=4, type=int, help="读图线程数")
    p.add_argument("--limit", default=0, type=int, help=">0 时只跑前 N 张，冒烟用")
    p.add_argument("--resume", default=True, action=argparse.BooleanOptionalAction,
                   help="断点续跑：跳过已完成且产物完整的图")
    p.add_argument("--retry-failed", default=False, action="store_true",
                   help="只重跑 failed.csv 里的图")
    p.add_argument("--force-continue", default=False, action="store_true",
                   help="任务指纹（权重/规格）与输出目录里的记录不一致时仍继续")
    p.add_argument("--save-mask", default=True, action=argparse.BooleanOptionalAction,
                   help="保存掩膜到 masks/")
    p.add_argument("--save-overlay", default=True, action=argparse.BooleanOptionalAction,
                   help="保存叠加图到 overlays/")
    p.add_argument("--mask-mode", default="index", choices=["index", "color"],
                   help="index=uint8 索引图（可评估）；color=彩色图（同 predict.py）")
    p.add_argument("--alpha", default=0.5, type=float, help="叠加图里掩膜的透明度")
    p.add_argument("--log-every", default=20, type=int, help="每 N 张打一行进度")
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

    inputs = list(args.input)
    srcs = list_images(inputs)
    if args.limit and args.limit > 0:
        srcs = srcs[:args.limit]
    print("[输入] %d 张（来自 %s）"
          % (len(srcs), ", ".join(os.path.abspath(p) for p in inputs[:3])))

    print("[指纹] 计算权重 sha1 ...")
    t_sig = time.time()
    digest = sha1_of_file(weights)
    print("[指纹] %s (%d MB, 用时 %s)"
          % (digest[:16], os.path.getsize(weights) // (1 << 20),
             fmt_dur(time.time() - t_sig)))

    adapter = SegFormerAdapter(
        weights=weights, device=device, variant=args.variant,
        decoder_dim=args.decoder_dim, num_classes=args.num_classes,
        input_shape=args.input_shape, alpha=args.alpha)

    signature = dict(
        task="batch_infer", model="segformer-" + adapter.variant, norm="imagenet",
        weights=weights, weights_sha1=digest, weights_bytes=os.path.getsize(weights),
        variant=adapter.variant, decoder_dim=adapter.decoder_dim,
        input_shape=adapter.side, num_classes=adapter.num_classes,
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

# -*- coding: utf-8 -*-
"""SegFormer 预测脚本。

交互与 U-Net / FCN / DeepLabV3+ 三个工程的 predict.py 保持一致：

  - `--input` 既可以是单张图片，也可以是文件夹（默认指向 WUH_sub25/test/image）
  - 结果写到 `run/predict/exp*/` 下，`masks/` 放彩色掩膜、`overlays/` 放原图叠加
  - 带进度条，结束时打印张数、总耗时与平均耗时

在此基础上多做了两件省事的事（另外三个工程没有，但 SegFormer 特别需要）：

  1. **权重自动定位**：`--weights` 留空时，自动取 `run/train/exp*/weights/best_model_2.pth`
     里最新的那个，并在屏幕上打印实际用了哪一份。
  2. **规格自动对齐**：从权重同目录的 `train_config.txt` 读回 variant / decoder_dim /
     num_classes / input_shape。SegFormer 的 b0~b3 参数量差十倍，规格填错会静默地
     只加载一部分参数、输出一堆噪声，这一步把它挡住；显式传参仍然优先。

预处理与验证期完全一致：等比缩放 + 居中灰边填充到 512，ImageNet 归一化；
前向得到 1/4 分辨率的 logits，插值回 512 后取 argmax，再裁掉灰边、还原到原图尺寸。

用法：
    python predict.py                              # 自动找权重，跑 test/image 全部
    python predict.py --input path/to/a.tif        # 单张
    python predict.py --max-images 20              # 先抽检 20 张
    python predict.py --weights run/train/exp1/weights/best_model_2.pth
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model.segformer import SegFormer
from utils.create_exp_folder import create_val_exp_folder
from utils.utils import cvtColor, load_weights_partial, preprocess_input

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABLATION = os.path.dirname(os.path.dirname(_HERE))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =============================================================================
# 预测配置：改这里即可，命令行不传参时全部生效
# =============================================================================
CONFIG = dict(
    input=os.path.join(_ABLATION, r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25\test\image"),
    weights=r"run\train\exp2\weights\best_model_2.pth",           # 留空 = 自动选 run/train/exp*/weights 下最新的 best_model_2.pth
    variant="b3",         # 仅在找不到 train_config.txt 时作为兜底
    num_classes=1,        # 前景类数（不含背景）；WHU 建筑填 1
    decoder_dim=256,
    input_shape=512,
    device="cuda",
    alpha=0.5,            # overlay 里掩膜的透明度
    save_mask=True,       # 保存彩色掩膜（masks/<stem>_mask.tif）
    save_overlay=True,    # 保存原图 + 掩膜叠加（overlays/<stem>_overlay.tif）
    max_images=0,         # >0 时只跑前 N 张，用于快速抽检
)


# ---------------------------------------------------------------- 权重与规格
def find_latest_weights():
    """在 run/train/<exp>/weights/ 下找最新的 best_model_*.pth。

    刻意不用 glob：工程路径里含 `[C]`（`D://[C]Remote_...`），
    glob 会把方括号当成字符类，导致匹配不到任何文件。这里用 listdir 逐层遍历。
    """
    root = os.path.join(_HERE, "run", "train")
    if not os.path.isdir(root):
        return None
    hits = []
    for exp in sorted(os.listdir(root)):
        wdir = os.path.join(root, exp, "weights")
        if not os.path.isdir(wdir):
            continue
        for fn in os.listdir(wdir):
            if fn.startswith("best_model_") and fn.lower().endswith(".pth"):
                hits.append(os.path.join(wdir, fn))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def read_train_config(weights_path):
    """读取权重同目录（上一级）的 train_config.txt，返回 dict。

    训练脚本会把所有 argparse 参数写进去，这里只取与建模型相关的几个。
    """
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(weights_path)), "train_config.txt")
    if not os.path.isfile(cfg_path):
        return {}
    cfg = {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def resolve_settings(args):
    """确定权重路径，并把 variant / decoder_dim / num_classes / input_shape 对齐到训练时的值。

    命令行显式给出的参数优先级最高（args 里非 None 就说明用户传了）。
    """
    weights = args.weights
    if not weights:
        weights = find_latest_weights()
        if weights is None:
            raise SystemExit(
                "找不到权重：run/train/ 下没有 best_model_*.pth。\n"
                "  先训练，或用 --weights 显式指定路径。")
        print(f"[自动] 使用最新权重：{weights}")
    if not os.path.isfile(weights):
        raise SystemExit(f"权重不存在：{weights}")

    cfg = read_train_config(weights)
    if cfg:
        print(f"[自动] 读取训练配置：{os.path.dirname(os.path.dirname(weights))}/train_config.txt")

    def pick(name, cli_value, cast=str):
        """命令行 > train_config.txt > CONFIG 兜底。"""
        if cli_value is not None:
            return cli_value
        if name in cfg:
            return cast(cfg[name])
        return cast(CONFIG[name])

    # 显式传了 variant 却和权重的训练配置对不上，先说一句（b1/b3 的部分张量形状相同，
    # 只靠后面的形状校验可能仍是"大部分能对上"，容易静默跑出偏低的精度）
    if args.variant is not None and "variant" in cfg and args.variant != cfg["variant"]:
        print(f"[警告] 你指定 variant={args.variant}，但该权重训练时是 variant={cfg['variant']}。"
              f"\n       除非你确实想跨规格加载，否则请删掉 --variant 让脚本自动读取。")

    args.weights = weights
    args.variant = pick("variant", args.variant)
    args.decoder_dim = pick("decoder_dim", args.decoder_dim, int)
    args.num_classes = pick("num_classes", args.num_classes, int)
    args.input_shape = pick("input_shape", args.input_shape, int)

    print(f"[Model] SegFormer-{args.variant.upper()} | decoder_dim {args.decoder_dim} | "
          f"类别 {args.num_classes + 1}（含背景）| 输入 {args.input_shape}")
    return args


# ---------------------------------------------------------------- 颜色
def build_palette(num_classes):
    """与 DeepLabV3+ 的 build_voc_palette 同一套配色，便于四模型结果并排看。

    类别 0（背景）黑，类别 1（建筑）暗红 RGB(128,0,0)。
    """
    def _bit(v, i):
        return (v & (1 << i)) != 0

    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= int(_bit(c, 0)) << (7 - j)
            g |= int(_bit(c, 1)) << (7 - j)
            b |= int(_bit(c, 2)) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


def colorize(mask, palette):
    """mask[H,W]（0..C-1）→ 彩色 [H,W,3]。"""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        out[mask == cls_id] = palette[cls_id]
    return out


def overlay(rgb, color_mask, alpha):
    """原图与彩色掩膜按 alpha 混合，alpha 越大掩膜越明显。"""
    out = rgb.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 模型
def load_model(args, device):
    model = SegFormer(variant=args.variant,
                      num_classes=args.num_classes + 1,
                      decoder_dim=args.decoder_dim,
                      dropout=0.0,
                      drop_path_rate=0.0,
                      img_size=args.input_shape)
    load_weights_partial(model, args.weights, map_location="cpu")

    # 规格填错时 load_weights_partial 会"能加载多少算多少"，输出会是噪声而不是报错。
    # 这里做一次覆盖率检查，把静默失败变成显式报错。
    sd = torch.load(args.weights, map_location="cpu")
    for key in ("state_dict", "model", "net"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    model_sd = model.state_dict()
    matched = sum(1 for k, v in sd.items()
                  if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape))
    # 双向覆盖：只看"模型被填满多少"会漏掉跨规格的情况——b1 与 b3 的多数层形状相同，
    # 拿 b3 权重喂 b1 模型也能填满九成参数，但权重里有大量层根本没用上。
    cover_model = matched / max(len(model_sd), 1)
    cover_ckpt = matched / max(len(sd), 1)
    if cover_model < 0.9 or cover_ckpt < 0.9:
        raise SystemExit(
            f"权重与模型结构不匹配：对上 {matched} 个参数（模型覆盖 {cover_model:.0%}，"
            f"权重使用 {cover_ckpt:.0%}）。\n"
            f"  当前规格：variant={args.variant}, decoder_dim={args.decoder_dim}, "
            f"num_classes={args.num_classes}\n"
            f"  请确认与该权重训练时一致，或删掉 --variant/--decoder-dim 让脚本自动读取。")

    model.eval().to(device)
    return model


@torch.no_grad()
def infer_one_image(model, pil_img, input_shape, device):
    """单张推理，返回与原图同尺寸的类别图 [H,W]（uint8）。

    预处理与验证期一致：等比缩放 + 居中灰边填充，不拉伸变形。
    """
    img = cvtColor(pil_img)
    iw, ih = img.size
    side = input_shape

    scale = min(side / iw, side / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    left, top = (side - nw) // 2, (side - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), (left, top))

    data = preprocess_input(np.array(canvas, np.float64), "imagenet")
    tensor = torch.from_numpy(np.transpose(data, (2, 0, 1))[None]).float().to(device)

    prob = torch.softmax(model(tensor), dim=1)
    pred = prob.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()

    # 裁掉灰边，再还原到原图尺寸（最近邻，避免类别被插值污染）
    pred = pred[top:top + nh, left:left + nw]
    return np.array(Image.fromarray(pred).resize((iw, ih), Image.NEAREST))


# ---------------------------------------------------------------- 批处理
def run_on_images(model, args, device, palette):
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{in_path}")

    if in_path.is_file():
        img_list = [in_path]
    else:
        img_list = sorted(p for p in in_path.iterdir()
                          if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.max_images and args.max_images > 0:
        img_list = img_list[:args.max_images]

    if not img_list:
        print(f"[Predict] {in_path} 下没有可识别的图片（支持 {sorted(IMG_EXTS)}）")
        return

    # --out 传了就用固定目录（可重复覆盖同一份结果），不传则维持"每次新建 expN"的原行为，
    # 因为断点续跑/对比看结果需要一个稳定路径，而临时抽检不需要污染历史目录。
    exp_dir = Path(args.out) if getattr(args, "out", None) else Path(create_val_exp_folder())
    mask_dir = exp_dir / "masks"
    overlay_dir = exp_dir / "overlays"
    label_dir = exp_dir / "labels"
    if args.save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "save_index", False):
        label_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Folder] 预测结果目录: {exp_dir}")
    print(f"[Predict] 输入 {in_path}，共 {len(img_list)} 张")

    total_fg = 0.0
    t0 = time.time()
    for p in tqdm(img_list, desc="Infer Images", ncols=100):
        pil_img = Image.open(p).convert("RGB")
        pred = infer_one_image(model, pil_img, args.input_shape, device)

        color = colorize(pred, palette)
        stem = p.stem
        if getattr(args, "save_index", False):
            # uint8 索引图（0 背景 / 1 建筑），与 batch_infer.py 的 --mask-mode index 同一口径，
            # 拿去做指标、转 GeoJSON、算面积都能直接用
            Image.fromarray(pred.astype(np.uint8)).save(label_dir / f"{stem}_label.tif")
        if args.save_mask:
            Image.fromarray(color).save(mask_dir / f"{stem}_mask.tif")
        if args.save_overlay:
            ov = overlay(np.array(pil_img), color, args.alpha)
            Image.fromarray(ov).save(overlay_dir / f"{stem}_overlay.tif")

        total_fg += float((pred == 1).mean())

    dt = time.time() - t0
    print(f"[Predict] 完成 {len(img_list)} 张 | 总耗时 {dt:.1f}s | "
          f"平均 {dt * 1000 / len(img_list):.0f} ms/张 | "
          f"前景占比均值 {total_fg / len(img_list):.2%}")
    if args.save_mask:
        print(f"[Predict] 彩色掩膜: {mask_dir}")
    if args.save_overlay:
        print(f"[Predict] 叠加图  : {overlay_dir}")
    if getattr(args, "save_index", False):
        print(f"[Predict] 索引掩膜: {label_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="SegFormer 预测（单张 / 文件夹）")
    p.add_argument("--input", default=CONFIG["input"],
                   help="输入图片或文件夹（默认 WUH_sub25/test/image）")
    p.add_argument("--weights", default=CONFIG["weights"],
                   help="权重路径；留空则自动取 run/train/exp*/weights 下最新的 best_model_*.pth")
    # 以下四项默认 None：不传就自动从权重的 train_config.txt 里读，读不到再用 CONFIG 兜底
    p.add_argument("--variant", default=None, choices=["b0", "b1", "b2", "b3"],
                   help="必须与训练该权重时的规格一致；不传则自动读取")
    p.add_argument("--num-classes", default=None, type=int,
                   help="前景类数（不含背景）；不传则自动读取")
    p.add_argument("--decoder-dim", default=None, type=int,
                   help="必须与训练时一致；不传则自动读取")
    p.add_argument("--input-shape", default=None, type=int,
                   help="网络输入边长；不传则自动读取")
    p.add_argument("--device", default=CONFIG["device"])
    p.add_argument("--alpha", default=CONFIG["alpha"], type=float, help="overlay 透明度（0~1）")
    p.add_argument("--save-mask", default=CONFIG["save_mask"], action=argparse.BooleanOptionalAction,
                   help="保存彩色掩膜到 masks/")
    p.add_argument("--save-overlay", default=CONFIG["save_overlay"],
                   action=argparse.BooleanOptionalAction, help="保存原图叠加到 overlays/")
    p.add_argument("--save-index", default=False, action=argparse.BooleanOptionalAction,
                   help="另外保存 uint8 索引掩膜到 labels/（与 batch_infer.py 同一口径）")
    p.add_argument("--out", default=None,
                   help="固定输出目录；不传则每次新建 run/predict/expN")
    p.add_argument("--max-images", default=CONFIG["max_images"], type=int,
                   help=">0 时只跑前 N 张，用于快速抽检")
    return p.parse_args()


def main():
    args = resolve_settings(parse_args())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args, device)
    print(f"[CKPT] {args.weights}")

    run_on_images(model, args, device, build_palette(args.num_classes + 1))


if __name__ == "__main__":
    main()

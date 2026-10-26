# predict.py
# -*- coding: utf-8 -*-
"""
DeepLabV3+ 推理脚本（支持：图片 / 文件夹）
- 自动创建输出目录：run/predict/exp*
- 支持保存：
    1) 彩色 mask（每个像素按类别上色）
    2) overlay（原图 + mask 叠加，可调透明度）
"""

import argparse
import os
import sys
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

import network
from utils.create_exp_folder import create_val_exp_folder


# -----------------------------
# 1) VOC 调色板（固定使用 voc）
# -----------------------------
def voc_cmap(N=256, normalized=False):
    """
    VOC 官方常用的调色板生成方法：
    输入类别 id（0~255）-> 输出 RGB 颜色
    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        cmap[i] = np.array([r, g, b])
    return (cmap / 255.0) if normalized else cmap


def build_voc_palette(num_classes: int) -> np.ndarray:
    """
    固定使用 VOC 调色板（稳定、可复现、颜色好看）
    返回 shape: [num_classes, 3] uint8
    """
    cmap = voc_cmap(256, normalized=False)
    return cmap[:num_classes]


# -----------------------------
# 2) checkpoint 加载兼容 + 自动推断 separable_conv
# -----------------------------
def _torch_load_compat(path: str, map_location="cpu"):
    """
    兼容不同 PyTorch 版本的 torch.load：
    - PyTorch 1.10：torch.load 不支持 weights_only 参数
    - PyTorch 2.x：torch.load 可能支持 weights_only，需要显式 weights_only=False（你信任 ckpt 来源时）
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    """
    处理 DataParallel 保存的权重（key 带 "module." 前缀）
    """
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k[len("module."):]] = v
    return new_sd


def infer_separable_conv_from_ckpt(ckpt: dict) -> bool:
    """
    自动推断训练时是否启用了 separable conv。

    优先级：
      1) 如果 ckpt 里保存了 'separable_conv' 字段 -> 直接用（最准确）
      2) 否则从 state_dict 的 key 名推断：
         - 只在 classifier.* 下查找 'depthwise'/'pointwise'
           （避免 backbone 本身含 depthwise 的误判，比如 Xception）
    """
    if isinstance(ckpt, dict) and "separable_conv" in ckpt:
        return bool(ckpt["separable_conv"])

    state = None
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", None)
    if not isinstance(state, dict):
        return False

    for k in state.keys():
        if k.startswith("classifier.") and ("depthwise" in k or "pointwise" in k):
            return True
    return False


# -----------------------------
# 3) 模型构建（自动过滤不支持的 kwargs，避免 unexpected keyword arg）
# -----------------------------
def build_model_from_opts(opts, separable_conv: bool):
    """
    从 network.modeling 里拿工厂函数构建模型。
    关键：只传“工厂函数签名里支持的参数”，避免报错。
    """
    model_fn = network.modeling.__dict__.get(opts.model, None)
    if model_fn is None:
        raise ValueError(f"[Predict] Unknown model name: {opts.model}")

    # 统一准备一套“可能用到”的参数，然后按签名过滤
    kwargs = {
        "num_classes": opts.num_classes,
        "output_stride": opts.output_stride,
        "separable_conv": separable_conv,
        # 推理时一般不需要 pretrained_backbone，但有些工厂函数签名里有它
        "pretrained_backbone": False,
    }

    sig = inspect.signature(model_fn)
    allowed = set(sig.parameters.keys())
    safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    model = model_fn(**safe_kwargs)
    return model


def load_checkpoint_strict(model, ckpt_path: str):
    """
    固定 strict=True 加载权重（推理建议必须严格匹配）
    - 兼容两种格式：
        1) {"model_state": state_dict, ...}  （train.py 保存的 ckpt）
        2) 直接就是 state_dict            （某些外部权重）
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[Predict] ckpt not found: {ckpt_path}")

    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    state_dict = strip_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("[CKPT] strict 加载失败：模型结构与权重不匹配。")
        print(f"  你当前 --model      : {getattr(model, '__class__', type(model)).__name__}")
        print(f"  你当前 --num_classes: {getattr(getattr(model, 'classifier', None), 'num_classes', 'N/A')}")
        print("  常见原因：")
        print("    - 训练时 num_classes 与现在不一致（例如训练是 21 类，但你现在是 3 类）")
        print("    - --model 选择和权重来源的模型不一致（mobilenet vs xception/resnet）")
        print("  解决建议：")
        print("    - 推理必须使用和训练时完全一致的：model / output_stride / num_classes（以及是否 separable conv）")
        raise e

    return ckpt


def load_model_for_predict(opts, device):
    """
    推理专用：自动推断 separable_conv -> 构建模型 -> strict 加载权重。
    """
    if opts.ckpt is None or not os.path.isfile(opts.ckpt):
        raise FileNotFoundError(f"找不到权重文件：{opts.ckpt}")

    ckpt = _torch_load_compat(opts.ckpt, map_location="cpu")
    separable_conv = infer_separable_conv_from_ckpt(ckpt)
    print(f"[Auto] separable_conv inferred = {separable_conv}")

    model = build_model_from_opts(opts, separable_conv=separable_conv)

    # 重新用 strict=True 加载（使用同一套兼容逻辑 + strip_module_prefix）
    load_checkpoint_strict(model, opts.ckpt)

    model.to(device)
    model.eval()
    return model


# -----------------------------
# 4) 输入预处理 + 推理
# -----------------------------
def preprocess_pil(pil_img: Image.Image):
    """
    推理时最简预处理（与你训练时 normalize 保持一致）：
    - PIL -> numpy -> torch tensor
    - ImageNet mean/std 标准化
    """
    img = pil_img.convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0  # [H,W,3], 0~1

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # [H,W,3] -> [3,H,W]
    tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1,3,H,W]
    return tensor


@torch.no_grad()
def infer_one_image(model, device, pil_img: Image.Image):
    """
    对单张 PIL 图像推理，返回 pred_mask（H,W，类别 id）
    """
    x = preprocess_pil(pil_img).to(device)
    logits = model(x)  # 你的 wrapper 一般会插值回输入大小
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # [H,W]
    return pred


def colorize_mask(mask: np.ndarray, palette: np.ndarray):
    """
    mask: [H,W]，值域 0~num_classes-1
    palette: [num_classes, 3]
    返回彩色 mask: [H,W,3] uint8
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        out[mask == cls_id] = palette[cls_id]
    return out


def overlay_image(rgb_img: np.ndarray, color_mask: np.ndarray, alpha: float):
    """
    rgb_img: [H,W,3] uint8
    color_mask: [H,W,3] uint8
    alpha: mask 透明度（0~1）
    """
    rgb = rgb_img.astype(np.float32)
    m = color_mask.astype(np.float32)
    out = rgb * (1 - alpha) + m * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# 5) 推理：图片 / 文件夹
# -----------------------------
def is_image_file(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def run_on_images(model, device, input_path: Path, out_dir: Path, palette: np.ndarray,
                  save_mask: bool, save_overlay: bool, alpha: float):
    """
    支持输入：
    - 单张图片
    - 文件夹（非递归：只扫当前目录）
    """
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    if save_mask:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        img_list = [input_path]
    else:
        img_list = sorted([p for p in input_path.iterdir() if p.is_file() and is_image_file(p)])

    if len(img_list) == 0:
        print(f"[Predict] No images found in: {input_path}")
        return

    for img_path in tqdm(img_list, desc="Infer Images", ncols=100):
        pil_img = Image.open(img_path).convert("RGB")
        pred = infer_one_image(model, device, pil_img)

        # mask 上色
        color_mask = colorize_mask(pred, palette)

        stem = img_path.stem
        if save_mask:
            Image.fromarray(color_mask).save(mask_dir / f"{stem}_mask.png")

        if save_overlay:
            rgb = np.array(pil_img).astype(np.uint8)
            ov = overlay_image(rgb, color_mask, alpha=alpha)
            Image.fromarray(ov).save(overlay_dir / f"{stem}_overlay.png")


# -----------------------------
# 6) 主函数
# -----------------------------
def main(opts):
    # 1) 创建输出目录：run/predict/exp*
    exp_dir = Path(create_val_exp_folder())
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Folder] Predict exp dir: {exp_dir}")

    # 2) 设备
    device = torch.device(opts.device if torch.cuda.is_available() and "cuda" in opts.device else "cpu")
    print(f"Device: {device}")

    # 3) 构建模型 + 自动推断 separable_conv + strict 加载权重
    model = load_model_for_predict(opts, device=device)
    print(f"[CKPT] Loaded weights: {opts.ckpt}")

    # 4) 调色板（固定 voc）
    palette = build_voc_palette(opts.num_classes)

    # 5) 输入路径
    in_path = Path(opts.input)
    if not in_path.exists():
        raise FileNotFoundError(f"[Predict] input not found: {in_path}")

    # 6) 推理（仅图片/文件夹）
    run_on_images(
        model=model, device=device, input_path=in_path, out_dir=exp_dir, palette=palette,
        save_mask=opts.save_mask, save_overlay=opts.save_overlay, alpha=opts.alpha
    )

    print("[Predict] Done.")


if __name__ == "__main__":
    # ---------------- 模型相关 ----------------
    available_models = sorted(
        name for name in network.modeling.__dict__
        if name.islower()
        and not (name.startswith("__") or name.startswith("_"))
        and callable(network.modeling.__dict__[name])
    )

    parser = argparse.ArgumentParser()
    # ---------------- 输入/模型 ----------------
    parser.add_argument("--input", type=str, default=r"VOCdevkit-2/VOC2012/JPEGImages",
                        help="输入：单张图片路径 / 文件夹路径（只推理图片）")
    parser.add_argument("--ckpt", type=str, default=r"run/train/exp10/weights/best_deeplabv3plus_resnet50_os16_num2.pth",
                        help="模型权重路径（train.py 保存的 best/latest 都可以）")
    parser.add_argument("--model", type=str, default="deeplabv3plus_resnet50",
                        choices=available_models, help="模型名称（必须和权重匹配）")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数（含背景），必须与训练一致（否则 strict 加载会失败）")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16], help="输出步幅（必须与训练一致）")
    # ---------------- 输出形式 ----------------
    parser.add_argument("--save_mask", action="store_true", default=True, help="保存彩色 mask")
    parser.add_argument("--save_overlay", action="store_true", default=True, help="保存 overlay（原图+mask）")
    parser.add_argument("--alpha", type=float, default=0.5, help="overlay 透明度（0~1）")
    # ---------------- 设备 ----------------
    parser.add_argument("--device", type=str, default="cuda:0", help="设备：cuda:0 / cpu")
    opt = parser.parse_args()
    # 如果用户都关掉了，就默认保存 overlay（避免啥都不输出）
    if (not opt.save_mask) and (not opt.save_overlay):
        opt.save_overlay = True
    main(opt)


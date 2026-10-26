# -*- coding: utf-8 -*-
"""SegFormer 独立评估脚本：加载权重，在 WUH val 划分上算一遍指标。

用途：训练结束后快速看单个权重的分数。四模型横向对比请用
`消融实验/eval_unified.py`（统一口径，且补齐参数量/显存/速度）。
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.segformer import SegFormer
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.train_and_eval import evaluate
from utils.utils import load_weights_partial, seed_everything

_HERE = os.path.dirname(os.path.abspath(__file__))
_ABLATION = os.path.dirname(os.path.dirname(_HERE))

CONFIG = dict(
    weights=os.path.join(_HERE, "run", "train", "exp", "weights", "best_model_2.pth"),
    data_path=os.path.join(_ABLATION, "WUH_sub25"),
    variant="b3",
    num_classes=1,          # 前景类数（不含背景）
    decoder_dim=128,
    input_shape=512,
    batch_size=8,
    workers=0,
    device="cuda",
    cls_weights="1,3",
    seed=11,
)


def parse_args():
    p = argparse.ArgumentParser(description="SegFormer 验证（WUH val 划分）")
    p.add_argument("--weights", default=CONFIG["weights"], help="要评估的权重（state_dict）")
    p.add_argument("--data-path", default=CONFIG["data_path"])
    p.add_argument("--variant", default=CONFIG["variant"], choices=["b0", "b1", "b2", "b3"])
    p.add_argument("--num-classes", default=CONFIG["num_classes"], type=int)
    p.add_argument("--decoder-dim", default=CONFIG["decoder_dim"], type=int)
    p.add_argument("--input-shape", default=CONFIG["input_shape"], type=int)
    p.add_argument("--batch-size", default=CONFIG["batch_size"], type=int)
    p.add_argument("--workers", default=CONFIG["workers"], type=int)
    p.add_argument("--device", default=CONFIG["device"])
    p.add_argument("--cls-weights", default=CONFIG["cls_weights"], type=str,
                   help='类别权重，逗号分隔（含背景），仅影响 Loss，不影响 IoU；留空则全 1')
    p.add_argument("--seed", default=CONFIG["seed"], type=int)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    num_classes = args.num_classes + 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = UnetDataset(args.data_path, [args.input_shape, args.input_shape],
                          num_classes, augmentation=False, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=unet_dataset_collate)
    print(f"验证集 {len(dataset)} 张 | 设备 {device}")

    model = SegFormer(variant=args.variant, num_classes=num_classes,
                      decoder_dim=args.decoder_dim, dropout=0.0, drop_path_rate=0.0,
                      img_size=args.input_shape)
    if not os.path.isfile(args.weights):
        raise SystemExit(f"权重不存在：{args.weights}")
    load_weights_partial(model, args.weights, map_location="cpu")
    model.eval().to(device)

    cls_weights = None
    if args.cls_weights:
        cls_weights = np.array([float(w) for w in args.cls_weights.split(",")],
                               dtype=np.float32)

    # dice / focal 与训练保持一致，只影响 Loss 数值，不影响 PA / IoU
    m = evaluate(model, loader, device, True, False, num_classes, cls_weights=cls_weights)

    print("\n===== 验证结果 =====")
    print(f"权重      : {args.weights}")
    print(f"PixelAcc  : {m['Pixel Accuracy']:.4f}")
    print(f"MeanAcc   : {m['Mean Accuracy']:.4f}")
    print(f"MeanIoU   : {m['Mean IoU']:.4f}")
    print(f"FWIoU     : {m['Frequency Weighted IoU']:.4f}")
    print(f"前景IoU   : {m['Foreground IoU']:.4f}")
    print(f"Loss      : {m['Loss']:.4f}")


if __name__ == "__main__":
    main()

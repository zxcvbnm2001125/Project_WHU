# SegFormer（消融版）— MiT 主干 + All-MLP 解码器

`XR_1` 下的第四个工程，与 U-Net / FCN / DeepLabV3+ 并列，用于 W3 消融 A 的模型对比。

本工程从 `Project_WUD/SegFormer` 复制而来，**只保留纯 SegFormer 与训练必需的部分**。
原始工程的完整功能（1/2 精修、TTA、LoveDA 支持等）仍在原目录，未做任何改动。

## 1. 相对原版删掉了什么

| 项 | 位置 | 删除理由 |
|---|---|---|
| `HighResRefine`（1/2 分辨率精修） | `model/segformer.py` | 官方 SegFormer 没有这个模块，属于后加的边界改进；开着它就不是"纯 SegFormer"基线 |
| 8 方向 TTA / 多尺度融合 | `utils/inference.py`、`evaluate()` | 推理期增强，训练与选优都不用；且会让评测口径与另外三个模型不一致 |
| 梯度累积 `accum_steps` | `train_one_epoch()` | 为 1024 全图训练加的功能，512/bs8 用不到 |
| 冻结主干 `freeze_epochs` | `train_one_epoch()` | 消融调节项，非默认路径 |
| 增强版本切换 `aug_version` v1/v3 | `utils/dataloader.py` | v1 已知会产生假标签，v3 只用于 1024 原图；本工程固定 v2 |
| 归一化切换 `norm`（imagenet/unit） | `utils/dataloader.py` | 固定 ImageNet 归一化，与 MiT 预训练一致 |
| 从 train 随机切验证集 `--val-ratio` | `train.py` | 验证集固定用官方 val 474 张，不从 train 切（避免相邻瓦片泄漏） |
| LoveDA 多类任务相关配置 | `train.py`、`tools/` | 本工程只做 WHU 二分类建筑 |
| `--amp` 占位参数 | `train.py` | 精度由 `--precision` 决定，该参数无作用 |

保留：断点续训（含 RNG 状态与逐轮历史复原）、`train_history.csv` 逐轮落盘、
按前景 IoU 选优、layer-wise lr decay、梯度裁剪、`val.py` / `predict.py`。

## 2. 目录

```
SegFormer/
  model/
    mit.py              MiT-B0~B3 主干（纯 PyTorch，不依赖 timm / mmseg）
    segformer.py        All-MLP 解码器 + 完整模型（已删 HighResRefine）
    unet_training.py    损失与学习率调度（与 U-Net 共用同一份实现）
  utils/
    dataloader.py       读 WUH 原生 <split>/image|label；增强固定 v2
    train_and_eval.py   训练一个 epoch / 评估（全局混淆矩阵）
    create_exp_folder.py、plot_results.py、utils.py
  train.py              训练入口（顶部有 CONFIG）
  val.py                单权重评估
  predict.py            单图推理（精简版：缩放 + 单次前向 + 阈值）
  weights/              mit_b1/b2/b3 ImageNet 主干权重
  run/                  训练产物（首次运行自动创建）
```

## 3. 数据与口径

直接读 `消融实验/WUH_sub25`，与另外三个工程同源同划分：

```
WUH_sub25/train/image|label   1184
WUH_sub25/val/image|label      474
```

输入 512×512、bs 8、20 轮、seed 11，按验证集前景（建筑）IoU 选优。
`train.py` 顶部的 `CONFIG` 是唯一配置入口，改完直接 F5 或 `python train.py`。

## 4. 训练配方

与三个 CNN 工程的差异是架构决定的，不是可选项：

| 项 | 值 | 原因 |
|---|---|---|
| 优化器 | AdamW | Transformer 需要解耦 weight decay |
| lr | 6e-5 | 官方分割配方；主干按 layer_decay 逐层衰减 |
| layer_decay | 0.9 | 浅层用更小的 lr，避免冲掉预训练特征 |
| wd | 0.01 | 比 CNN 常用的 1e-4 大得多 |
| 精度 | bf16 autocast | LayerNorm/GELU 在 fp16 下容易 inf；bf16 不需要 GradScaler |
| 损失 | CE + Dice，类别权重 1:3 | 与 U-Net 口径一致（注意 FCN / DeepLab 只有 CE） |

## 5. 用法

```bash
cd XR_1/SegFormer
python train.py                    # 默认 b3 / bs8 / 20 轮
python train.py --variant b1       # 换规格（需对应 weights/mit_b1_imagenet.pth）
python val.py --weights run/train/exp/weights/best_model_2.pth
python predict.py --source path/to/image.tif
```

VSCode 里 F5 选第 4 条配置即可，cwd 已锁定到本目录。

## 6. 注意

- **b3 + bs8 在 8GB 显存上可能吃紧**。原工程全量训练 b3 用的是 bs6。若 OOM，
  要么降到 bs4/b6（此时与另外三个模型的 bs8 不再等价，需在表里注明），
  要么改用 b1/b2。首次开跑先看 1 个 epoch 的显存。
- `weights/` 里没有 b0 权重。用 `--variant b0` 会退化成从随机初始化训练主干，
  日志里会给出提示；需要的话跑 `download_pretrained.py`。
- 本工程与 `eval_unified.py` 的 `segformer` 分支已对接，横向出表走那个脚本。

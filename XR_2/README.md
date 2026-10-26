# XR_2 — 消融 B（CE vs CE+Dice）

本目录是 `XR_1` 的副本，用于 W3 消融 B：**在同一模型上比较纯 CE 与 CE+Dice 两种损失**。

代码与 `XR_1` 同源，结构完全一致（四个工程并列），唯一的功能性差异是新增了一个
统一的 `dice_loss` 开关。`XR_1` 里那批权重、`run/` 历史、`VOCdevkit` 等未复制过来。

```
XR_2/
  U-Net/          ← 已有 CE+Dice（上游默认），本次只补开关与配置落盘
  FCN/            ← 原本只有 CE，本次新实现 Dice
  DeepLabV3Plus/  ← 原本只有 CE / Focal，本次新实现 Dice
  SegFormer/      ← 本来就有 dice 开关，本次只补日志
```

## 1. 两臂怎么跑

四个工程的开关名字统一，默认值保持各自上游原状（`U-Net`/`SegFormer` 默认开，`FCN`/`DeepLab` 默认关）。
**正式跑请显式指定，不要依赖默认值**，否则容易把两臂搞混。

| 工程 | 纯 CE 臂 | CE+Dice 臂 |
|---|---|---|
| U-Net | `python train.py --no-dice_loss` | `python train.py --dice_loss` |
| FCN | `python train.py --no-dice_loss` | `python train.py --dice_loss` |
| DeepLabV3+ | `python train.py --no-dice_loss` | `python train.py --dice_loss` |
| SegFormer | `python train.py --no-dice` | `python train.py --dice` |

也可以直接改各自 `train.py` 顶部 `CONFIG` 里的 `dice_loss`（SegFormer 叫 `dice`）。

启动时控制台会打一行 `损失函数: CE` 或 `损失函数: CE+Dice`，便于当场确认。

## 2. 两次运行怎么区分

每个 exp 目录下都写了一份 `train_config.txt`（本次为四个工程统一补上）：

```
loss_mode: CE+Dice
dice_loss: True
batch_size: 8
epochs: 25
...
```

`run/train/exp`、`exp1`、`exp2` 是按创建顺序递增的，光看目录名分不清哪次是哪个臂，
**以 `train_config.txt` 的 `loss_mode` 为准**。跑完一臂建议直接把目录改名（如 `exp_ce`、`exp_cedice`）。

## 3. 改了什么

| 工程 | 文件 | 改动 |
|---|---|---|
| U-Net | `train.py` | 新增 `--dice_loss/--no-dice_loss`；写入 `train_config.txt`；打印损失模式 |
| FCN | `utils/train_and_eval.py` | 新增 `to_onehot()` / `dice_loss()`，`criterion()` 增加 `use_dice` 分支 |
| FCN | `train.py` | `CONFIG` 增加 `dice_loss`、新增命令行开关、`train_config.txt`、透传给训练与评估 |
| DeepLabV3+ | `utils/loss.py` | 新增 `DiceLoss`、`CombinedLoss` |
| DeepLabV3+ | `utils/engine.py` | `build_criterion()` 读取 `opts.dice_loss` |
| DeepLabV3+ | `train.py` | `CONFIG` 增加 `dice_loss`、`--dice_loss` 开关、`train_config.txt` |
| DeepLabV3+ | `train.py` / `utils/engine.py` | 顺带修掉 CPU 下 `nn.DataParallel` 与 `model.module` 的两处崩溃 |
| SegFormer | `train.py` | 打印损失模式（`dice` 开关原本就有） |

Dice 的实现四个工程同一套公式：softmax 概率上的逐类软 Dice，取各类均值，`smooth=1e-5`；
**忽略像素（如 FCN 补边的 255）既不进分子也不进分母**。组合权重固定 CE:Dice = 1:1。

## 4. 判读要点

消融 B 的价值不在整体 mIoU，而在两处：

1. **前景（建筑）IoU**：Dice 直接优化重叠度，通常对前景类更友好；CE 在背景占多数时容易被背景主导。
2. **边界与小目标的召回**：Dice 对"整体偏小"很敏感，常见现象是前景 IoU 上升但**误检（FP）增加**，
   表现为建筑粘连变多、边界外扩。要配合 PR 曲线或至少看一眼 FN/FP 的 IoU 分解，不能只看一个数。

建议在表里同时给出：前景 IoU、背景 IoU、mIoU、PA，并附一张典型失败案例（Dice 臂的过分割）。
如果 Dice 臂只是把前景 IoU 换来了更多 FP，这个"提升"在工程交付上是负收益。

## 5. 注意

- 两臂必须同数据（1184/474）、同轮数、同种子、同超参，只改损失。
- `XR_1` 的既有结果不要拿来当 CE 臂：那次 U-Net 是 CE+Dice、FCN 是纯 CE、DeepLab 是纯 CE、SegFormer 是 CE+Dice，
  口径混杂，不能拼成消融 B 的基线。两臂都在本目录重跑。
- DeepLab 当前 `CONFIG` 是 `deeplabv3plus_xception16` + `batch_size=6`，与其他三个模型的 bs8 不等价，
  出表时需注明；要统一就改 `model`/`batch_size` 并确认显存。

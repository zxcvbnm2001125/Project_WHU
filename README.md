# 建筑提取基线：从切片训练到工程交付

> 四结构消融 → 损失函数配对实验 → ONNX 量化部署 → 边界误差归因。README 只写结论和踩过的坑，过程数据见各目录报告。
>
> **本仓库只含代码与报告**：权重（5.06 GB / 28 个文件）和 WHU 数据集不入库，clone 下来不能直接推理，见 [前置条件](#前置条件数据与权重)。

**目录**：[结果一览](#结果一览) · [结构](#结构) · [前置条件](#前置条件数据与权重) · [快速开始](#快速开始) · [三个工程细节](#三个工程细节) · [算力与口径](#算力与口径的诚实说明)

---

## 结果一览

### 消融 A：四种结构（同一数据/轮数/种子）

| 模型 | 参数量 | 前景 IoU | mIoU | 延迟(bs1) | 吞吐(bs8) | 备注 |
|---|---|---|---|---|---|---|
| **U-Net (ResNet50+scSE)** | 43.93 M | **0.8802** | **0.9322** | 28.8 ms | 32.1 img/s | 精度最好 |
| SegFormer-B1 | **13.68 M** | 0.8727 | 0.9278 | **9.4 ms** | **96.8 img/s** | 性价比最好 |
| SegFormer-B2 | 24.72 M | 0.8758 | 0.9295 | 15.8 ms | 60.6 img/s | — |
| DeepLabV3+ (ResNet101) | 58.75 M | 0.7220 | 0.8415 | 20.1 ms | 38.4 img/s | 见下方说明 |
| DeepLabV3+ (ResNet50) | 39.76 M | 0.7087 | 0.8333 | **14.5 ms** | 53.4 img/s | — |
| FCN (ResNet50) | 32.95 M | 0.6583 | 0.8051 | 32.9 ms | 26.7 img/s | 无 skip，边界糊 |

- 口径：WHU 航拍子集 train 1184 / val 474 / test 604（25% 分层抽样，seed 11），**25 轮**，512 输入，batch 8，全局混淆矩阵。
- **DeepLabV3+ 全线落后不是结构问题**：lr 1e-3 偏大 + OS16 + 25 轮只跑 3700 iter（PolyLR 随总步数退火）+ VOC 21 类分类头重初始化。同主干换成 3e-4 / 80 轮 / OS8 通常会追到 0.86 附近，但算力预算不允许（本项目算力硬约束见下）。

### 消融 B：CE vs CE+Dice（8 对配对实验）

| 模型 | 基线 FG IoU | +Dice | 增益 |
|---|---|---|---|
| FCN | 0.4034 | 0.4666 | **+0.0632** |
| DeepLabV3+ Xception | 0.6844 | 0.7414 | +0.0570 |
| DeepLabV3+ R50 | 0.7087 | 0.7611 | +0.0524 |
| DeepLabV3+ R101 | 0.7220 | 0.7734 | +0.0514 |
| SegFormer-B1 | 0.8609 | 0.8727 | +0.0118 |
| SegFormer-B2 | 0.8646 | 0.8758 | +0.0112 |
| U-Net | 0.8510 | 0.8538 | +0.0028 |
| SegFormer-B3 | 0.8711 | 0.8710 | −0.0001 |

**核心结论：Dice 的增益与基线强度单调负相关。** 基线越弱（边界糊、无 skip 的 FCN），Dice 帮忙越大；基线已经练到 0.87 时，Dice 的梯度信号接近饱和甚至反向。所以"加 Dice 一定更好"是错的，"先知道你的基线在哪"才是对的。

### 部署侧（ONNX，val 474 平均）

| | PyTorch FP32 | PyTorch FP16 | ORT-GPU FP32 | **ORT-GPU FP16** |
|---|---|---|---|---|
| U-Net | 29.06 ms / 34.4 img/s | 17.11 / 58.4 | 14.71 / 68.0 | **9.56 / 104.7** |
| SegFormer-B1 | 25.57 / 39.1 | 22.99 / 43.5 | 22.65 / 44.2 | **16.75 / 59.7** |

导出物：U-Net 167.52 MB(FP32) → 83.76 MB(FP16)；SegFormer-B1 52.03 → 26.02 MB。
**验收标准**：输出 shape 一致 → argmax 逐像素一致（实测 0 差异）→ 固定样本预测完全一致 → 指标无下降（IoU 一致到最后一位）。

### test 604 张的错误分析

- 全局 **前景 IoU 0.8603 / mIoU 0.9211 / P 0.9240 / R 0.9258**；逐图中位 0.8686。
- **误差预算：69.2% 的错像素集中在 GT 边界 ±3 px 的窄带里**（内缩 36.9% + 外扩 32.3%），真正"整栋没找到"只占 10.9%。
- 面积总量几乎无偏：`pred/gt = 1.00193`（+0.19%）；但边界 F1 只有 **0.6027**。
- 最弱场景是**稀疏**（建筑占比 0–2% 的图 IoU 0.6596）而不是密集（0.8691）；暗影像比亮影像低约 10 个点。
- 详细报告见 [`W5/错误分析报告.md`](W5/错误分析报告.md)。

---

## 结构

```
.
├── WUH_sub25/            ✗不入库 # 分层抽样后的数据子集（硬链接，不占额外空间）
│   ├── train/{image,label}     # 1184 张
│   ├── val/{image,label}       # 474 张（空图 137 张，28.9%）
│   ├── test/{image,label}      # 604 张
│   └── manifest/*.csv          # 每张的建筑面积占比 / 分层 / 源路径
├── XR_1/                       # 消融 A：四模型原样训练 + 统一评测脚本
│   ├── eval_unified.py         # 训练代码零改动，对 best 权重用全局混淆矩阵重算
│   └── sliding_window.py       # 大图滑窗推理（25% 重叠 + 加权融合 + 接缝量化）
├── XR_2/                       # 消融 B：同一份代码仅切 dice_loss 开关
├── W4/                         # 工程闭环
│   ├── U-Net/
│   │   ├── predict.py          # 单图/批量推理（薄壳，默认参数在文件末尾 MANUAL）
│   │   ├── batch_infer.py      # 批量推理 + 断点续跑 + 失败清单
│   │   ├── to_geotiff.py       # 仿射参数 → GeoTIFF（支持瓦片目录 / 拼幅）
│   │   ├── export_onnx.py      # ONNX 导出 + 四层校验 + 四方延迟对比
│   │   └── utils/{batch_runner,quality_check,onnx_export}.py
│   └── SegFormer/              # 同上（对照模型）
├── W5/                         # 错误分析
│   ├── error_analysis.py       # 误差预算 / 距离分带 / 形态归因 / 案例图
│   ├── 错误分析报告.md
│   └── out/                    # stats.json / error_images.csv / cases/*.png
└── results/                    # 消融报告、滑窗报告
```

`✗不入库`：`.gitignore` 排除了 `WUH_sub25/`（数据集）与所有 `*.pth` / `*.onnx`（权重与导出件，合计 5.06 GB）。`XR_1` / `XR_2` / `W4` 看着高度重复，是因为**消融的前提是"训练代码零改动"**——抽公共库就等于改上游代码，"同一份代码只切一个开关"这个前提也就没了。

**设计取向：所有脚本的参数都集中在文件末尾的 `MANUAL` 字典里**，命令行参数可覆盖。理由很实际——交付给业务方时，他们改的是"数据在哪、权重在哪"，不是背 CLI 参数。

---

## 前置条件：数据与权重

clone 下来**不能直接推理**，缺的三样东西都不是代码能补的：

| 资产 | 期望位置 | 为什么不在仓库 |
|---|---|---|
| WHU 子集 1184 / 474 / 604 | `WUH_sub25/{split}/{image,label}` | [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html) 仅授权学术用途、不允许再分发。抽样口径 seed 11 / 25% 分层，`manifest/*.csv` 记录每张的面积占比与来源 |
| ImageNet 预训练主干 | `{XR_1,XR_2,W4}/SegFormer/weights/mit_bN_imagenet.pth` | 可由 HuggingFace `nvidia/mit-bN` 转换，转换脚本 `download_pretrained.py` 与该目录 `README.txt` 都在 |
| 训练产物 `best_*.pth` | `*/run/train/exp*/weights/` | 单个 168–449 MB，超 GitHub 单文件 100 MB 上限（8 个 DeepLab 最佳权重合计约 3 GB） |

**已知的可复现性缺口**：`WUH_sub25` 的抽样脚本没有归档，仓库里只有抽样口径（seed 11 / 25% 分层）和各报告里的逐图 CSV。要严格复现 1184/474/604 这个划分，得按同样的分层规则重写一次抽样。写在这里，免得下一个人以为是自己的环境问题。

---

## 快速开始

```powershell
# 0) 环境：Python 3.10 + torch(CUDA)；其余依赖见根目录 requirements.txt
pip install -r requirements.txt

# 1) 单图/批量推理（结果落 run/predict/exp*）
cd W4\U-Net
python predict.py --data_path D:\data\my_images --weights run\train\exp\weights\best_model_1.pth

# 2) 大批量推理（断点续跑：中断后重跑同一条命令即可）
python batch_infer.py --input D:\data\my_images --out D:\out\batch01

# 3) 结果质检（7 类检查 + 与 GT 对照）
python -m utils.quality_check --out D:\out\batch01 --gt-dir D:\data\my_labels

# 4) 写回 GeoTIFF
python to_geotiff.py --input D:\data\my_images --out D:\out\geotiff

# 5) ONNX 导出 + 四方对比
python export_onnx.py

# 6) 错误分析（--pred 指向第 2 步的批量推理输出；--pred2 是对照模型，可省）
cd ..\..\W5
python error_analysis.py --pred D:\out\batch01 --pred2 D:\out\batch01_seg --max-cases 20
```

---

## 三个工程细节

**1. 标签是 PIL `mode='1'`（bool 数组），不是灰度图。**
直接 `Image.convert('L')` 会得到 0/255，配 `CrossEntropyLoss(ignore_index=255)` 时**前景会被整片当成忽略区**，训练信号归零且不报错。统一入口 `load_mask()` 转成 uint8 的 {0,1}。

**2. 指标口径能差出 0.25 个 IoU。**
同一个 U-Net 权重、同一份 val 474，三种算法给出三个数：逐批平均 0.8538 / 逐图平均 0.5799 / 全局混淆矩阵 0.8802。原因是 137 张空图（28.9%）在"逐图平均"里逐张记 0 分。**所以本仓库所有对比一律用全局混淆矩阵**，并且把三种口径的对照表写进报告。

**3. 断点续跑的判据是"结果完整"，不是"文件存在"。**
进程被 kill 时 `*_mask.png` 可能只写了半个文件。所以 `batch_infer.py` 的完成判据是：mask 存在 **且** 能被 PIL 完整解码 **且** 尺寸匹配；写文件用 `临时文件 + os.replace` 原子落盘；manifest 只追加，重跑时按内容去重。

---

## 算力与口径的诚实说明

- 消融实验一律用 25% 子集（1184/474/604）+ 固定 25 轮 + seed 11**，只比相对优劣，**绝对指标不能外推到全量**。
- 三个配对的 batch size 不完全一致（DeepLabV3+ 的 Dice 臂是 bs6/bs4，CE 臂是 bs8；SegFormer-B3 一侧只有 23 轮）。已在报告里逐条标注，并用 DeepLab 家族里唯一的同 bs 对照（Xception，bs4 vs bs4）作为旁证：干净对照的增益 +0.0570 **不小于**脏对照的 +0.0524/+0.0514，说明 bs 差异没有虚增结论。
- README 早期版本写"20 轮"，实际训练全部是 25 轮，已在报告口径表中更正。

---

## License / 数据来源

代码 MIT。数据集为 [WHU Building Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html)（学术用途），不在本仓库内分发。

## 参考

- Ronneberger et al., U-Net (2015)
- Chen et al., DeepLabV3+ (2018)
- Xie et al., SegFormer (2021)
- Mnih, Machine Learning for Aerial Image Labeling (2013) — boundary F1 的来源

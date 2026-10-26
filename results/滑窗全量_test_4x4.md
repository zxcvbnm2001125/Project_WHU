# 滑窗拼接全量验证（test 划分，4×4 块）

- 权重：`XR_1/U-Net/run/train/exp/weights/best_model_1.pth`（`matched=340 skipped=0 missing=0`）
- 数据：`WUH_sub25/test`，1702 张 512×512，行宽 23（74 行 × 23 列），拼成 23×74 网格
- 块：4×4 瓦片 = 1024×1024 px，共 **90 块**，覆盖 1440/1702 张
- 参数：`--overlap 0.25 --blend hann --seam-half 2`
- 耗时：**226.7 s**（单块 1.5–1.6 s）
- 输出：`_tools/sliding_batch/blocks/test/<r>_<c>/`（`triplet.png` `seam_zoom.png` `metrics.json`），汇总 `_tools/sliding_batch/blocks.csv`

## 汇总（90 块均值 ± 标准差）

| 指标 | A 直接切片拼接 | B 滑窗 + 25% 重叠 + Hann 融合 | 变化 |
|---|---|---|---|
| 前景 IoU（全图） | 0.8197 ± 0.1546 | 0.8317 ± 0.1354 | **+0.0120** |
| 前景 IoU（接缝带 ±2px） | 0.7057 ± 0.1846 | 0.8459 ± 0.1372 | **+0.1402**（配对增益 +0.1152） |
| 前景 IoU（非接缝区） | 0.8210 ± 0.1542 | 0.8317 ± 0.1352 | +0.0107 |
| 跨缝跳变率（1−SSIM） | 0.0600 ± 0.0394 | 0.0080 ± 0.0061 | **降至 13.7%** |
| 逐像素一致率 A/B | — | 0.9940 ± 0.0037 | — |

- **82/85 块**接缝带 IoU 被改善（96.5%），最差 −0.0166，最好 +0.3434；5 块因 GT 前景≈0（全空图）指标为 nan，已排除。
- 结论：换块大小只影响"每块里有多少条接缝"，不影响结论方向——接缝带是唯一的性能洼地，滑窗把它抬到与非接缝区持平（0.8459 vs 0.8317）。

## 复现

```powershell
Set-Location -LiteralPath 'D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\miniconda3\envs\unet\python.exe' XR_1\sliding_window.py `
    --mode batch --split test --block 4 --out _tools\sliding_batch
```

换成 2×2（407 块，每块接缝更密，约 10–12 min）：

```powershell
& 'D:\miniconda3\envs\unet\python.exe' XR_1\sliding_window.py `
    --mode batch --split test --block 2 --out _tools\sliding_batch_2x2
```

## 已知局限

1. 块内 GT 是整幅，但每块只覆盖 16/1702 张瓦片；块与块之间不重叠（`step = 4` 而不是 `2`），所以同一张瓦片只出现在一个块里。
2. `pred_fg_ratio` 列已加入 CSV——用于兜底发现"模型塌缩成全判背景"这类故障（v1 就是因为权重结构不匹配，90 块指标全 0 而脚本不报错）。

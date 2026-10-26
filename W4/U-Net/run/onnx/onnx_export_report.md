# ONNX 导出报告 — unet_resnet50

## 一、结论
- 导出**成功**：PyTorch 与 ONNX Runtime 的 argmax 掩膜 **逐像素完全一致（12 张，3145728 像素）**。
- 与原推理管线（`adapter.prep → infer → post`）的差异：**0 像素**。
- 烘进图里的归一化：`x / 255`，与训练期一致，部署端**不需要**再做归一化。
- 指标对齐：torch 前景 IoU 0.7676 vs ORT 0.7676（8 张有真值）。
- 校验四层：结构 / 形状 / argmax 数值 / 与原管线口径 —— 全部通过。

## 二、产物
| 项目 | 值 |
|---|---|
| 权重 | `d:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\W4\U-Net\run\train\exp\weights\best_model_1.pth` |
| ONNX | `d:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\W4\U-Net\run\onnx\unet_resnet50.onnx` |
| 文件大小 | 167.52 MB |
| opset / ir_version | [17] / 8 |
| 节点数 | 156 |
| 导出器 | torchscript |
| ORT 执行后端 | **CUDA**（onnxruntime 1.23.2）|
| 动态维 | batch=True, H/W=False |

## 三、输入输出契约
```
input  'input'  float32 ['batch', 3, 512, 512]   # 原始像素 0~255，未归一化
output 'logits' float32 ['batch', 2, 512, 512]
掩膜 = np.argmax(logits, axis=1).astype(np.uint8)   # 0 背景 / 1 建筑
```

## 四、延迟与吞吐
| 后端 | batch | 中位延迟 ms | P90 ms | 吞吐 img/s |
|---|---|---|---|---|
| torch_fp32_gpu_bs1 | 1 | 25.52 | 25.96 | 39.19 |
| torch_fp32_gpu_bs8 | 8 | 216.35 | 216.63 | 36.98 |
| torch_fp16_gpu_bs1 | 1 | 16.12 | 16.50 | 62.04 |
| torch_fp16_gpu_bs8 | 8 | 116.70 | 117.23 | 68.55 |
| ort_fp32_cuda_bs1 | 1 | 26.54 | 27.13 | 37.68 |
| ort_fp32_cuda_bs8 | 8 | 212.74 | 212.98 | 37.60 |
| ort_fp16_cuda_bs1 | 1 | 15.13 | 22.17 | 66.08 |
| ort_fp16_cuda_bs8 | 8 | 119.36 | 119.98 | 67.02 |

> ORT 走的是 **CUDAExecutionProvider**（onnxruntime 1.23.2），与 torch 同一张 GPU，
> 各后端数字可以直接横向比。
> FP16 行用的是同一张图转成的 `unet_resnet50_fp16.onnx`（转换器 onnxruntime.transformers，IO 保持 FP32），
> 与 FP32 的 argmax 差异 **0 像素**、logits 最大差 3.158e-02。

## 五、逐样本校验
| 文件 | argmax 差异像素 | 与原管线差异 | ORT 前景占比 |
|---|---|---|---|
| val_0.tif | 0 | 0 | 0.0000 |
| val_10.tif | 0 | 0 | 0.0000 |
| val_100.tif | 0 | 0 | 0.0000 |
| val_1001.tif | 0 | 0 | 0.0368 |
| val_1005.tif | 0 | 0 | 0.0051 |
| val_1006.tif | 0 | 0 | 0.1411 |
| val_1011.tif | 0 | 0 | 0.0089 |
| val_1012.tif | 0 | 0 | 0.0000 |
| val_1013.tif | 0 | 0 | 0.0701 |
| val_1014.tif | 0 | 0 | 0.0162 |
| val_1015.tif | 0 | 0 | 0.2385 |
| val_1018.tif | 0 | 0 | 0.0000 |

## 六、已知限制
1. 动态 H/W 默认关闭：开启动态尺寸后部分算子（如插值）在某些 EP 上会退化，
   需要的话用 `--dynamic-hw` 单独验证。
2. 输入尺寸要与训练一致（512，且 32 的整数倍），尺寸不符会让精度明显下降。
3. ONNX 图里返回的是 logits，不是概率；要概率自己加 softmax/sigmoid，
   但取掩膜用 argmax 就够，省这一步。
4. FP16 图要 GPU 上的 CUDA/TensorRT EP 才有意义；CPU EP 上 FP16 反而更慢
   （要插 Cast 节点做浮点 16/32 转换）。

## 七、复现
```bash
python export_onnx.py --weights <权重> --input <图片目录> --out <输出目录>
# 想在 GPU 上测 ORT：先用 pip 装 onnxruntime-gpu（或解压 wheel 到某目录）
python export_onnx.py --ort-libs <onnxruntime-gpu 所在目录> --ort-fp16
```

环境：torch 2.14.0+cu126 / cuda / GPU NVIDIA GeForce RTX 4060 Laptop GPU

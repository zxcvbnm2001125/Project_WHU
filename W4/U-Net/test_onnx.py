# -*- coding: utf-8 -*-
"""ONNX 最小推理示例 —— U-Net / unet_resnet50.onnx

为什么要开头那几行：
  onnxruntime-gpu 里的 onnxruntime_providers_cuda.dll 只是一个"薄壳"，
  它运行时还要去加载 cublasLt64_12.dll、cublas64_12.dll、cudnn64_9.dll、
  cudart64_12.dll 这些 CUDA 运行库。
  PyTorch 的 wheel 自带了一整套这些库（在 torch\\lib 下），但 Windows 的 DLL
  搜索路径里没有 torch\\lib，于是报：

      Error loading "onnxruntime_providers_cuda.dll" which depends on
      "cublasLt64_12.dll"  ...  (Error 126: 找不到指定的模块)

  Error 126 = ERROR_MOD_NOT_FOUND，意思是"依赖的 DLL 找不到"，
  不是显卡不行、也不是驱动不行。
  Python 3.8 以后，光改 PATH 对 DLL 搜索不可靠，必须用
  os.add_dll_directory() 显式把目录登记进搜索列表 —— 这就是下面 3 行在干的事。

用法：
    cd W4\\U-Net
    D:\\miniconda3\\envs\\unet\\python.exe test_onnx.py
"""
import os
import time

import numpy as np
from PIL import Image

# --- 1) 把 PyTorch 自带的 CUDA 运行库目录登记进 DLL 搜索路径 ---------------
import torch  # 先 import torch，才能拿到它自带的库目录

_TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.isdir(_TORCH_LIB):
    os.add_dll_directory(_TORCH_LIB)
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")

# --- 2) 必须放在上面之后，否则 CUDA EP 加载不起来 --------------------------
import onnxruntime as ort

# --- 3) 可改的参数（也可以命令行覆盖，见文件末尾说明） --------------------
ONNX_PATH = r"run\onnx\unet_resnet50.onnx"
IMG_PATH = r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25\test\image\2_33.tif"
INPUT_NAME = "input"          # 报告的 §三 里写着输入名是 input
N_WARMUP = 3                  # 预热次数（第一次跑会分配显存，不计入耗时）
N_RUN = 10                    # 正式计时次数


def _resolve(p: str) -> str:
    """相对路径先按当前目录找，找不到就按脚本所在目录找。"""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), p)
    return alt if os.path.exists(alt) else p


def main():
    onnx_path = _resolve(ONNX_PATH)
    img_path = _resolve(IMG_PATH)
    if not os.path.exists(onnx_path):
        raise SystemExit("找不到 ONNX 文件：" + onnx_path)
    if not os.path.exists(img_path):
        raise SystemExit("找不到测试影像：" + img_path)

    sess = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print("onnxruntime", ort.__version__)
    print("可用后端  :", ort.get_available_providers())
    print("实际后端  :", sess.get_providers()[0], "  <-- 这里应当是 CUDAExecutionProvider")
    print("输入约定  :", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
    print("输出约定  :", [(o.name, o.type, o.shape) for o in sess.get_outputs()])

    img = np.asarray(Image.open(img_path).convert("RGB"), np.float32)
    print("影像      :", os.path.basename(img_path), img.shape, "取值范围", f"{img.min():.0f}~{img.max():.0f}")
    if img.shape[:2] != (512, 512):
        raise SystemExit("影像不是 512×512，ONNX 图只认这个尺寸")

    # 关键：保持 0~255。归一化已经烘进 ONNX 图里了，这里再除 255 就是错的。
    x = img.transpose(2, 0, 1)[None]          # (1, 3, 512, 512)
    print("输入张量  :", x.shape, x.dtype, "取值范围", f"{x.min():.0f}~{x.max():.0f}")

    for _ in range(N_WARMUP):
        sess.run(None, {INPUT_NAME: x})

    ts = []
    for _ in range(N_RUN):
        t0 = time.perf_counter()
        logits = sess.run(None, {INPUT_NAME: x})[0]
        ts.append((time.perf_counter() - t0) * 1000.0)

    print("输出 logits:", logits.shape, logits.dtype,
          "取值范围", f"{logits.min():.3f}~{logits.max():.3f}")
    mask = logits.argmax(1)[0].astype(np.uint8)   # (512, 512)，0 背景 / 1 建筑
    fg = float(mask.mean())
    print("掩膜      :", mask.shape, mask.dtype, "唯一值", np.unique(mask).tolist())
    print("前景占比  : %.4f%%" % (fg * 100.0))
    ts.sort()
    print("耗时      : 中位 %.2f ms  最小 %.2f ms  最大 %.2f ms  (bs1, %s)" %
          (ts[len(ts) // 2], ts[0], ts[-1], sess.get_providers()[0]))
    print("\n对照：报告 §四 里 torch_fp32 的 bs1 延迟约 25~29 ms，"
          "ort_fp32 约 26~27 ms —— 同一量级就说明跑对了。")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 想换模型/换图，改上面 ONNX_PATH、IMG_PATH 两行即可。
# 想确认"换尺寸会报错"（理解 ONNX 的固定形状层），把 IMG_PATH 换成任意非 512 的图，
# 会在 sess.run 处直接抛 shape 不匹配 —— 这正是它挡住错误输入的方式。
# 想长期免去开头那 3 行注册：把 "%CONDA_PREFIX%\Lib\site-packages\torch\lib"
# （即 _TORCH_LIB）加进系统环境变量 PATH，然后重开终端。
# ---------------------------------------------------------------------------

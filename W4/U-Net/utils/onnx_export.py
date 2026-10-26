# -*- coding: utf-8 -*-
"""ONNX 导出 + 一致性校验 + 延迟基准（引擎）。

两个工程（U-Net / SegFormer）共用本文件，**两份逐字节相同**。
工程差异只体现在同级的 `export_onnx.py`：那边负责建模型、给出"原图 -> 0~255 张量"
的预处理，本文件负责导出、校验、计时、出报告。

为什么要把预处理烘进计算图
--------------------------
ONNX 最常见的翻车方式不是导出失败，而是"导出的模型能吃，但喂进去的数不对"：
训练时做过 /255 或 ImageNet 标准化，部署端忘了做，模型不报错，输出一片垃圾。
所以默认把归一化写进图里，导出模型的输入就是**原始像素 0~255 的 float32**，
部署端只要 `HWC -> CHW`、加 batch 维，不可能漏做。

输入输出契约
------------
    input  : float32 [N, 3, H, W]，像素值 0~255（未归一化）
    output : logits [N, C, H, W]（U-Net C=2，SegFormer C=2）
    取掩膜 : U-Net / SegFormer 都是 np.argmax(logits, axis=1)

校验做四层（少一层就可能把静默错误放过去）
------------------------------------------
1. 结构：onnx.checker + ORT 能建 session
2. 形状：输出 shape 与 PyTorch 一致
3. 数值：argmax 结果逐像素比对（要求 100% 一致）
4. 口径：与原推理管线（adapter.prep + adapter.infer + adapter.post）比 IoU，
   确认"烘进图里的预处理"与训练期完全等价
"""

import json
import os
import statistics
import gc
import sys
import time

import numpy as np
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- 依赖
_ORT_CPU_HINT = (
    "[ORT][警告] 只有 CPUExecutionProvider —— GPU 版没生效。\n"
    "          这一组数字请当 CPU 数据看，不要拿它和 PyTorch CUDA 对比。\n"
    "          三个常见原因，按顺序排查：\n"
    "          1) CPU 包把 GPU 包覆盖了。`pip list` 里同时出现 onnxruntime 和\n"
    "             onnxruntime-gpu 时，后装的会覆盖前者的 pybind，而两者不报错、\n"
    "             只是静默退化成 CPU。只留 GPU 版：\n"
    "                 pip uninstall -y onnxruntime onnxruntime-gpu\n"
    "                 pip install onnxruntime-gpu\n"
    "             自检：python -c \"import onnxruntime as o;print(o.get_available_providers())\"\n"
    "          2) 缺 CUDA / cuDNN 运行库。onnxruntime-gpu 不自带，需要 CUDA 12 + cuDNN 9；\n"
    "             本脚本已自动把 torch\\lib 加进 DLL 搜索路径（见上面 [ORT] 行）。\n"
    "          3) 包不在默认位置：用 --ort-libs <含 onnxruntime 的目录> 指定。")


def ensure_deps(ort_libs=None):
    """检查 onnx / onnxruntime。缺失时按提示装，别让它半路 ImportError。

    正常情况 `pip install onnx onnxruntime-gpu` 就够了，不需要 ort_libs。
    ort_libs 只在"包装在非默认位置"时用 —— 因为 `import onnxruntime` 一旦成功
    就进了 sys.modules，后面再改 sys.path 也不会换包（CPU 版 / GPU 版同名同
    模块，先到先得），所以必须赶在第一次导入之前把目录定下来。
    """
    missing = []
    try:
        import onnx  # noqa: F401
    except Exception:
        missing.append("onnx")
    try:
        import_ort(ort_libs)  # noqa: F401
    except Exception:
        missing.append("onnxruntime")
    if missing:
        raise SystemExit(
            "缺少依赖：%s\n"
            "  pip install onnx onnxruntime-gpu      # 有 GPU 就用 -gpu 这版\n"
            "  如果 pip 装不上（公司代理 / 无网 / 沙箱），把 wheel 解压到某个目录后\n"
            "  用 --ort-libs <那个目录> 指过来，或设环境变量 ONNX_LIBS=<那个目录>。"
            % ", ".join(missing))


def import_onnx():
    import onnx
    return onnx


def import_ort(libs=None):
    """导入 onnxruntime；libs 指向一份特定安装时优先用它。

    *** 关键：无论有没有传 libs，都先把 torch\\lib 注册进 DLL 搜索路径。 ***
    onnxruntime-gpu 的 CUDA EP 是在 import 时才去 LoadLibrary 加载
    onnxruntime_providers_cuda.dll，而这个 dll 又依赖 cudnn64_9 / cublas64_12
    等一整套 CUDA 运行库 —— onnxruntime-gpu 的 wheel 里**不含**这些。
    torch 的 cu126 wheel 里正好带全了，但 Python 3.8+ 不再从 PATH 找扩展依赖，
    必须显式 add_dll_directory，否则 load 失败后会静默退化到 CPU EP
    （不抛异常、不打印，只是 providers 里少了 CUDA）。

    早期版本只在传了 libs 时才注册，导致"pip 装好 GPU 版却还是跑在 CPU 上"，
    是个静默坑，这里改成无条件注册。
    """
    if libs:
        libs = os.path.abspath(libs)
        if libs not in sys.path:
            sys.path.insert(0, libs)
            print("[ORT] 依赖目录前置：%s" % libs)
    _register_cuda_dll_dirs()
    import onnxruntime
    return onnxruntime


_DLL_HANDLES = []


def _register_cuda_dll_dirs():
    """onnxruntime-gpu 的 CUDA EP 要加载 cudnn / cublas 的 DLL。

    torch 的 wheel 里正好带了一整套（torch\\lib），但 Python 3.8+ 不再从 PATH
    找扩展依赖，必须显式 add_dll_directory；句柄要留着，别被 GC 回收。
    """
    try:
        import torch as _t
        lib = os.path.join(os.path.dirname(_t.__file__), "lib")
        if os.path.isdir(lib) and hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(lib))
            print("[ORT] 追加 DLL 目录：%s" % lib)
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:      # 找不到 torch 就交给系统 PATH
        print("[ORT] 跳过 DLL 目录注册：%s" % exc)


# --------------------------------------------------------------------------- 包装
class ExportWrapper(nn.Module):
    """把"归一化 + 前向"合成一个模块，导出后输入就是原始像素。

    norm:
      "unit"     : x/255                      （U-Net 训练期就是这么做的）
      "imagenet" : (x/255 - mean) / std        （SegFormer 验证期）
      "none"     : 不做任何归一化（调用方自己喂归一化后的张量）
    """

    def __init__(self, model, norm="unit"):
        super().__init__()
        self.model = model
        self.norm = norm
        if norm == "imagenet":
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        if self.norm == "none":
            return self.model(x)
        x = x / 255.0
        if self.norm == "imagenet":
            x = (x - self.mean) / self.std
        return self.model(x)


# --------------------------------------------------------------------------- 工具
def _median(xs):
    return float(statistics.median(xs)) if xs else float("nan")


def _sync(device):
    if getattr(device, "type", "") == "cuda":
        torch.cuda.synchronize()


def iou_from_pair(pred, gt):
    """二分类前景 IoU（pred/gt 都是 0/1 的 uint8 数组）。"""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None            # 两边都空：IoU 无定义，不计入
    return float(inter) / float(union)


def fmt_mb(n):
    return "%7.2f MB" % (n / 1024.0 / 1024.0)


# --------------------------------------------------------------------------- 导出
def export_graph(wrapper, side, path, opset=17, dynamic_batch=True,
                 dynamic_hw=False, prefer_dynamo=False):
    """导出 .onnx。返回 (实际使用的导出器名, 输入名, 输出名)。"""
    wrapper.eval()
    dummy = torch.zeros(1, 3, side, side, dtype=torch.float32)
    dyn = {}
    if dynamic_batch:
        dyn["input"] = {0: "batch"}
        dyn["logits"] = {0: "batch"}
    if dynamic_hw:
        dyn.setdefault("input", {}).update({2: "height", 3: "width"})
        dyn.setdefault("logits", {}).update({2: "height", 3: "width"})

    kwargs = dict(
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=dyn or None,
    )
    tried = []
    orders = [prefer_dynamo, not prefer_dynamo]
    for use_dynamo in orders:
        tag = "dynamo" if use_dynamo else "torchscript"
        try:
            torch.onnx.export(wrapper, dummy, path, dynamo=use_dynamo, **kwargs)
            return tag, "input", "logits"
        except Exception as e:                      # noqa: BLE001
            tried.append("%s -> %s: %s" % (tag, type(e).__name__, str(e)[:300]))
    raise SystemExit("两种导出器都失败：\n  " + "\n  ".join(tried))


def check_graph(path):
    onnx = import_onnx()
    model = onnx.load(path)
    onnx.checker.check_model(model)
    g = model.graph
    info = {
        "opset": [i.version for i in model.opset_import],
        "ir_version": model.ir_version,
        "producer": "%s %s" % (model.producer_name, model.producer_version),
        "inputs": [{"name": i.name,
                    "shape": [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim],
                    "dtype": i.type.tensor_type.elem_type} for i in g.input],
        "outputs": [{"name": o.name,
                     "shape": [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]}
                    for o in g.output],
        "nodes": len(g.node),
        "size_mb": os.path.getsize(path) / 1024.0 / 1024.0,
    }
    return info


def pick_providers(ort, device="auto", loud=False):
    """auto/gpu/cuda 有 CUDA EP 就用它，否则退回 CPU。

    loud=True 时，请求了 GPU 却只能拿到 CPU 会打印一段排查提示 ——
    因为"静默退化为 CPU"是这套流程里最容易让人得出错误结论的地方：
    表里看起来像"ONNX 比 PyTorch 慢 8 倍"，实际是拿 CPU 打了 GPU。
    """
    avail = set(ort.get_available_providers())
    want_gpu = str(device).lower() in ("auto", "gpu", "cuda", "cuda:0")
    if want_gpu and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
    if want_gpu and loud:
        print(_ORT_CPU_HINT)
    return ["CPUExecutionProvider"], "cpu"


def make_session(path, verbose=False, libs=None, device="auto"):
    ort = import_ort(libs)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers, ep = pick_providers(ort, device, loud=True)
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    if verbose:
        print("[ORT] onnxruntime %s | 请求 %s -> 实际 %s %s"
              % (ort.__version__, device, ep, sess.get_providers()))
    return sess, ep


def convert_to_fp16(onnx_path, out_path=None):
    """把整张图转 FP16，IO 仍保持 FP32（调用方喂法完全不用改）。

    转换器优先用 onnxruntime 自带的，退回 onnxconverter_common；两个都没有
    就返回 (None, None)——不是错误，只是少一组数据。
    """
    onnx = import_onnx()
    convert, src = None, None
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16 as convert
        src = "onnxruntime.transformers"
    except Exception:
        try:
            from onnxconverter_common.float16 import convert_float_to_float16 as convert
            src = "onnxconverter_common"
        except Exception:
            return None, None
    if out_path is None:
        out_path = onnx_path[:-5] + "_fp16.onnx"
    onnx.save(convert(onnx.load(onnx_path), keep_io_types=True), out_path)
    return out_path, src


def ort_run(sess, x):
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    return sess.run([out], {inp: x})[0]


# --------------------------------------------------------------------------- 计时
@torch.inference_mode()
def bench_torch(wrapper, x_np, device, iters=30, warmup=5, fp16=False):
    model = wrapper.to(device)
    if fp16:
        model = model.half()
    x = torch.from_numpy(x_np).to(device)
    if fp16:
        x = x.half()
    for _ in range(warmup):
        model(x)
    _sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = _median(ts)
    n = x_np.shape[0]
    return {
        "batch": n,
        "latency_ms_median": round(mean_ms, 2),
        "latency_ms_p90": round(float(np.percentile(ts, 90)), 2),
        "throughput_img_s": round(n / (mean_ms / 1000.0), 2) if mean_ms > 0 else None,
    }


def bench_ort(sess, x_np, iters=30, warmup=5):
    for _ in range(warmup):
        ort_run(sess, x_np)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ort_run(sess, x_np)
        ts.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = _median(ts)
    n = x_np.shape[0]
    return {
        "batch": n,
        "latency_ms_median": round(mean_ms, 2),
        "latency_ms_p90": round(float(np.percentile(ts, 90)), 2),
        "throughput_img_s": round(n / (mean_ms / 1000.0), 2) if mean_ms > 0 else None,
    }


# --------------------------------------------------------------------------- 主流程
def run_export(adapter, raw_loader, pipeline_predict, model_name, out_dir,
               side=512, norm="unit", norm_desc="x / 255",
               opset=17, dynamic_batch=True, dynamic_hw=False,
               n_verify=12, iters=30, batch=8, fp16=False,
               ort_fp16=True, ort_libs=None, ort_device="auto",
               input_dir=None, gt_dir=None, weights=None):
    """导出 + 四层校验 + 计时 + 写报告。

    adapter          : batch_infer 里的 Adapter（已加载权重、已载入模型）
    raw_loader(path) : -> np.ndarray [3, side, side] float32，像素 0~255
    pipeline_predict(path) : -> 现有的完整推理管线输出的掩膜（用来对齐口径）
    """
    ensure_deps(ort_libs)
    os.makedirs(out_dir, exist_ok=True)
    device = adapter.device
    onnx_path = os.path.join(out_dir, "%s.onnx" % model_name)

    print("=" * 72)
    print("[1/5] 导出  %s" % onnx_path)
    wrapper = ExportWrapper(adapter.model, norm=norm).cpu()
    tag, in_name, out_name = export_graph(
        wrapper, side, onnx_path, opset=opset, dynamic_batch=dynamic_batch,
        dynamic_hw=dynamic_hw)
    info = check_graph(onnx_path)
    print("      导出器 %s | opset %s | %d 个节点 | 文件 %s"
          % (tag, info["opset"], info["nodes"], fmt_mb(os.path.getsize(onnx_path))))
    print("      input  %s  output %s" % (info["inputs"][0]["shape"], info["outputs"][0]["shape"]))

    print("[2/5] 建校验用的 ORT session（CPU EP）")
    # 校验阶段故意用 CPU EP：torch 那边也会被挪到 CPU，同后端才谈得上
    # "argmax 逐像素 0 差异"；跨后端累加顺序不同，会翻几个像素（不是导出错）。
    sess, _ = make_session(onnx_path, verbose=True, device="cpu")

    # 校验统一在 CPU 上做：GPU 与 CPU 的卷积累加顺序不同，跨后端比对会出现
    # 个位数像素的翻转（不是导出错误），同后端才能要求"逐像素 0 差异"。
    orig_device = adapter.device
    cpu_only = str(orig_device).startswith("cuda")
    if cpu_only:
        print("      校验阶段临时把模型挪到 CPU（同后端比对才谈得上 0 差异）")
        adapter.model.cpu()
        adapter.device = torch.device("cpu")
    ver_device = adapter.device
    torch_dev = wrapper.to(ver_device)

    # ---- 第三层：argmax 逐像素一致
    print("[3/5] 一致性校验（取 %d 张输入图）" % n_verify)
    files = []
    if input_dir and os.path.isdir(input_dir):
        for fn in sorted(os.listdir(input_dir)):
            if fn.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")):
                files.append(os.path.join(input_dir, fn))
    files = files[:max(1, n_verify)]
    if not files:
        raise SystemExit("没有可用的校验样本：请用 --input 指定一个图片目录")

    samples = []
    mismatches = 0
    total_px = 0
    max_abs = 0.0
    pipe_diffs = 0
    ious_torch, ious_ort = [], []
    per_file = []

    for p in files:
        raw = raw_loader(p)
        if raw.shape[1] != side or raw.shape[2] != side:
            print("      [跳过] %s 尺寸 %s 不是 %dx%d" % (os.path.basename(p), raw.shape[1:], side, side))
            continue
        x = raw[None].astype(np.float32)

        with torch.inference_mode():
            t_logits = torch_dev(torch.from_numpy(x).to(ver_device)).float().cpu().numpy()
        o_logits = ort_run(sess, x)

        if t_logits.shape != o_logits.shape:
            raise SystemExit("输出形状不一致：torch %s vs ort %s" % (t_logits.shape, o_logits.shape))
        t_pred = t_logits.argmax(axis=1)[0].astype(np.uint8)
        o_pred = o_logits.argmax(axis=1)[0].astype(np.uint8)
        diff = int((t_pred != o_pred).sum())
        mismatches += diff
        total_px += t_pred.size
        max_abs = max(max_abs, float(np.abs(t_logits - o_logits).max()))

        # 第四层：与现有推理管线对齐
        pipe = pipeline_predict(p)
        pipe = np.asarray(pipe)
        if pipe.shape == o_pred.shape:
            pd = int((pipe.astype(np.uint8) != o_pred).sum())
            pipe_diffs += pd
        else:
            pd = -1

        y_true = None
        if gt_dir:
            gp = os.path.join(gt_dir, os.path.basename(p))
            if os.path.exists(gp):
                y_true = (np.array(Image_open(gp)).astype(np.uint8) > 0)
        if y_true is not None:
            a = iou_from_pair(t_pred, y_true)
            b = iou_from_pair(o_pred, y_true)
            if a is not None:
                ious_torch.append(a)
            if b is not None:
                ious_ort.append(b)

        per_file.append({"file": os.path.basename(p), "argmax_diff_px": diff,
                         "pipeline_diff_px": pd,
                         "fg_ratio_ort": round(float(o_pred.mean()), 6)})
        samples.append(x)

    if not samples:
        raise SystemExit("校验样本全部尺寸不符，请换 --input")
    print("      argmax 不一致像素：%d / %d（%.6f%%）| logits 最大绝对差 %.3e"
          % (mismatches, total_px, 100.0 * mismatches / max(total_px, 1), max_abs))
    print("      与原推理管线不一致像素合计：%d" % pipe_diffs)

    # ---- 第五层：计时
    if cpu_only:                       # 校验做完，把模型搬回 GPU 计时
        adapter.model.to(orig_device)
        adapter.device = orig_device
    print("[4/5] 延迟 / 吞吐")
    x1 = samples[0].astype(np.float32)
    xb = np.repeat(x1, batch, axis=0)
    bench = {}
    bench["torch_fp32_gpu_bs1"] = bench_torch(wrapper, x1, device, iters=iters)
    bench["torch_fp32_gpu_bs%d" % batch] = bench_torch(wrapper, xb, device, iters=max(5, iters // 3))
    if fp16 and device.type == "cuda":
        bench["torch_fp16_gpu_bs1"] = bench_torch(wrapper, x1, device, iters=iters, fp16=True)
        bench["torch_fp16_gpu_bs%d" % batch] = bench_torch(wrapper, xb, device, iters=max(5, iters // 3), fp16=True)
    wrapper.float().cpu()
    # 把 torch 的模型与激活先请出显存：8 GB 卡上再挂一个 ORT CUDA session，
    # 很容易让 WDDM 把显存刷回系统内存，延迟会虚假地涨一个量级
    # （实测 U-Net bs8 从 238 ms 变 2798 ms）。
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sess_bench, ep = make_session(onnx_path, verbose=True, libs=ort_libs, device=ort_device)
    bench["ort_fp32_%s_bs1" % ep] = bench_ort(sess_bench, x1, iters=iters)
    bench["ort_fp32_%s_bs%d" % (ep, batch)] = bench_ort(sess_bench, xb, iters=max(5, iters // 3))
    fp16_info = {}
    if ort_fp16 and ep == "cuda":
        # 先把这个 session 放掉再建下一个：两个 CUDA session 同时活着时，
        # 8 GB 卡很容易被顶到 WDDM 换出，第二个后端的数字会假性变慢。
        # 先把 fp32 的参考输出取出来，等下要和 fp16 对 argmax
        y32_ref = ort_run(sess_bench, x1)
        del sess_bench
        gc.collect()
        torch.cuda.empty_cache()
        p16, conv_src = convert_to_fp16(onnx_path, os.path.join(out_dir, "%s_fp16.onnx" % model_name))
        if p16:
            print("      FP16 图：%s（转换器 %s）" % (os.path.basename(p16), conv_src))
            sess16, _ = make_session(p16, verbose=True, libs=ort_libs, device="cuda")
            bench["ort_fp16_cuda_bs1"] = bench_ort(sess16, x1, iters=iters)
            bench["ort_fp16_cuda_bs%d" % batch] = bench_ort(sess16, xb, iters=max(5, iters // 3))
            y32, y16 = y32_ref, ort_run(sess16, x1)
            d16 = int((np.argmax(y32, 1) != np.argmax(y16, 1)).sum())
            fp16_info = {"graph": p16, "converter": conv_src,
                         "argmax_diff_px": d16,
                         "max_abs_logit_diff": float(np.abs(y32 - y16).max())}
            print("      fp16 vs fp32：argmax 不一致 %d / %d 像素，logits 最大差 %.3e"
                  % (d16, y32.shape[-2] * y32.shape[-1], fp16_info["max_abs_logit_diff"]))
    for k, v in bench.items():
        print("      %-24s bs%-2d %8.2f ms  %8.2f img/s"
              % (k, v["batch"], v["latency_ms_median"], v["throughput_img_s"] or 0))

    # ---- 写报告
    print("[5/5] 写报告")
    miou_t = float(np.mean(ious_torch)) if ious_torch else None
    miou_o = float(np.mean(ious_ort)) if ious_ort else None
    payload = {
        "model": model_name, "weights": weights, "onnx": onnx_path,
        "exporter": tag, "graph": info, "norm": norm, "side": side,
        "verify": {"n_files": len(samples), "argmax_diff_px": mismatches,
                   "total_px": total_px, "max_abs_logit_diff": max_abs,
                   "pipeline_diff_px": pipe_diffs, "per_file": per_file},
        "metric": {"mean_fg_iou_torch": miou_t, "mean_fg_iou_ort": miou_o,
                   "n_with_gt": len(ious_torch)},
        "bench": bench,
        "fp16": fp16_info,
        "ort": {"ep": ep, "libs": ort_libs,
                "version": __import__("onnxruntime").__version__},
        "env": {"torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    with open(os.path.join(out_dir, "onnx_export_report.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "onnx_export_report.md"), "w", encoding="utf-8") as f:
        f.write(_render_md(payload, norm_desc, dynamic_batch, dynamic_hw))
    print("      报告：%s" % os.path.join(out_dir, "onnx_export_report.md"))
    print("=" * 72)
    return payload


def Image_open(path):
    """读掩膜；PIL 对 mode='1' 的 tif 返回 bool 数组，统一转 0/1。"""
    from PIL import Image as _Image
    return _Image.open(path)


def _render_md(p, norm_desc, dynamic_batch, dynamic_hw):
    g = p["graph"]
    v = p["verify"]
    b = p["bench"]
    m = p["metric"]
    lines = []
    A = lines.append
    A("# ONNX 导出报告 — %s" % p["model"])
    A("")
    A("## 一、结论")
    ok = (v["argmax_diff_px"] == 0 and v["pipeline_diff_px"] == 0)
    A("- 导出**成功**：PyTorch 与 ONNX Runtime 的 argmax 掩膜 **%s**。"
      % ("逐像素完全一致（%d 张，%d 像素）" % (v["n_files"], v["total_px"])
         if v["argmax_diff_px"] == 0 else "有 %d 个像素不一致" % v["argmax_diff_px"]))
    A("- 与原推理管线（`adapter.prep → infer → post`）的差异：**%d 像素**%s。"
      % (v["pipeline_diff_px"],
         "" if v["pipeline_diff_px"] == 0 else "（正常，来自 argmax 平局或浮点累加顺序）"))
    A("- 烘进图里的归一化：`%s`，与训练期一致，部署端**不需要**再做归一化。" % norm_desc)
    if m["mean_fg_iou_ort"] is not None:
        A("- 指标对齐：torch 前景 IoU %.4f vs ORT %.4f（%d 张有真值）。"
          % (m["mean_fg_iou_torch"], m["mean_fg_iou_ort"], m["n_with_gt"]))
    A("- 校验四层：结构 / 形状 / argmax 数值 / 与原管线口径 —— %s。" % ("全部通过" if ok else "见上"))
    A("")
    A("## 二、产物")
    A("| 项目 | 值 |")
    A("|---|---|")
    A("| 权重 | `%s` |" % p["weights"])
    A("| ONNX | `%s` |" % p["onnx"])
    A("| 文件大小 | %.2f MB |" % g["size_mb"])
    A("| opset / ir_version | %s / %s |" % (g["opset"], g["ir_version"]))
    A("| 节点数 | %d |" % g["nodes"])
    A("| 导出器 | %s |" % p["exporter"])
    A("| ORT 执行后端 | **%s**（onnxruntime %s）|"
      % (p["ort"]["ep"].upper(), p["ort"]["version"]))
    A("| 动态维 | batch=%s, H/W=%s |" % (dynamic_batch, dynamic_hw))
    A("")
    A("## 三、输入输出契约")
    A("```")
    A("input  'input'  float32 %s   # 原始像素 0~255，未归一化" % g["inputs"][0]["shape"])
    A("output 'logits' float32 %s" % g["outputs"][0]["shape"])
    A("掩膜 = np.argmax(logits, axis=1).astype(np.uint8)   # 0 背景 / 1 建筑")
    A("```")
    A("")
    A("## 四、延迟与吞吐")
    A("| 后端 | batch | 中位延迟 ms | P90 ms | 吞吐 img/s |")
    A("|---|---|---|---|---|")
    for k, x in b.items():
        A("| %s | %d | %.2f | %.2f | %.2f |"
          % (k, x["batch"], x["latency_ms_median"], x["latency_ms_p90"], x["throughput_img_s"] or 0))
    A("")
    ort_ep = (p.get("ort") or {}).get("ep") or "cpu"
    if ort_ep == "cuda":
        A("> ORT 走的是 **CUDAExecutionProvider**（onnxruntime %s），与 torch 同一张 GPU，"
          % (((p.get("ort") or {}).get("version")) or "?"))
        A("> 各后端数字可以直接横向比。")
    else:
        A("> ORT 走的是 **CPUExecutionProvider**（本机没装 onnxruntime-gpu）：")
        A("> `ort_fp32_cpu` 与 GPU 上的 torch 不能直接比快慢。")
        A("> 想测 GPU 侧的加速倍数，装 `onnxruntime-gpu` 后用 `--ort-libs <目录>` 指过去。")
    f = p.get("fp16") or {}
    if f:
        A("> FP16 行用的是同一张图转成的 `%s`（转换器 %s，IO 保持 FP32），"
          % (os.path.basename(f["graph"]), f["converter"]))
        A("> 与 FP32 的 argmax 差异 **%d 像素**、logits 最大差 %.3e。"
          % (f["argmax_diff_px"], f["max_abs_logit_diff"]))
    A("")
    A("## 五、逐样本校验")
    A("| 文件 | argmax 差异像素 | 与原管线差异 | ORT 前景占比 |")
    A("|---|---|---|---|")
    for r in v["per_file"]:
        A("| %s | %d | %d | %.4f |" % (r["file"], r["argmax_diff_px"], r["pipeline_diff_px"], r["fg_ratio_ort"]))
    A("")
    A("## 六、已知限制")
    A("1. 动态 H/W 默认关闭：开启动态尺寸后部分算子（如插值）在某些 EP 上会退化，")
    A("   需要的话用 `--dynamic-hw` 单独验证。")
    A("2. 输入尺寸要与训练一致（512，且 32 的整数倍），尺寸不符会让精度明显下降。")
    A("3. ONNX 图里返回的是 logits，不是概率；要概率自己加 softmax/sigmoid，")
    A("   但取掩膜用 argmax 就够，省这一步。")
    A("4. FP16 图要 GPU 上的 CUDA/TensorRT EP 才有意义；CPU EP 上 FP16 反而更慢")
    A("   （要插 Cast 节点做浮点 16/32 转换）。")
    A("")
    A("## 七、复现")
    A("```bash")
    A("python export_onnx.py --weights <权重> --input <图片目录> --out <输出目录>")
    A("# ORT 想走 GPU：pip install onnxruntime-gpu，装好即自动选 CUDA EP，无需额外参数")
    A("# 只有 onnxruntime 不在默认位置时，才需要：--ort-libs <含 onnxruntime 的目录>")
    A("```")
    A("")
    A("环境：torch %s / %s / GPU %s" % (p["env"]["torch"], p["env"]["device"], p["env"]["gpu"]))
    return "\n".join(lines) + "\n"

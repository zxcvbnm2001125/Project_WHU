"""
下载 MiT（SegFormer 主干）的 ImageNet 预训练权重，并转换成本项目 model/mit.py 的参数命名。

为什么需要这个脚本：
    SegFormer 的论文结论全部建立在 ImageNet 预训练主干之上。Transformer 没有
    BatchNorm 那种"结构先验"，在 4000 多张图上从零训练，几乎不可能收敛到论文水平。

为什么还要"转换"：
    三家发布方的参数命名各不相同，而本项目 model/mit.py 用的是**官方 mmseg 命名**：

        mmseg / NVlabs 官方   patch_embed1.proj.weight   block1.0.attn.kv.weight   norm1.weight
        HuggingFace 转存      encoder.patch_embeddings.0.proj.weight
        （本项目 = 第一行）

    HuggingFace 版本还有一处**结构差异**：它把 Q/K/V 拆成三个独立的 Linear，
    而官方把 K、V 融合成一个 `kv` Linear（dim → 2*dim）。所以转换时不能只改名字，
    还要把 key/value 两个权重沿输出维拼回去。本脚本的 convert_state_dict() 负责这件事。

用法：
    python download_pretrained.py                                  # 默认下载 b1
    python download_pretrained.py --variants b0 b1 b2
    python download_pretrained.py --endpoint https://hf-mirror.com  # 国内镜像

下载转换完成后文件落在 weights/mit_{variant}_imagenet.pth，
train.py 的 --backbone-weights 留空时会自动按这个名字去找。

如果你手上已经有官方 mmseg 命名的权重（mit_b1.pth），不需要下载，直接：
    python download_pretrained.py --from-local mit_b1.pth
"""

import argparse
import os
import re
import sys
import urllib.error
import urllib.request

import torch

# HuggingFace 上的 MiT 主干仓库（nvidia 官方转存，keys 为 transformers 命名）
HF_REPO = "nvidia/mit-{variant}"

OFFICIAL_NOTE = """
如果 HuggingFace 也访问不了，可以手动下载官方权重：
    SegFormer 官方仓库: https://github.com/NVlabs/SegFormer
    在 README 的 "pretrained backbone" 一节可以找到 mit_b0.pth ~ mit_b4.pth
下载后执行 python download_pretrained.py --from-local mit_b1.pth 即可。
"""


# ------------------------------------------------------------------ #
#   键名转换：transformers 命名 → 官方 mmseg 命名
# ------------------------------------------------------------------ #
_PATCH_RE = re.compile(r"^patch_embeddings\.(\d+)\.(.+)$")
_BLOCK_RE = re.compile(r"^block\.(\d+)\.(\d+)\.(.+)$")
_LN_RE = re.compile(r"^layer_norm\.(\d+)\.(.+)$")

# 块内部的普通改名（顺序有意义：长的、更具体的放前面）
_INNER_RENAMES = [
    ("layer_norm_1.", "norm1."),
    ("layer_norm_2.", "norm2."),
    ("attention.self.layer_norm.", "attn.norm."),   # 必须排在 attention.self.sr. 之前判定无冲突，但保持显式
    ("attention.self.sr.", "attn.sr."),
    ("attention.output.dense.", "attn.proj."),
    ("mlp.dense1.", "mlp.fc1."),
    ("mlp.dense2.", "mlp.fc2."),
    ("mlp.dwconv.dwconv.", "mlp.dwconv."),
]

_STRIP_PREFIXES = ("segformer.", "model.", "module.", "encoder.", "backbone.")


def strip_prefix(key):
    """剥掉整模型/骨干包装带来的前缀（可能叠加多层，循环剥干净）。"""
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _convert_block(stage, depth, items):
    """
    转换一个 Transformer 块（混合了 HF 的 attention.self.key / value）。

    返回 {mmseg 键: 张量}。
    """
    prefix = f"block{stage + 1}.{depth}."
    out = {}
    kv_parts = {}       # suffix('weight'/'bias') -> {'k': t, 'v': t}
    leftover = []

    for rest, value in items.items():
        if rest.startswith("attention.self.query."):
            out[prefix + "attn.q." + rest[len("attention.self.query."):]] = value
        elif rest.startswith("attention.self.key."):
            kv_parts.setdefault(rest[len("attention.self.key."):], {})["k"] = value
        elif rest.startswith("attention.self.value."):
            kv_parts.setdefault(rest[len("attention.self.value."):], {})["v"] = value
        else:
            for old, new in _INNER_RENAMES:
                if rest.startswith(old):
                    out[prefix + new + rest[len(old):]] = value
                    break
            else:
                leftover.append(rest)

    # K、V 融合成一个 kv 线性层：沿输出维（dim 0）拼接
    for suffix, pair in kv_parts.items():
        if "k" in pair and "v" in pair:
            out[prefix + "attn.kv." + suffix] = torch.cat([pair["k"], pair["v"]], dim=0)
        else:
            leftover.append("attn.kv." + suffix)

    # 认不出来的原样保留，交给后面的"按名字+形状匹配"去报告
    for rest in leftover:
        out[prefix + rest] = items[rest]

    return out


def convert_state_dict(state):
    """
    把任意命名风格的 SegFormer 主干权重转成本项目 model/mit.py 的命名。

    输入：
    - state: {旧键名: 张量}
    输出：
    - {mmseg 键名: 张量}，已剔除 decode_head / conv_seg 等分割头参数
    """
    out, blocks = {}, {}

    for key, value in state.items():
        key = strip_prefix(key)

        m = _BLOCK_RE.match(key)
        if m:
            blocks.setdefault((int(m.group(1)), int(m.group(2))), {})[m.group(3)] = value
            continue

        m = _PATCH_RE.match(key)
        if m:
            stage = int(m.group(1)) + 1
            rest = m.group(2).replace("layer_norm.", "norm.")
            out[f"patch_embed{stage}.{rest}"] = value
            continue

        m = _LN_RE.match(key)
        if m:
            out[f"norm{int(m.group(1)) + 1}.{m.group(2)}"] = value
            continue

        # 已经是 mmseg 命名（或无法识别的键）直接透传。
        # 唯一要修的是注意力模块名：NVlabs / mmseg 的原始权重里叫 attn1
        # （attn2 是解码器用的），本项目统一叫 attn，这里顺手改掉，
        # 让保存下来的 mit_bN_imagenet.pth 键名和 model/mit.py 完全一致。
        out[key.replace("attn1.", "attn.")] = value

    for (stage, depth), items in blocks.items():
        out.update(_convert_block(stage, depth, items))

    # 去掉不属于主干的参数
    head_prefixes = ("decode_head.", "conv_seg.", "head.", "aux_head.", "classifier.")
    return {k: v for k, v in out.items() if not k.startswith(head_prefixes)}


def load_state_dict_any(path):
    """
    读取任意格式的权重文件，返回 {键名: 张量}。

    兼容：
        - 纯 state_dict
        - {'state_dict': ...} / {'model': ...} / {'net': ...} / {'module': ...} 包裹
        - 整模型权重（含 decode_head / conv_seg，转换时会被过滤掉）
        - .bin / .pth / .pt / .ckpt / .safetensors
    """
    if path.lower().endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise SystemExit("读取 .safetensors 需要先安装 safetensors：pip install safetensors")
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(state, dict):
        for wrapper in ("state_dict", "model", "net", "module"):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
                break

    # 只保留张量，过滤掉 epoch / optimizer 之类的杂项
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


# ------------------------------------------------------------------ #
#   下载
# ------------------------------------------------------------------ #
def download_file(url, save_path, quiet=False):
    """带进度显示的下载（不依赖 requests / tqdm）。"""

    def _hook(count, block_size, total_size):
        if quiet or total_size <= 0:
            return
        done = min(count * block_size, total_size)
        pct = done * 100.0 / total_size
        sys.stdout.write(f"\r    {pct:5.1f}%  {done / 1048576:6.1f} / {total_size / 1048576:.1f} MB")
        sys.stdout.flush()

    tmp_path = save_path + ".part"
    urllib.request.urlretrieve(url, tmp_path, reporthook=_hook)
    if not quiet:
        sys.stdout.write("\n")
    os.replace(tmp_path, save_path)


def fetch_variant(variant, endpoint, out_dir, quiet=False):
    """下载并转换一个规格的主干权重，返回保存路径（失败返回 None）。"""
    variant = variant.lower()
    repo = HF_REPO.format(variant=variant)
    save_path = os.path.join(out_dir, f"mit_{variant}_imagenet.pth")

    if os.path.isfile(save_path):
        print(f"[跳过] {save_path} 已存在")
        return save_path

    # 先试 pytorch_model.bin，再试 model.safetensors
    raw_path = None
    for filename in ("pytorch_model.bin", "model.safetensors"):
        url = f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}"
        print(f"[下载] {variant}: {url}")
        target = os.path.join(out_dir, f".raw_{variant}_{filename}")
        try:
            download_file(url, target, quiet=quiet)
            raw_path = target
            break
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            print(f"    失败：{exc}")
            continue

    if raw_path is None:
        print(f"[错误] {variant} 下载失败。{OFFICIAL_NOTE}")
        return None

    print(f"[转换] {variant}: transformers 命名 → mmseg 命名（含 K/V 融合）")
    converted = convert_state_dict(load_state_dict_any(raw_path))
    torch.save(converted, save_path)
    try:
        os.remove(raw_path)
    except OSError:
        pass

    print(f"[完成] {variant} → {save_path}（{len(converted)} 个张量）")
    return save_path


# ------------------------------------------------------------------ #
#   校验：把转换后的权重套到真实模型上，看能对上多少
# ------------------------------------------------------------------ #
def verify(variant, weights_path):
    """
    按"名字 + 形状"把权重套到真实模型上，打印匹配率。

    这是唯一能确认"命名转换是否成功"的手段：
        匹配率 100%  → 转换正确
        匹配率很低  → 命名规则没对上，训练时会退化成随机初始化（且只有警告，不会报错）
    """
    from model.mit import build_mit

    target = build_mit(variant=variant).state_dict()
    state = load_state_dict_any(weights_path)

    matched, mismatch_shape, extra = 0, [], 0
    for key, value in state.items():
        if key not in target:
            extra += 1
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            mismatch_shape.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        matched += 1

    total = len(target)
    rate = matched / total * 100.0
    print(f"[校验] {variant}: 模型共 {total} 个参数张量，成功匹配 {matched} 个（{rate:.1f}%）")
    if mismatch_shape:
        print(f"       形状不符 {len(mismatch_shape)} 个，例如：")
        for key, got, want in mismatch_shape[:3]:
            print(f"         {key}: 权重{got} vs 模型{want}")
    if extra:
        print(f"       权重里多出 {extra} 个模型没有的键（多半是别的任务的头部，可忽略）")
    if rate < 90.0:
        print("       [警告] 匹配率偏低，说明命名转换没对上。")
        print("              解决办法：改用官方 mmseg 命名的 mit_b*.pth（--from-local），")
        print("              或把上面没匹配上的键名贴出来，再修 convert_state_dict()。")
    return rate


# ------------------------------------------------------------------ #
#   入口
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="下载并转换 SegFormer (MiT) ImageNet 预训练主干权重")
    parser.add_argument("--variants", nargs="+", default=["b3"],
                        choices=["b0", "b1", "b2", "b3"],
                        help="要下载的规格，可写多个，如 --variants b0 b1 b2")
    parser.add_argument("--endpoint", default="https://huggingface.co", type=str,
                        help="HuggingFace 端点；国内可换 https://hf-mirror.com")
    parser.add_argument("--out-dir", default="weights", type=str, help="保存目录")
    parser.add_argument("--from-local", default="", type=str,
                        help="跳过下载，直接把本地已有的权重（mmseg / 整模型 / HF）转换后存进 weights/")
    parser.add_argument("--quiet", default=False, action="store_true", help="不显示下载进度条")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.from_local:
        if not os.path.isfile(args.from_local):
            raise SystemExit(f"文件不存在：{args.from_local}")
        # 一个本地文件只对应一个规格（b0/b1/b2 的张量形状完全不同，转不了第二遍）
        if len(args.variants) != 1:
            raise SystemExit("--from-local 一次只能配一个 --variants，例如：\n"
                             "  python download_pretrained.py --variants b1 --from-local mit_b1.pth")
        converted = convert_state_dict(load_state_dict_any(args.from_local))
        variant = args.variants[0]
        save_path = os.path.join(args.out_dir, f"mit_{variant}_imagenet.pth")
        torch.save(converted, save_path)
        print(f"[转换] {args.from_local} → {save_path}（{len(converted)} 个张量）")
        verify(variant, save_path)
        return

    ok = []
    for variant in args.variants:
        path = fetch_variant(variant, args.endpoint, args.out_dir, quiet=args.quiet)
        if path:
            verify(variant, path)
            ok.append(path)

    if not ok:
        print("\n所有规格都下载失败。建议：")
        print("  1) 换镜像：python download_pretrained.py --endpoint https://hf-mirror.com")
        print("  2) 手动下载后转换：python download_pretrained.py --from-local 路径")
        raise SystemExit(1)

    print("\n下一步：")
    print(f"  python train.py --variant {args.variants[0]} --data-path ..\\U-Net\\WUH")


if __name__ == "__main__":
    main()

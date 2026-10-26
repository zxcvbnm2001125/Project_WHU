"""
MiT（Mix Transformer）编码器 —— SegFormer 的主干网络。

与 ResNet / VGG 那类"纯卷积"主干相比，MiT 有三处关键设计：

1) 重叠式 patch merging（OverlapPatchEmbed）
   第一级用 kernel=7 / stride=4、后面三级用 kernel=3 / stride=2 的卷积来降采样，
   而不是 ViT 那种"kernel=stride=16 的不重叠切块"。
   重叠意味着相邻 patch 共享边缘像素，局部连续性不会被切碎 —— 这一点对
   "建筑边界"这类高频细节任务非常重要。

2) 高效自注意力（sr_ratio > 1）
   标准自注意力要算 N×N 的注意力图，N 是 patch 数，分辨率一大就爆显存。
   这里在计算 K / V 之前先用一个 stride = sr_ratio 的卷积把序列长度降下来，
   相当于只让"稀疏的键值对"参与注意力，复杂度从 O(N²) 降到 O(N²/R²)。
   R = [8, 4, 2, 1]：分辨率越高降得越狠，最后一级保持全分辨率。

3) Mix-FFN：用 3×3 深度卷积代替位置编码
   公式是
       x = MLP( GELU( DWConv3x3( MLP(x) ) ) )
   ViT 需要位置编码是因为自注意力本身对顺序不敏感；但固定长度的位置编码
   一换分辨率就得插值，插值又会让精度掉点。
   3×3 深度卷积自带"局部位置信息"，因此 SegFormer 完全不需要位置编码，
   训练用 512、推理用 1024 也能直接跑，这一点在遥感大幅面影像上很实用。

本文件不依赖 timm，只依赖 PyTorch，方便离线部署与逐行阅读。
参数命名刻意与官方（NVlabs/SegFormer、mmsegmentation）保持一致，
这样官方发布的 ImageNet 预训练权重可以直接加载。
"""

import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_, zeros_, ones_


# ------------------------------------------------------------------ #
#   基础组件
# ------------------------------------------------------------------ #
class DropPath(nn.Module):
    """
    Stochastic Depth（随机深度）。

    训练时以概率 drop_prob 把整条残差分支置零（该样本的这条分支不参与前向与反向），
    推理时什么都不做。作用类似 dropout，但作用在"层"而不是"神经元"上，
    是 Transformer 里最常用的正则手段之一。

    注意：被丢弃的样本的残差分支输出直接是 0，所以保留下来的分支要除以 keep_prob，
    保证训练 / 推理两阶段的期望输出一致。
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # 每个样本一个独立的伯努利掩码，形状 (B, 1, 1, ...)，对整条分支做广播
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"


class OverlapPatchEmbed(nn.Module):
    """
    重叠式 patch 嵌入（同时完成降采样）。

    与 ViT 的 Conv2d(kernel=16, stride=16) 不同，这里的 kernel > stride，
    相邻窗口之间会有重叠，等价于一次带局部邻域信息的卷积降采样。
    padding = kernel // 2，保证 H、W 都被整齐地除以 stride。

    forward 返回 (x, H, W)：
        x 形状 [B, H*W, C]（序列形式，供注意力使用）
        H、W 是降采样后的空间尺寸，后面的注意力需要用它们把序列还原成二维图
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                      # [B, C, H, W]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)      # [B, H*W, C]
        x = self.norm(x)
        return x, H, W


class Mlp(nn.Module):
    """
    Mix-FFN：带 3×3 深度卷积的两层前馈网络。

    结构：Linear → DWConv3x3 → GELU → Linear
    （官方代码里 dropout 放在 GELU 之后和最后一层 Linear 之后，
      本项目 dropout 默认 0，保持结构一致即可）

    DWConv 之所以加在两层 Linear 中间，是为了让前馈网络自己具备"看邻域"的能力，
    从而替代 ViT 里的位置编码。注意它是 depthwise 卷积（groups=C），
    参数量只有 9×C，非常轻。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        # 序列 → 二维图，做深度卷积后再变回序列
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    高效自注意力（Efficient Self-Attention）。

    相比标准多头注意力，唯一的区别是 K / V 的序列长度被 sr_ratio 降采样：

        有 sr_ratio = R > 1 时：
            x_[B,C,H,W] = sr(conv stride=R)(x)   →  [B, C, H/R, W/R]
            K, V 只在这 H*W/R² 个 token 上计算
        注意力图尺寸是 [B, heads, N, N/R²]，而不是 [B, heads, N, N]

    为什么要降：ViT 的注意力是 O(N²)。在 512×512 输入下，1/4 分辨率的 N = 128×128 =
    16384，全量 N² 是 2.7 亿，显存放不下；R=8 之后 K/V 只剩 256 个 token，
    计算量直接降到原来的 1/64，而 Q 仍然来自全分辨率，所以"每个位置"依然能看到全局。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须能被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 缩放因子，防止点积随维度增大而数值爆炸
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 用 stride = R 的卷积把 K/V 的序列长度降下来
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Q：[B, heads, N, head_dim]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # 序列 → 二维 → 降采样 → 序列
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # kv：[2, B, heads, N/R², head_dim]
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, heads, N, N/R²]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer 基本块：Pre-Norm 残差结构。

        x = x + DropPath( Attention( LayerNorm(x) ) )
        x = x + DropPath( MLP( LayerNorm(x) ) )

    Pre-Norm（先归一化再进子层）是 Transformer 训练稳定的关键，
    也是 SegFormer 全网络没有任何 BatchNorm 的原因 —— 它对 batch size 不敏感，
    用 batch 4 还是 16 结果都一样，这一点对只有 8GB 显存的笔记本很友好。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


# ------------------------------------------------------------------ #
#   主干网络
# ------------------------------------------------------------------ #
class MixVisionTransformer(nn.Module):
    """
    分四阶段输出的分层 Transformer 主干。

    每个 stage 的输出分辨率依次是输入的 1/4、1/8、1/16、1/32，
    输出一个长度为 4 的列表，元素形状 [B, C_i, H/s, W/s]（通道在前，与卷积特征一致），
    这样解码器可以直接拿来做 F.interpolate 融合。
    """

    def __init__(self, img_size=512, in_chans=3,
                 embed_dims=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=True, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
                 depths=(3, 4, 6, 3),
                 sr_ratios=(8, 4, 2, 1),
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.depths = list(depths)
        self.embed_dims = list(embed_dims)
        self.img_size = img_size

        # drop path 在 0 ~ drop_path_rate 之间按"全局层序号"线性递增：
        # 浅层几乎不丢弃，深层丢弃得多，这是 ViT 系列的标准做法
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        # ---------------- Stage 1：1/4 ----------------
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4,
                                              in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0],
                  mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        # ---------------- Stage 2：1/8 ----------------
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1],
                  mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        # ---------------- Stage 3：1/16 ----------------
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2],
                  mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        # ---------------- Stage 4：1/32 ----------------
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3],
                  mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer,
                  sr_ratio=sr_ratios[3])
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    # ------------------------------------------------------------- #
    #   初始化
    # ------------------------------------------------------------- #
    @staticmethod
    def _init_weights(m):
        """
        与官方实现一致的初始化策略：
        - Linear：截断正态 N(0, 0.02)，截断在 ±2σ；
        - LayerNorm：weight=1、bias=0；
        - Conv（含深度卷积）：N(0, sqrt(2 / fan_out))，即 ReLU/GELU 友好的 He 初始化。
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        """从头初始化整条主干（不加载预训练时使用）。"""
        self.apply(self._init_weights)

    # ------------------------------------------------------------- #
    #   逐层学习率衰减（layer-wise lr decay）需要的辅助信息
    #   SegFormer 官方训练配方里，越靠近输入的层学习率越小（decay 0.9 每层），
    #   目的是不让随机初始化的小学习率层把预训练特征带偏。
    # ------------------------------------------------------------- #
    @property
    def num_layers(self) -> int:
        """把 patch_embed / 每个 block / 每个 stage 的 norm 都算作"一层"。"""
        return sum(self.depths) + 1

    def get_layer_id(self, name: str) -> int:
        """
        根据参数名判断它属于第几层（0 最浅，num_layers-1 最深）。

        参数：
        - name: 去掉 'backbone.' 前缀之后的参数名，如 'block3.2.attn.q.weight'
        """
        offset = 0
        for stage in range(1, 5):
            depth = self.depths[stage - 1]
            if name.startswith(f"patch_embed{stage}"):
                # patch 嵌入与上一 stage 的末层放在同一学习率档
                return offset
            if name.startswith(f"block{stage}."):
                block_idx = int(name.split(".")[1])
                return offset + block_idx + 1
            if name.startswith(f"norm{stage}."):
                return offset + depth
            offset += depth
        # 兜底：认不出来的参数按最深层处理
        return self.num_layers - 1

    # ------------------------------------------------------------- #
    #   前向
    # ------------------------------------------------------------- #
    def forward_features(self, x):
        """返回 4 个 stage 的特征列表，形状依次为 [B,C,1/4]、[B,C,1/8]、[B,C,1/16]、[B,C,1/32]。"""
        B = x.shape[0]
        outs = []

        for i in range(4):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            blocks = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            # 序列 → 通道在前的二维图，便于和解码器做卷积/插值
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        return self.forward_features(x)

    # ------------------------------------------------------------- #
    #   预训练权重加载
    # ------------------------------------------------------------- #
    def load_pretrained(self, weights_path, map_location="cpu", verbose=True):
        """
        加载 ImageNet 预训练的 MiT 权重。

        之所以不直接用 load_state_dict(strict=True)，是因为官方权重的命名
        在不同版本里有细微差别（如 attention 叫 attn 还是 attn1、是否带
        'backbone.' / 'layers.' 前缀）。这里按"名字 + 形状"匹配，
        并对前缀和 attn/attn1 做归一化，能兼容绝大多数来源的权重文件。

        参数：
        - weights_path:  .pth / .bin 权重文件路径
        - map_location:  torch.load 的设备映射
        - verbose:       是否打印加载统计
        返回：
        - 加载后的模型
        """
        state = torch.load(weights_path, map_location=map_location)

        # 兼容被外层字典包裹的情况
        if isinstance(state, dict):
            for key in ("state_dict", "model", "net", "encoder"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

        own = self.state_dict()
        prefixes = ("backbone.", "layers.", "module.", "model.",
                    "segformer.encoder.", "encoder.")
        loaded = set()

        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            # 生成候选名字：原名 → 去前缀 → attn/attn1 互换
            cands = [key]
            for p in prefixes:
                if key.startswith(p):
                    cands.append(key[len(p):])
            for c in list(cands):
                cands.append(c.replace(".attn1.", ".attn."))
                cands.append(c.replace(".attn.", ".attn1."))

            for c in cands:
                target = own.get(c)
                if target is not None and target.shape == value.shape:
                    own[c] = value
                    loaded.add(c)
                    break

        self.load_state_dict(own)

        if verbose:
            total = len(own)
            print(f"[MiT 预训练] {weights_path}")
            print(f"[MiT 预训练] 成功加载 {len(loaded)}/{total} 个主干参数")
            if len(loaded) < total * 0.5:
                print("[MiT 预训练][警告] 加载比例低于 50%，"
                      "很可能是权重来源的命名与本实现差异较大，请核对权重文件")
        return self


# ------------------------------------------------------------------ #
#   四个常用规格
#   embed_dims / depths / heads 均与官方保持一致
# ------------------------------------------------------------------ #
MIX_TRANSFORMER_CFGS = {
    "b0": dict(embed_dims=(32, 64, 160, 256), depths=(2, 2, 2, 2),
               num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1),
               mlp_ratios=(4, 4, 4, 4)),
    "b1": dict(embed_dims=(64, 128, 320, 512), depths=(2, 2, 2, 2),
               num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1),
               mlp_ratios=(4, 4, 4, 4)),
    "b2": dict(embed_dims=(64, 128, 320, 512), depths=(3, 4, 6, 3),
               num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1),
               mlp_ratios=(4, 4, 4, 4)),
    "b3": dict(embed_dims=(64, 128, 320, 512), depths=(3, 4, 18, 3),
               num_heads=(1, 2, 5, 8), sr_ratios=(8, 4, 2, 1),
               mlp_ratios=(4, 4, 4, 4)),
}

# 各规格的主干参数量（百万），仅用于打印提示
MIT_PARAMS_M = {"b0": 3.7, "b1": 13.2, "b2": 24.7, "b3": 44.6}


def build_mit(variant="b1", img_size=512, drop_path_rate=0.1,
              drop_rate=0.0, attn_drop_rate=0.0, in_chans=3):
    """
    按规格名构造 MiT 主干。

    参数：
    - variant:       'b0' / 'b1' / 'b2' / 'b3'
    - img_size:      训练时输入边长（只作为记录，不影响结构，因为无位置编码）
    - drop_path_rate: 随机深度的最大丢弃率（0.1 是论文里的常用值）
    """
    variant = str(variant).lower()
    if variant not in MIX_TRANSFORMER_CFGS:
        raise ValueError(f"不支持的 variant: {variant!r}，可选 {list(MIX_TRANSFORMER_CFGS)}")
    cfg = dict(MIX_TRANSFORMER_CFGS[variant])
    return MixVisionTransformer(img_size=img_size, in_chans=in_chans,
                                drop_rate=drop_rate,
                                attn_drop_rate=attn_drop_rate,
                                drop_path_rate=drop_path_rate,
                                **cfg)

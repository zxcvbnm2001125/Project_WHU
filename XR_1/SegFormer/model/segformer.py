"""
SegFormer：MiT 编码器 + 轻量 MLP 解码器。

与 U-Net / DeepLabV3+ 的解码器相比，这里刻意做得"极简"：

    U-Net       ：4 次上采样，每次都要和同尺度特征 concat + 两层 3×3 卷积，
                  解码器约占全模型一半的算力；
    DeepLabV3+  ：ASPP（多尺度空洞卷积）在 1/16 上做重计算，再靠一条低层跳连补细节；
    SegFormer   ：4 个 stage 的特征各自过一个 Linear 投到同一个通道数，
                  全部插值到 1/4 分辨率后 concat，再用一个 1×1 卷积融合，
                  最后一次性插值回原图。解码器只有十几万参数。

"解码器这么简陋，为什么精度还够？"
    因为主干本身已经是分层的，1/4 那一层就带着细节，1/32 那一层带着全局语义，
    fusion 在 1/4 上完成 —— 也就是说 SegFormer 一直在 1/4 分辨率（512 输入下是
    128×128）上做预测，等价于 U-Net 的 output_stride = 4。
    相比之下，姊妹项目 U-Net-OS 的基线是 OS=32（16×16 预测，再插值 32 倍），
    细粒度信息差了 8 倍。这就是 SegFormer 在小目标 / 细边界上更占优的根本原因。

本实现相对官方 mmseg 的唯一有意改动：
    fusion 卷积后的归一化用 GroupNorm 而不是 BatchNorm。
    原因：本项目要在 8GB 显存的笔记本上跑，B2 档可能得把 batch 降到 4；
    BatchNorm 在 batch 变小时统计量不稳，而 GroupNorm 与 batch 完全无关，
    使整个 SegFormer 彻底没有 BN，训练 / 推理行为完全一致（单图推理也不需要
    running stats）。改回 BN 只需把 `norm_layer` 换成 nn.BatchNorm2d。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, zeros_

from model.mit import MIX_TRANSFORMER_CFGS, MixVisionTransformer, build_mit


# ------------------------------------------------------------------ #
#   解码器
# ------------------------------------------------------------------ #
class SegFormerHead(nn.Module):
    """
    SegFormer 的 MLP 解码器（All-MLP Decoder）。

    流程：
        4 个 stage 特征 [1/4, 1/8, 1/16, 1/32]
            → 各自 Linear 投影到 embed_dim
            → 全部双线性插值到 1/4 分辨率
            → concat（4×embed_dim）
            → 1×1 卷积 + Norm + ReLU 融合（回到 embed_dim）
        输出：1/4 分辨率的融合特征

    参数：
    - in_channels:  主干四个 stage 的输出通道，如 (64,128,320,512)
    - embed_dim:    统一后的通道数（官方默认 256；本实现默认 128，省显存）
    - dropout:      fusion 之后的 Dropout2d 比例
    - norm_layer:   融合卷积用的归一化层
    """

    def __init__(self, in_channels, embed_dim=128, dropout=0.1,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            # GroupNorm 分组数取 32，要求 embed_dim % 32 == 0
            groups = 32 if embed_dim % 32 == 0 else 8
            norm_layer = lambda c: nn.GroupNorm(groups, c)

        self.embed_dim = embed_dim
        # 每个 stage 一个 1×1 的"线性投影"，官方实现就是 nn.Linear + 逐像素
        self.linear_layers = nn.ModuleList([
            nn.Linear(c, embed_dim) for c in in_channels
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, inputs):
        """
        参数：
        - inputs: 长度 4 的特征列表，元素形状 [B, C_i, H_i, W_i]
        返回：
        - [B, embed_dim, H, W]，H、W 为输入图的 1/4
        """
        # 1) 逐 stage 做逐像素线性投影：[B,C,H,W] → [B,C,H,W]
        laterals = []
        for i, x in enumerate(inputs):
            # 把通道维换到最后做 Linear，再换回来（等价于 1×1 卷积）
            y = x.flatten(2).transpose(1, 2)          # [B, H*W, C_in]
            y = self.linear_layers[i](y)              # [B, H*W, C_emb]
            y = y.transpose(1, 2).reshape(x.shape[0], self.embed_dim, *x.shape[2:])
            laterals.append(y)

        # 2) 全部插值到最高分辨率那一层（1/4）
        size = inputs[0].shape[2:]
        if any(lat.shape[2:] != size for lat in laterals):
            laterals = [
                lat if lat.shape[2:] == size else
                F.interpolate(lat, size=size, mode="bilinear", align_corners=False)
                for lat in laterals
            ]

        # 3) 拼接后用 1×1 卷积融合
        fused = torch.cat(laterals, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.dropout(fused)
        return fused


# ------------------------------------------------------------------ #
#   完整模型
# ------------------------------------------------------------------ #
class SegFormer(nn.Module):
    """
    参数：
    - variant:        主干规格，'b0' / 'b1' / 'b2' / 'b3'
    - num_classes:    类别数（含背景）
    - decoder_dim:    解码器统一通道数
    - dropout:        解码器 dropout
    - drop_path_rate: 主干随机深度上限
    - img_size:       训练输入边长（仅记录用，无位置编码，不影响结构）
    """

    def __init__(self, variant="b1", num_classes=2, decoder_dim=128,
                 dropout=0.1, drop_path_rate=0.1, img_size=512):
        super().__init__()
        self.variant = str(variant).lower()
        self.num_classes = num_classes

        # 主干预设：SegFormer 官方在分割训练时 drop_path_rate=0.1
        self.backbone = build_mit(variant=self.variant, img_size=img_size,
                                  drop_path_rate=drop_path_rate)
        self.decode_head = SegFormerHead(self.backbone.embed_dims,
                                         embed_dim=decoder_dim,
                                         dropout=dropout)

        # 最终的逐像素分类器
        self.conv_seg = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

        self.init_head_weights()

    # ------------------------------------------------------------- #
    #   初始化
    # ------------------------------------------------------------- #
    def init_head_weights(self):
        """
        解码器与分类头的初始化。

        conv_seg 用小方差正态初始化（std=0.01）并把 bias 置 0，
        这样训练初期各类别的输出接近 0，softmax 接近均匀分布，
        避免"一上来就全预测背景"的退化。
        """
        for m in self.decode_head.modules():
            if isinstance(m, nn.Linear):
                normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    zeros_(m.bias)
        normal_(self.conv_seg.weight, 0.0, 0.01)
        if self.conv_seg.bias is not None:
            zeros_(self.conv_seg.bias)

    # ------------------------------------------------------------- #
    #   前向
    # ------------------------------------------------------------- #
    def forward_features(self, x):
        """返回主干四个 stage 的特征（供可视化 / 调试用）。"""
        return self.backbone(x)

    def forward(self, x):
        """
        参数：
        - x: [B, 3, H, W]，H、W 需为 32 的整数倍
        返回：
        - [B, num_classes, H, W] 的 logits（与输入同尺寸）
        """
        feats = self.backbone(x)                 # 4 个 stage，1/4 ~ 1/32
        fused = self.decode_head(feats)          # 1/4

        logits = self.conv_seg(fused)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:],
                                   mode="bilinear", align_corners=False)
        return logits

    # ------------------------------------------------------------- #
    #   工具
    # ------------------------------------------------------------- #
    def num_parameters(self, trainable_only=True):
        """统计参数量（个）。"""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def param_groups_with_layer_decay(self, base_lr, weight_decay,
                                      layer_decay=0.9, verbose=False):
        """
        构造 AdamW 的参数组：逐层学习率衰减 + 区分是否施加 weight decay。

        为什么需要 layer-wise lr decay：
            Transformer 微调时，越浅的层学到的越是通用特征，
            用一个统一的大学习率去更新它们，容易把预训练学到的特征"冲掉"。
            官方 SegFormer 配方给每一层乘一个 layer_decay^(深度差) 的系数
            （默认 0.9），最深层用满 lr，最浅层只剩 lr × 0.9^(N-1)。

        weight decay 的分组规则（与 BEiT / MAE 一致）：
            一维参数（LayerNorm 的 weight/bias、所有 bias）不做衰减，
            其余（Linear / Conv 的权重）做衰减。

        返回：
        - 参数组列表，每个元素带 'lr_scale' 键，供 set_optimizer_lr 使用
        """
        layers = self.backbone.num_layers
        grouped = {}   # (layer_id, lr_scale, use_wd) -> [params]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("backbone."):
                sub_name = name[len("backbone."):]
                layer_id = self.backbone.get_layer_id(sub_name)
                lr_scale = layer_decay ** (layers - 1 - layer_id)
            else:
                # 解码器与分类头用满学习率（它们是随机初始化的新模块）
                layer_id = layers - 1
                lr_scale = 1.0

            no_decay = param.ndim <= 1 or name.endswith(".bias")
            key = (layer_id, round(lr_scale, 10), no_decay)
            grouped.setdefault(key, []).append(param)

        # 让 lr 最大的组排在前面：train_one_epoch 打印的 get_lr() 取的是第 0 组，
        # 这样屏幕上显示的就是"解码器的真实 lr"，更有参考价值
        ordered_keys = sorted(grouped.keys(), key=lambda k: (-k[1], k[0], k[2]))

        param_groups = []
        for key in ordered_keys:
            layer_id, lr_scale, no_decay = key
            param_groups.append({
                "params": grouped[key],
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else weight_decay,
                "layer_id": layer_id,
                "name": f"layer{layer_id}{'_no_wd' if no_decay else ''}",
            })

        if verbose:
            print(f"[优化器] 共 {len(param_groups)} 个参数组，层数 {layers}，"
                  f"layer_decay={layer_decay}")
            scales = sorted({round(pg['lr_scale'], 6) for pg in param_groups})
            print(f"[优化器] lr 倍率取值: {scales[0]:.6f} ~ {scales[-1]:.6f}")

        return param_groups


def build_segformer(variant="b1", num_classes=2, **kwargs):
    """按规格名构造 SegFormer（便于在脚本里一行调用）。"""
    return SegFormer(variant=variant, num_classes=num_classes, **kwargs)

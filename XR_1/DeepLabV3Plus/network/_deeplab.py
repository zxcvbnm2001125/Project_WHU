import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel

__all__ = [
    "DeepLabV3Plus",              # 通用 segmentation wrapper，本工程只用于 DeepLabV3+
    "DeepLabHeadV3Plus",      # DeepLabV3+ 的 head（ASPP + decoder）
    "ASPP",
    "AtrousSeparableConvolution",
    "convert_to_separable_conv",
]


class DeepLabV3Plus(_SimpleSegmentationModel):
    """
    通用 DeepLabV3+ 分割模型壳（wrapper）

    - 不重写 forward，完全复用 _SimpleSegmentationModel 的实现：
        1) features = backbone(x) -> 返回 dict，如 {'out': ..., 'low_level': ...}
        2) logits = classifier(features) -> head 输出（DeepLabHeadV3Plus）
        3) 再把 logits interpolate 回输入大小，得到最终语义分割结果

    本工程中：backbone 可以是 ResNet / Xception / MobileNetV2，
    classifier 固定是 DeepLabHeadV3Plus。
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    """
    DeepLabV3+ 的 Head（ASPP + Decoder）

    输入：feature dict，要求至少包含：
      - feature['out']       : 高层语义特征（通常 stride = OS，如 16/8）
      - feature['low_level'] : 低层细节特征（通常 stride = 4，用于补边界细节）

    结构（与论文一致的核心思路）：
      1) low_level -> 1x1 Conv 降维到 48（减少拼接后的计算量）
      2) out -> ASPP -> 256 通道语义特征
      3) 将 ASPP 输出上采样到 low_level 的空间尺寸
      4) concat([low_level(48), aspp(256)]) => 304 通道
      5) 3x3 Conv(304->256) + 1x1 Conv(256->num_classes) 输出 logits
    """

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=(12, 24, 36)):
        super(DeepLabHeadV3Plus, self).__init__()

        # (1) low-level feature projection：用 1x1 Conv 把 low_level_channels -> 48
        # 论文里这样做是为了“减少 concat 后的计算量”（否则 low-level 通道太大会很慢）
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # (2) ASPP：对高层 out 做多尺度空洞卷积融合
        self.aspp = ASPP(in_channels, aspp_dilate)

        # (3) decoder classifier：concat 后 refinement + 输出类别 logits
        # concat 后通道数 = 48 + 256 = 304
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # refinement（细化）
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)  # 类别预测
        )
        self._init_weight()

    def forward(self, feature):
        # low_level: stride=4 的细节特征（例如 ResNet 的 layer1 / Xception 的 block1）
        low_level_feature = self.project(feature['low_level'])

        # out: 高层语义特征（stride=OS）
        output_feature = self.aspp(feature['out'])

        # 将 ASPP 输出上采样到 low-level 的空间尺寸（这样才能 concat）
        # - OS=16 时：out 通常是 1/16，low_level 是 1/4 => 上采样约 4倍
        # - OS=8  时：out 通常是 1/8， low_level 是 1/4 => 上采样约 2倍
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # concat：沿通道维拼接，然后做分类预测
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        # 常规初始化：Conv 用 kaiming，BN/GN 权重=1，bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class ASPPConv(nn.Sequential):
    """
    ASPP 的一个 3x3 atrous conv 分支：
    Conv(3x3, dilation=rate) -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    ASPP 的 image pooling 分支：
    AdaptiveAvgPool2d(1) -> 1x1 Conv -> BN -> ReLU -> 上采样回原 size

    作用：
    - 提供“全局上下文”（global context）
    - 与其他 atrous conv 分支一起 concat
    """

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]  # 记录输入特征图尺寸
        x = super(ASPPPooling, self).forward(x)
        # pool 后是 1x1，插值回原尺寸
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    ASPP 模块（Atrous Spatial Pyramid Pooling）

    分支（5路）：
      1) 1x1 conv
      2) 3x3 atrous conv (rate1)
      3) 3x3 atrous conv (rate2)
      4) 3x3 atrous conv (rate3)
      5) image pooling

    concat 后通道：5 * 256 = 1280
    再用 project(1x1) 压回 256，并 dropout。
    """

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256

        modules = []
        # 分支1：1x1 conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        # 分支2~4：不同 dilation 的 3x3 atrous conv
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))

        # 分支5：image pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # concat 后的融合：1x1 -> BN -> ReLU -> Dropout
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        # 收集每个分支输出
        res = []
        for conv in self.convs:
            res.append(conv(x))
        # 沿通道拼接
        res = torch.cat(res, dim=1)
        # 1x1 融合/压缩
        return self.project(res)





class AtrousSeparableConvolution(nn.Module):
    """
    Atrous Separable Convolution（空洞可分离卷积）：
    - depthwise: groups=in_channels 的 kxk 卷积（可 dilation）
    - pointwise: 1x1 卷积做通道混合

    用途：
    - 将普通 3x3 atrous conv 替换成更省算的 depthwise separable atrous conv
    - 在 paper / repo 里常作为“实现层面的优化”，结构不变、算子变轻量
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # depthwise conv：每个输入通道独立做空间卷积
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=bias, groups=in_channels
            ),
            # pointwise conv：1x1 混合通道
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



def convert_to_separable_conv(module):
    """
    递归替换器：把 module 里所有 kernel_size>1 的 nn.Conv2d 替换成 AtrousSeparableConvolution。

    注意：
    - 这里的替换是“按算子类型替换”，不只 ASPP，decoder/classifier 里的 3x3 conv 也可能被替换。
    - 目的：减少计算量/参数量，加速推理。
    """
    new_module = module

    # 只替换普通卷积（Conv2d）且 kernel_size>1（例如 3x3）
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.bias
        )

    # 递归替换子模块
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))

    return new_module

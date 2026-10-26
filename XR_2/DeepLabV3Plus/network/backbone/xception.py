import math
import torch.nn as nn


bn_mom = 0.0003


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=False, activate_first=True, inplace=True):
        """
        深度可分离卷积：
        = depthwise conv（按通道分组卷积） + BN + ReLU
          + pointwise 1x1 conv + BN + ReLU
        activate_first:
            True  时：先 ReLU 再 depthwise（Xception 里的常规写法）
            False 时：先 depthwise 再 ReLU，适配某些 block 结构。
        """
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)

        # depthwise：groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)

        # pointwise：1x1 卷积做通道变换
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)

        self.activate_first = activate_first

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)

        x = self.depthwise(x)
        x = self.bn1(x)

        if not self.activate_first:
            x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)

        if not self.activate_first:
            x = self.relu2(x)

        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1, atrous=None,
                 grow_first=True, activate_first=True, inplace=True):
        """
        Xception 中的基本残差块：
        - 3 个 SeparableConv2d 串联
        - 旁路（skip）做 1x1 投影或 identity
        - 支持空洞卷积 atrous（可以是 int 或 list[3]）
        """
        super(Block, self).__init__()

        if atrous is None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous = [atrous] * 3

        self.head_relu = True
        if out_filters != in_filters or strides != 1:
            # 输出通道 / 步长不一致时，用 1x1 conv 做 skip 投影
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip = None

        # 这一层的中间特征，用于 hook（低层特征）
        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters

        # 三个串联的可分离卷积
        self.sepconv1 = SeparableConv2d(in_filters, filters, 3, stride=1, padding=1 * atrous[0],
                                        dilation=atrous[0], bias=False, activate_first=activate_first,
                                        inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters, out_filters, 3, stride=1, padding=1 * atrous[1],
                                        dilation=atrous[1], bias=False, activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters, out_filters, 3, stride=strides, padding=1 * atrous[2],
                                        dilation=atrous[2], bias=False, activate_first=activate_first, inplace=inplace)

    def forward(self, inp):
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        # 这里保留中间特征，给主干外部拿低层特征用
        self.hook_layer = x
        x = self.sepconv3(x)

        x += skip
        return x


class Xception(nn.Module):
    """
    分割用的 Xception backbone（来自 bubbliiiing）：
    - entry flow: conv1, conv2, block1, block2, block3
    - middle flow: block4 ~ block19（默认 16 个 block）
    - exit flow: block20 + 三个 SeparableConv2d(conv3/4/5)

    本版本在原实现基础上增加：
        middle_blocks 参数，控制 middle flow 实际使用多少个 block（8 或 16），
        但仍然构建完整 16 个 block 名字，方便兼容已有预训练权重。
    """

    def __init__(self, downsample_factor=16, middle_blocks=16):
        """
        Args:
            downsample_factor: 输出特征的总 stride（8 或 16）
            middle_blocks:     中间流程重复多少个 Block（1~16，通常用 8 或 16）
        """
        super(Xception, self).__init__()

        if downsample_factor == 8:
            # 最后两个下采样位点改成 dilation，整体 OS=8
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            # 标准 Xception：OS=16
            stride_list = [2, 2, 1]
        else:
            raise ValueError("xception.py: output stride=%d is not supported (only 8 or 16)." % downsample_factor)

        # 允许把 middle_blocks 调到 8（轻量版）或 16（论文中较重的版本）
        self.middle_blocks = max(1, min(16, int(middle_blocks)))

        # -------- Entry flow --------
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        # 这里不立刻 ReLU，forward 里再做

        self.block1 = Block(64, 128, strides=2)
        self.block2 = Block(128, 256, strides=stride_list[0], inplace=False)
        self.block3 = Block(256, 728, strides=stride_list[1])

        # -------- Middle flow（最多 16 个 Block，都带 atrous=rate） --------
        rate = 16 // downsample_factor

        # 和原实现保持一致：先全部构建 block4 ~ block19
        self.block4 = Block(728, 728, 1, atrous=rate)
        self.block5 = Block(728, 728, 1, atrous=rate)
        self.block6 = Block(728, 728, 1, atrous=rate)
        self.block7 = Block(728, 728, 1, atrous=rate)

        self.block8 = Block(728, 728, 1, atrous=rate)
        self.block9 = Block(728, 728, 1, atrous=rate)
        self.block10 = Block(728, 728, 1, atrous=rate)
        self.block11 = Block(728, 728, 1, atrous=rate)

        self.block12 = Block(728, 728, 1, atrous=rate)
        self.block13 = Block(728, 728, 1, atrous=rate)
        self.block14 = Block(728, 728, 1, atrous=rate)
        self.block15 = Block(728, 728, 1, atrous=rate)

        # 下面这 4 个在原实现里写成 atrous=[rate, rate, rate]，效果与 atrous=rate 相同
        self.block16 = Block(728, 728, 1, atrous=[rate, rate, rate])
        self.block17 = Block(728, 728, 1, atrous=[rate, rate, rate])
        self.block18 = Block(728, 728, 1, atrous=[rate, rate, rate])
        self.block19 = Block(728, 728, 1, atrous=[rate, rate, rate])

        # 为了在 forward 里统一循环，这里收集一下中间所有 block
        self._middle_flow = [
            self.block4,
            self.block5,
            self.block6,
            self.block7,
            self.block8,
            self.block9,
            self.block10,
            self.block11,
            self.block12,
            self.block13,
            self.block14,
            self.block15,
            self.block16,
            self.block17,
            self.block18,
            self.block19, ]

        # -------- Exit flow --------
        self.block20 = Block(728, 1024, strides=stride_list[2], atrous=rate, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1,
                                     padding=1 * rate, dilation=rate, activate_first=False)
        self.conv4 = SeparableConv2d(1536, 1536, 3, 1,
                                     padding=1 * rate, dilation=rate, activate_first=False)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1,
                                     padding=1 * rate, dilation=rate, activate_first=False)

        # ------- 参数初始化（保持原实现风格） -------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # ----------------------------------------

    def forward(self, input):
        """
        返回：
            low_featrue_layer : 低层特征（来自 block2.sepconv2 的输出），给 decoder 做细节融合
            x                 : 高层特征（2048 通道），给 ASPP / decoder 使用
        """

        # ------ entry flow ------
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        low_featrue_layer = self.block2.hook_layer  # 低层特征（比如 H/4）

        x = self.block3(x)


        # ------ middle flow：只执行前 middle_blocks 个 block ------
        # 默认 middle_blocks=16 => 执行 block4~block19（原版行为）
        # 若 middle_blocks=8   => 只执行 block4~block11（轻量版）
        for idx, block in enumerate(self._middle_flow):
            if idx >= self.middle_blocks:
                break
            x = block(x)

        # ------ exit flow ------
        x = self.block20(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return low_featrue_layer, x



def xception(pretrained=False, downsample_factor=16, middle_blocks=16):
    """
    通用 Xception 工厂函数：
        middle_blocks:
            - 16: 对应“完整 Middle flow”（原始 DeepLab Xception 结构）
            - 8 :  对应“轻量版 Middle flow”，只执行 block4~block11
    """
    model = Xception(
        downsample_factor=downsample_factor,
        middle_blocks=middle_blocks,
    )
    return model


def xception8(pretrained=False, downsample_factor=16):
    """
    轻量版 Xception：
    - Middle flow 仅使用 8 个 block（block4~block11）
    - 可以配合 DeepLabv3+ 做“Xception-41”风格的主干
    """
    return xception(
        pretrained=pretrained,
        downsample_factor=downsample_factor,
        middle_blocks=8,
    )


def xception16(pretrained=False, downsample_factor=16):
    """
    完整版 Xception：
    - Middle flow 使用 16 个 block（block4~block19）
    - 对齐原始 bubbliiiing / DeepLabv3+ 论文中的较深主干
    """
    return xception(
        pretrained=pretrained,
        downsample_factor=downsample_factor,
        middle_blocks=16,
    )


class XceptionBackbone(nn.Module):
    """
    封装 Xception，使其 forward 返回 dict：
        {
            "out": 高层特征 (ASPP/decoder 输入),
            "low_level": 浅层特征 (decoder 融合)
        }
    同时提供 variant 参数，支持：
        - "xception8"  : 轻量版 middle_blocks=8
        - "xception16" : 完整版 middle_blocks=16（默认）
    """

    def __init__(self,
                 output_stride=16,
                 pretrained_backbone=False,
                 variant="xception16"):
        super().__init__()

        if output_stride not in (8, 16):
            raise ValueError(
                f"XceptionBackbone 目前只支持 output_stride=8 或 16, 但得到 {output_stride}" )

        # 选择具体的 Xception 变体
        if variant == "xception8":
            backbone_fn = xception8
        elif variant in ("xception16", "xception"):
            backbone_fn = xception16
        else:
            raise ValueError(f"Unknown Xception variant: {variant}")

        # 调用本文件里的工厂函数构建 backbone
        self.backbone = backbone_fn(
            downsample_factor=output_stride,
            pretrained=pretrained_backbone,
        )

    def forward(self, x):
        # 注意：这里假设 Xception.forward 返回的是 (low_level, high_level)
        low_level, high_level = self.backbone(x)

        return {
            "out": high_level,
            "low_level": low_level,
        }
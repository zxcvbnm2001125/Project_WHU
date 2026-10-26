from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHeadV3Plus, DeepLabV3Plus
from .backbone import resnet, mobilenetv2, xception


def _segm_resnet(backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    构建 ResNet backbone 的 DeepLabV3+。

    output_stride(OS) 的实现要点：
    - OS=16：只取消 layer4 的 stride=2（用 dilation 替代） -> replace_stride_with_dilation=[False, False, True]
    - OS=8 ：取消 layer3/layer4 的 stride=2（用 dilation 替代） -> replace_stride_with_dilation=[False, True, True]

    同时根据 OS 切换 ASPP 的 atrous rates：
    - OS=16： [6, 12, 18]
    - OS=8 ： [12, 24, 36]
    """
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    # 构建 backbone（resnet50/resnet101），并配置 dilation 替换策略
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    # ResNet layer4 输出通道固定 2048；layer1 输出通道 256（Bottleneck expansion=4）
    inplanes = 2048
    low_level_planes = 256

    # DeepLabV3+：需要 out + low_level
    return_layers = {"layer4": "out", "layer1": "low_level"}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    # 让 backbone 输出变成 dict：{'out': ..., 'low_level': ...}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3Plus(backbone, classifier)
    return model


def _segm_xception(backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    构建基于 Xception 主干的 DeepLabV3+ 模型。

    支持三种字符串：
        - 'xception'   : 默认等价于使用完整 Middle flow 的 'xception16'
        - 'xception16' : Middle flow 使用 16 个 block（论文中较深的版本）
        - 'xception8'  : Middle flow 使用 8 个 block（相对轻量版）

    参数：
        backbone_name: 见上
        num_classes:   类别数（含背景）
        output_stride: 输出步长，8 或 16
        pretrained_backbone: 是否加载（ImageNet / VOC）预训练的 Xception 主干
    """

    # 1) ASPP 的空洞率设置，和 ResNet / MobileNet 版本保持一致
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:  # output_stride == 16
        aspp_dilate = [6, 12, 18]

    # 2) 根据 backbone_name 选择具体的 Xception 变体
    #    - 'xception' / 'xception16' -> variant='xception16'
    #    - 'xception8'               -> variant='xception8'
    if backbone_name in ("xception", "xception16"):
        variant = "xception16"
    elif backbone_name == "xception8":
        variant = "xception8"
    else:
        raise ValueError(f"Unknown xception backbone_name: {backbone_name}")

    # 构建 Xception 主干（封装后 forward 返回 dict：{"out", "low_level"}）
    backbone = xception.XceptionBackbone(
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        variant=variant,
    )

    # 根据 Xception 实现：
    #   - high-level 特征：通道数 2048
    #   - low-level 特征：通道数 256
    inplanes = 2048
    low_level_planes = 256

    # 3) DeepLabV3+ 的 head（ASPP + Decoder）
    classifier = DeepLabHeadV3Plus(
        in_channels=inplanes,
        low_level_channels=low_level_planes,
        num_classes=num_classes,
        aspp_dilate=aspp_dilate,
    )

    # 4) 封装成统一的 DeepLabV3Plus 模型外壳
    # 注意：这里不再用 IntermediateLayerGetter，
    # 因为 XceptionBackbone 自己已经返回 dict 了
    model = DeepLabV3Plus(backbone, classifier)
    return model


def _segm_mobilenet(backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    构建 MobileNetV2 backbone 的 DeepLabV3+。

    mobilenet_v2 内部支持 output_stride：
    - 到达目标 OS 后停止继续下采样（stride->1），改用 dilation 累积。
    """
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(
        pretrained=pretrained_backbone,
        output_stride=output_stride,
    )

    # 关键技巧：把 features 切成 low/high 两段，供 V3+ decoder 使用
    # low_level_features: features[0:4]  （一般是 stride=4 左右的浅层细节）
    # high_level_features: features[4:-1]（更深层语义）
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    # mobilenetv2 的 high_level 最终输出通道 320，low_level 输出通道 24
    inplanes = 320
    low_level_planes = 24

    return_layers = {"high_level_features": "out", "low_level_features": "low_level"}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3Plus(backbone, classifier)
    return model


def _load_model(backbone, num_classes, output_stride, pretrained_backbone):
    """
    统一入口：根据 backbone 字符串选择不同的 DeepLabV3+ 组装函数。
    已支持：
        - 'resnet50' / 'resnet101'
        - 'mobilenetv2'
        - 'xception'   / 'xception16'（完整 Middle flow）
        - 'xception8'  （轻量 Middle flow）
    """
    if backbone == "mobilenetv2":
        model = _segm_mobilenet(
            backbone, num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone.startswith("resnet"):
        model = _segm_resnet(
            backbone, num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone in ("xception", "xception16", "xception8"):
        model = _segm_xception(
            backbone, num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        raise NotImplementedError(f"Backbone {backbone} is not supported.")
    return model


# --------------------------
# 只保留 Deeplab v3+ 工厂函数
# --------------------------
def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model(
        "resnet50", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model(
        "resnet101", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=False):
    return _load_model(
        "mobilenetv2", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_xception8(num_classes=21, output_stride=8, pretrained_backbone=False):
    """
    Xception 轻量版（middle_blocks = 8）对应的 DeepLabV3+。
    """
    return _load_model(
        "xception8", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_xception16(num_classes=21, output_stride=8, pretrained_backbone=False):
    """
    Xception 完整版（middle_blocks = 16）对应的 DeepLabV3+。
    """
    return _load_model(
        "xception16", num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )



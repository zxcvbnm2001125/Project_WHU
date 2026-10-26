from collections import OrderedDict

from typing import Dict

from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50


class IntermediateLayerGetter(nn.ModuleDict):
    # 版本信息，表示当前类的版本为2
    _version = 2
    # 用于注解返回层的字典类型
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        """
                初始化 IntermediateLayerGetter 类

                Args:
                    model (nn.Module): 输入的模型，用于从中提取中间层的特征
                    return_layers (Dict[str, str]): 一个字典，指定要返回的层及其对应的输出名称
        """

        # 检查 return_layers 中的层名称是否都在模型的子模块中存在
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        # 保存原始的 return_layers 字典，防止修改
        orig_return_layers = return_layers
        # 将字典的键和值转换为字符串类型
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建 backbone（骨干网络），仅保留需要的层
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            # 如果当前层是需要返回的层，则删除它
            if name in return_layers:
                del return_layers[name]
            # 如果所有需要的层已经被处理完，停止循环
            if not return_layers:
                break

        # 使用 nn.ModuleDict 初始化，保存需要的层
        super(IntermediateLayerGetter, self).__init__(layers)
        # 保存原始的返回层字典
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
                定义前向传播操作，提取指定的中间层特征

                Args:
                    x (Tensor): 输入数据，通常是一个图像或特征图

                Returns:
                    Dict[str, Tensor]: 返回一个字典，包含中间层的特征
        """
        # 创建一个 OrderedDict 用来存储中间层的输出
        out = OrderedDict()
        # 对模型的每一层进行前向传播
        for name, module in self.items():
            # 将输入 x 传递给当前层
            x = module(x)
            # 如果当前层在需要返回的层列表中，将其输出存入字典中
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        # 计算中间通道数，通常将输入通道数减少到四分之一
        inter_channels = in_channels // 4

        # 定义分类头部的层
        layers = [
            # 第一个卷积层：将输入通道数in_channels减少为inter_channels，使用3x3卷积，padding=1，确保输出大小与输入相同
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),

            # 批归一化：标准化卷积输出，增加训练稳定性
            nn.BatchNorm2d(inter_channels),

            # ReLU激活函数：引入非线性，使模型能够学习复杂的函数映射
            nn.ReLU(),

            # Dropout：防止过拟合，随机丢弃10%的神经元
            nn.Dropout(0.1),

            # 第二个卷积层：将中间通道数inter_channels减少到最终的输出通道数channels，使用1x1卷积来减少特征图的维度
            nn.Conv2d(inter_channels, channels, 1)
        ]

        # 使用nn.Sequential将所有层组合起来，形成一个顺序执行的模型
        super(FCNHead, self).__init__(*layers)


class FCN(nn.Module):
    def __init__(self, backbone, classifier):
        """
                初始化 FCN 网络

                Args:
                    backbone (nn.Module): 用于提取特征的网络（例如，ResNet）
                    classifier (nn.Module): 用于生成密集预测（逐像素分类）的模块
        """
        super(FCN, self).__init__()
        self.backbone = backbone  # 特征提取网络
        self.classifier = classifier  # 主分类器

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
                执行前向传播

                Args:
                    x (Tensor): 输入图像，通常形状为 (batch_size, channels, height, width)

                Returns:
                    Dict[str, Tensor]: 包含主分类器和（如果有的话）辅助分类器输出的字典
        """
        # 获取输入的空间尺寸（高度和宽度）
        input_shape = x.shape[-2:]
        # 获取 backbone 提取的特征，这里返回的是一个字典（包括 "out"）
        features = self.backbone(x)

        # 初始化返回结果字典
        result = OrderedDict()
        # 获取主分类器输出（从 backbone 的 "out" 特征图开始）
        x = features["out"]
        x = self.classifier(x)  # 通过主分类器生成预测

        # 使用双线性插值将输出调整到与输入相同的尺寸
        # 虽然原论文中使用的是 ConvTranspose2d，但由于权重被冻结，实际上这就是一个双线性插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        # 将主分类器输出添加到结果字典中
        result["out"] = x
        return result


def fcn_resnet50(num_classes=21):
    # 使用ResNet50作为FCN的骨干网络
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    # 设置ResNet50最后一层的输出特征维度
    out_inplanes = 2048

    # 决定返回哪些层（提取特定层的输出）
    return_layers = {'layer4': 'out'}

    # 使用IntermediateLayerGetter来从ResNet50中提取所需的层
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # 创建主分类器
    classifier = FCNHead(out_inplanes, num_classes)

    # 将骨干网络、主分类器结合成一个完整的FCN模型
    model = FCN(backbone, classifier)

    return model


# from torchsummary import summary
#
# model = fcn_resnet50(num_classes=21)  # 替换为你的模型
# summary(model, (3, 480, 480))  # 假设输入为 224x224 RGB 图像



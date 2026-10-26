import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class _SimpleSegmentationModel(nn.Module):
    """
    语义分割模型的通用 wrapper（本工程中用于 DeepLabV3+）

    流程：
      1) features = backbone(x)  -> 返回 dict，例如 {'out': ..., 'low_level': ...}
      2) x = classifier(features) -> head 输出 logits
      3) logits 上采样回输入大小，得到最终分割输出
    """
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    从 backbone 中按“子模块名”提取中间层输出，返回一个 OrderedDict。

    使用方式（在 modeling.py 中配置 return_layers）：
      - ResNet:   {'layer4': 'out', 'layer1': 'low_level'}
      - Xception: {'conv4': 'out',  'block1': 'low_level'}
      - MobileNet:{'high_level_features': 'out', 'low_level_features': 'low_level'}

    注意：
      - 只能抓 model 的“直接子模块”（named_children）名称。
      - return_layers 必须是 model.named_children() 的子集，否则报错。
      - 会在收集到所有目标层后提前 break，避免不必要的 forward 计算。
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

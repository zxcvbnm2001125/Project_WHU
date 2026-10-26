import torch
import torch.nn as nn
try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url


# 只保留 DeepLab 常用的两个 backbone：resnet50 / resnet101
__all__ = ['ResNet', 'resnet50', 'resnet101']


# 只保留这两个权重下载地址（ImageNet-1K 预训练）
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """
    3x3 卷积（支持 dilation 空洞卷积）：
    - padding=dilation：保证 stride=1 时空间尺寸不变
    - dilation>1：扩大感受野但不下采样（DeepLab 的 atrous conv）
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积：用于通道变换或 shortcut 下采样对齐。"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    ResNet50/101 使用的 bottleneck 残差块：
      1x1 -> 3x3 -> 1x1，expansion=4
    中间 3x3 conv2 支持 stride 和 dilation（DeepLab 的关键）。
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        # 这里的 stride/dilation 决定是否下采样、是否空洞
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut 分支对齐（尺寸/通道不一致时）
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet Backbone（用于 DeepLab 的关键：replace_stride_with_dilation）

    标准分类 ResNet 最终输出步长 OS=32：
      conv1 stride=2, maxpool stride=2 => /4
      layer2 stride=2 => /8
      layer3 stride=2 => /16
      layer4 stride=2 => /32

    DeepLab 常用做法（在 modeling.py 里配置）：
      - OS=16：replace_stride_with_dilation=[False, False, True]
              => layer4 取消 stride=2，用 dilation=2 替代
      - OS=8 ：replace_stride_with_dilation=[False, True, True]
              => layer3/layer4 取消 stride=2，用 dilation=2/4 替代
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple/list")

        # stem: /2 -> /4
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4个 stage（layer1~layer4）
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 分类头：分割任务一般不会用到，但保留以便加载 ImageNet 预训练权重
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化（保持原风格）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 可选：让 residual 初始更接近 identity（分类论文技巧，保留）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        DeepLab 的关键逻辑：dilate=True 时把 stride=2 替换成 dilation
        - 取消下采样：stride=1
        - 增大空洞率：self.dilation *= 原 stride（通常乘2）
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        # shortcut 对齐：stride 或通道不一致时走 downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # stage 的第一个 block 负责 stride/shortcut 对齐
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 后续 block 使用更新后的 self.dilation
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头（分割一般不会用到，但保留以便加载预训练权重）
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(arch, layers, pretrained, progress, **kwargs):
    """构建 resnet 并可选加载 ImageNet 预训练权重（保持原风格写法）。"""
    model = ResNet(Bottleneck, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    """ResNet-50: Bottleneck x [3, 4, 6, 3]"""
    return _resnet('resnet50', [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """ResNet-101: Bottleneck x [3, 4, 23, 3]"""
    return _resnet('resnet101', [3, 4, 23, 3], pretrained, progress, **kwargs)

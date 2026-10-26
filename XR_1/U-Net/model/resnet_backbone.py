import math
import torch.nn as nn



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding

    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        stride (int): 卷积步幅，默认为 1
        groups (int): 卷积分组数，默认为 1
        dilation (int): 卷积扩张系数，默认为 1

    Returns:
        nn.Conv2d: 定义的 3x3 卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution
    Args:
        in_planes (int): 输入通道数
        out_planes (int): 输出通道数
        stride (int): 步幅，默认为 1

    Returns:
        nn.Conv2d: 定义的 1x1 卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
                        初始化 Bottleneck 模块

                        Args:
                            inplanes (int): 输入通道数
                            planes (int): 输出通道数（卷积层的通道数）
                            stride (int): 卷积步幅，默认为 1
                            downsample (nn.Module, optional): 用于调整输入尺寸的层（例如：当步幅不为 1 时，需要下采样）
                            groups (int): 分组卷积的数量，默认为 1
                            base_width (int): 基础宽度，影响每层的宽度
                            dilation (int): 卷积扩张系数，默认为 1
                            norm_layer (nn.Module, optional): 归一化层类型，默认为 `nn.BatchNorm2d`
        """
        super(Bottleneck, self).__init__()

        # 如果没有提供 norm_layer，则使用 BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 计算宽度，groups 会影响每个卷积层的通道数
        width = int(planes * (base_width / 64.)) * groups

        # 第一层 1x1 卷积，作用是降维
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)  # 归一化层
        # 第二层 3x3 卷积，作用是进行空间下采样
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)  # 归一化层
        # 第三层 1x1 卷积，作用是升维
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)  # 归一化层

        # 激活函数 ReLU
        self.relu = nn.ReLU(inplace=True)

        # 下采样层，默认为 None
        self.downsample = downsample
        # 存储步幅
        self.stride = stride

    def forward(self, x):
        """
                        正向传播

                        Args:
                            x (Tensor): 输入张量

                        Returns:
                            Tensor: 输出张量
        """
        identity = x  # 存储输入张量，用于跳跃连接（residual connection）

        # 通过第一个 1x1 卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 通过第二个 3x3 卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 通过第三个 1x1 卷积层
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 跳跃连接：将输入（identity）与输出相加
        out += identity

        # 激活函数（ReLU）再次作用在输出上
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        # 设置初始通道数，供残差模块使用（后续根据 expansion 自动更新）
        self.inplanes = 64

        # 输入通道数为 3，输出为 64，7x7 大卷积核，步长为 2，padding 为 3，不使用 bias,为一个CBR模块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 最大池化层，3x3 核，步长为 2，无 padding，向上取整模式
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change

        # 构建四个 stage 的残差网络，每个 stage 包含 layers[i] 个残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 平均池化，用于分类场景下整合空间信息（此实现中未启用）
        self.avgpool = nn.AvgPool2d(7)
        # 分类输出全连接层：输入通道为最后一层的输出通道（含 expansion），输出为类别数
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 模块初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He 初始化：N(0, sqrt(2/n))，适合 ReLU 激活函数
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # BN 权重初始化为 1，偏置初始化为 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
                构建一个残差层（stage），由多个残差块 block 组成。
                - block: 残差块类型（如 Bottleneck 或 BasicBlock）
                - planes: 当前 stage 的基础输出通道数
                - blocks: 当前 stage 包含的残差块数量
                - stride: 第一个 block 的步长（用于空间下采样）
        """
        downsample = None
        # 如果输入通道与输出通道不一致，或 stride ≠ 1，则需要下采样对齐
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # 第一个 block：可能包含下采样和通道数调整
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 更新当前输入通道数为新的输出通道数
        self.inplanes = planes * block.expansion

        # 其余 block：不下采样，通道保持一致
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("输入数据：", x.shape)
        """
                定义前向传播过程，输出多个阶段的特征图（适合用于下游任务如 FPN、检测、分割等）
        """
        x = self.conv1(x)
        x = self.bn1(x)
        feat1 = self.relu(x)  # 输出特征图
        # print("feat1:", feat1.shape)

        x = self.maxpool(feat1)
        # print("池化层：", x.shape)

        feat2 = self.layer1(x)  # stage 1 输出
        feat3 = self.layer2(feat2)  # stage 2 输出
        feat4 = self.layer3(feat3)   # stage 3 输出
        feat5 = self.layer4(feat4)   # stage 4 输出

        # 返回所有层的中间特征图（而非分类结果），常用于特征提取任务
        return [feat1, feat2, feat3, feat4, feat5]



def resnet50(**kwargs):
    # 构建一个 ResNet-50 模型实例，使用 Bottleneck 结构
    # 每个 stage 分别包含 3, 4, 6, 3 个残差块（符合 ResNet-50 配置）
    # 额外参数通过 kwargs 传入 ResNet 构造函数（如 num_classes, input_shape 等）
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    # 删除全局平均池化层（avgpool）和全连接层（fc），以去除分类模块
    # 模型将仅保留主干部分（stem + stage1~stage4），适合特征提取任务
    del model.avgpool
    del model.fc

    # 返回去除分类头的 ResNet-50 模型，用于下游任务（如检测、分割）
    return model

import torch
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
        self.bn1 = norm_layer(width) # 归一化层
        # 第二层 3x3 卷积，作用是进行空间下采样
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width) # 归一化层
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

        # 如果存在下采样层，则对输入进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 跳跃连接：将输入（identity）与输出相加
        out += identity

        # 激活函数（ReLU）再次作用在输出上
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """
               初始化 ResNet 模型

               Args:
                   block (nn.Module): 残差块类型，如 Bottleneck 或 BasicBlock
                   layers (list): 每个阶段包含的残差块数量
                   num_classes (int): 输出类别数，默认为 1000（ImageNet 分类）
                   zero_init_residual (bool): 如果为 True，则初始化残差分支的最后 BatchNorm 权重为零
                   groups (int): 卷积分组数，默认为 1
                   width_per_group (int): 每个组的宽度，默认为 64
                   replace_stride_with_dilation (list): 指定哪些卷积层使用扩张卷积替代步幅卷积
                   norm_layer (nn.Module): 归一化层类型，默认为 `nn.BatchNorm2d`
        """
        super(ResNet, self).__init__()

        # 如果没有提供 norm_layer，则默认使用 BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 初始化输入通道数
        self.inplanes = 64
        self.dilation = 1

        # 如果没有传入 replace_stride_with_dilation，则设置默认值
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        # 确保传入的 replace_stride_with_dilation 列表长度为 3
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group


        # 第一层卷积：7x7 卷积，步幅为 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建各个阶段的残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 自适应平均池化（AdaptiveAvgPool），将特征图的尺寸缩放到 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，将特征图展平并映射到类别数
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化所有卷积层和批归一化层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 如果需要零初始化残差分支的最后一个 BatchNorm 层
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
                构建一个包含多个残差块的阶段

                Args:
                    block (nn.Module): 残差块类型，如 Bottleneck
                    planes (int): 每个阶段输出的通道数
                    blocks (int): 当前阶段包含的残差块数量
                    stride (int): 步幅，默认为 1
                    dilate (bool): 是否使用扩张卷积，默认为 False

                Returns:
                    nn.Sequential: 由多个残差块组成的序列
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # 如果使用扩张卷积，将步幅设置为 1，并调整 dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # 如果步幅不为 1 或者输入和输出通道数不同，则需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []

        # 添加第一个残差块
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        # 返回由多个残差块组成的序列
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """
                正向传播的实现

                Args:
                    x (Tensor): 输入张量

                Returns:
                    Tensor: 输出张量
        """
        # 依次通过各个层进行前向计算
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 通过自适应平均池化将特征图的大小变为 1x1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.fc(x)  # 全连接层进行分类

        return x

    def forward(self, x):
        """
                调用 _forward_impl 执行前向传播

                Args:
                    x (Tensor): 输入张量

                Returns:
                    Tensor: 输出张量
        """
        return self._forward_impl(x)



def _resnet(block, layers, **kwargs):
    """
       构建一个 ResNet 模型

       Args:
           block (nn.Module): 用于构建 ResNet 的基本模块。常见的有 Bottleneck（例如 ResNet-50）和 BasicBlock（例如 ResNet-18）
           layers (list): 每个阶段中使用多少个 block（例如 [3, 4, 6, 3] 表示 ResNet-50）
           **kwargs: 额外的关键字参数，会传递给 ResNet 类的构造函数（例如是否加载预训练权重）

       Returns:
           model (nn.Module): 构建好的 ResNet 模型
       """
    # 使用给定的 block 和 layers 参数创建一个 ResNet 模型
    model = ResNet(block, layers, **kwargs)

    # 返回创建好的模型
    return model


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 调用 _resnet 函数来构建 ResNet-50 模型，指定使用 Bottleneck 模块和特定的层数
    # Bottleneck: ResNet-50 的基本构建块
    # [3, 4, 6, 3]: 表示 ResNet-50 每个阶段使用的 Bottleneck 块的数量
    # kwargs: 额外的参数（例如 pretrained 和 progress）会被传递给 _resnet 函数
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)

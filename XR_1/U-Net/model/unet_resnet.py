import torch
import torch.nn as nn

from model.resnet_backbone import resnet50

# 定义一个 U-Net 解码模块（上采样模块）
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        """
               构造函数
               参数：
               - in_size: 输入通道数（等于上采样特征图通道数 + 跳跃连接特征图通道数）
               - out_size: 输出通道数（经过卷积后的输出通道数）
        """
        super(unetUp, self).__init__()
        # 第一层卷积：输入通道为 in_size，输出通道为 out_size，卷积核大小 3x3，padding=1 保持尺寸不变
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # 第二层卷积：输入通道和输出通道均为 out_size，进一步特征提取
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        # 上采样层：使用双线性插值将特征图放大两倍（scale_factor=2）
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # ReLU 激活函数，inplace=True 表示原地操作，节省内存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        """
               前向传播
               参数：
               - inputs1: 编码器部分通过跳跃连接传来的特征图（高分辨率）
               - inputs2: 来自上一级解码器的输出（低分辨率，需要上采样）
        """
        # 先对 inputs2 进行上采样，然后与 inputs1 在通道维度上进行拼接
        # 拼接后的通道数为 in_size
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # 第一次卷积 + ReLU
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        # 第二次卷积 + ReLU
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        # 返回该解码模块的输出特征图
        return outputs


# 定义 U-Net 主体结构
class Unet(nn.Module):
    def __init__(self, num_classes=21):
        """
                构造函数
                参数：
                - num_classes: 最终分类的类别数（用于语义分割任务）
        """
        super(Unet, self).__init__()

        # 使用 ResNet50 作为编码器，提取多尺度特征
        self.resnet = resnet50()  # 假设 resnet50() 返回 5 层特征：feat1 ~ feat5
        # 编码器输出的特征通道数（每层输出的特征图通道数）
        in_filters = [192, 512, 1024, 3072]   # 通常是低到高的顺序（层级）
        # 解码器每一层希望恢复到的输出通道数
        out_filters = [64, 128, 256, 512]   # 解码层输出通道数逐步降低

        # 定义 4 层上采样模块（从深到浅）
        # 每层通过双线性插值上采样 + 拼接 + 两次卷积 + ReLU
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])  # 最深层输出
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])  # 次深层
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])  # 中间层
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])  # 最浅层

        # 对最后一层解码器的输出再上采样一倍并做两次卷积处理，进一步提升分辨率
        self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),   # 上采样
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),  # 卷积
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),  # 卷积
                nn.ReLU(),
            )
        # 最后一层卷积，用于生成最终每类像素的得分图（通道数=num_classes）
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        """
                前向传播
                参数：
                - inputs: 输入图像（通常为 RGB 图像，形状为 B×3×H×W）

                返回：
                - final: 每个像素点在 num_classes 个类别上的预测（未Softmax）
        """
        # 编码器提取五层特征图（假设顺序为由浅到深）
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # 解码过程：由深到浅依次上采样 + 融合跳跃连接特征
        up4 = self.up_concat4(feat4, feat5)  # 使用 feat5 上采样后与 feat4 融合
        up3 = self.up_concat3(feat3, up4)   # 再将上一层上采样后与 feat3 融合
        up2 = self.up_concat2(feat2, up3)   # 同理
        up1 = self.up_concat1(feat1, up2)   # 最浅层融合

        # 可选的最终进一步上采样（恢复到输入图尺寸）
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        # 最终生成每类的预测图
        final = self.final(up1)
        return final

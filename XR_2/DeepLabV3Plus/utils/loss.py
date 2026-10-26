import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    """
    Focal Loss（常用于类别不均衡）：
      FL = alpha * (1 - pt)^gamma * CE
    - pt = exp(-CE) 代表预测为真类的概率（近似）
    - gamma 越大：越强调“难样本”
    - ignore_index：忽略某些像素（VOC常用 255 表示 ignore）

    main.py 中 opts.loss_type == 'focal_loss' 时会用它
    """
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) logits
        targets: (N, H, W) 标签（每个像素一个类别 id）
        """
        # 逐像素 CE，先不做 mean/sum（reduction='none'）
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)

        # pt = exp(-CE)
        pt = torch.exp(-ce_loss)

        # focal reweight：难样本（pt 小）权重更大
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        # size_average=True：取 mean；否则 sum
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DiceLoss(nn.Module):
    """
    逐类软 Dice，与 U-Net / SegFormer 的 Dice 同一套公式（消融 B 口径一致）。

    inputs : (N, C, H, W) logits
    targets: (N, H, W) 类别 id，ignore_index 的像素不参与任何类的 tp/fp/fn
    返回   : 1 - mean_over_classes(Dice)
    """

    def __init__(self, num_classes, ignore_index=255, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        if inputs.shape[2:] != targets.shape[1:]:
            inputs = F.interpolate(inputs, size=targets.shape[1:],
                                   mode="bilinear", align_corners=True)

        probs = torch.softmax(inputs, dim=1)

        # one-hot 多留一列给忽略像素，取 [:, :-1] 后与 probs 的 C 个通道对齐
        t = targets.clone()
        t[t == self.ignore_index] = self.num_classes
        onehot = F.one_hot(t.long(), self.num_classes + 1).permute(0, 3, 1, 2).float()

        # 忽略像素既不进分子也不进分母，否则会被当作 FP 计入惩罚
        probs = probs * (targets != self.ignore_index).unsqueeze(1).float()

        tp = (onehot[:, :-1] * probs).sum(dim=(0, 2, 3))
        fp = probs.sum(dim=(0, 2, 3)) - tp
        fn = onehot[:, :-1].sum(dim=(0, 2, 3)) - tp

        score = (2 * tp + self.smooth) / (2 * tp + fn + fp + self.smooth)
        return 1 - score.mean()


class CombinedLoss(nn.Module):
    """基础损失 + Dice，权重固定 1:1（与 U-Net 的 CE+Dice 口径一致）。"""

    def __init__(self, base_loss, dice_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.base_loss = base_loss
        self.dice_loss = dice_weight

    def set_num_classes(self, num_classes):
        """由 build_criterion 注入类别数后重建 Dice 项。"""
        self.dice = DiceLoss(num_classes)
        return self

    def forward(self, inputs, targets):
        return self.base_loss(inputs, targets) + self.dice_loss * self.dice(inputs, targets)

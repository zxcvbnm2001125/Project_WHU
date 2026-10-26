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

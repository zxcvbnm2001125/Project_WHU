from torch.optim.lr_scheduler import _LRScheduler, StepLR


class PolyLR(_LRScheduler):
    """
    Poly 学习率衰减（分割里非常常用）：
      lr = base_lr * (1 - iter/max_iters)^power
    并设置 min_lr 下限，避免最后 lr 变成 0。

    main.py 中：
      scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    且每次迭代都 scheduler.step()（iter-based）
    """

    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        计算每个 param group 的 lr（支持 backbone/classifier 不同 base_lr）。
        self.last_epoch 在这里实际表示“当前迭代数”（因为每 iter 都 step 一次）。
        """
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

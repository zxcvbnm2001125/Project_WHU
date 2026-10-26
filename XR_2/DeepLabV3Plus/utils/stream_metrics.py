import numpy as np


# 本文件用于语义分割指标的“流式计算”：累计混淆矩阵 -> 计算 Acc / mIoU 等。
# 说明：本版本为“最终精简版 + 保留必要注释”，仅移除了未使用的 sklearn import。


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        # 从累计混淆矩阵计算：
        # Overall Acc：总体像素准确率
        # Mean Acc：每类像素准确率的平均
        # Mean IoU：每类 IoU 的平均（最常用）
        # FreqW Acc：按类频率加权的 IoU
        # Class IoU：每个类别的 IoU（字典）
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    # StreamSegMetrics：维护 n_classes x n_classes 的混淆矩阵 confusion_matrix
    # 行=真实类别，列=预测类别；每次 update 把当前 batch 的统计累加进去。
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        # label_trues/label_preds 通常是 list/iterable，每个元素形状为 (H, W)。
        # 这里逐张图 flatten 后用 _fast_hist 统计并累加到总混淆矩阵。
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        # string+='Class IoU:\n'
        # for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        # 快速构建当前样本的混淆矩阵：
        # - mask 过滤掉非法标签（<0 或 >=n_classes）
        # - bincount 统计 (true, pred) 组合的出现次数，再 reshape 成 (n_classes, n_classes)。
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix

        # 如果整个 hist 都是 0，说明压根没收到有效样本，直接返回全 0，避免 0/0
        if hist.sum() == 0:
            return {
                "Overall Acc": 0.0,
                "Mean Acc": 0.0,
                "FreqW Acc": 0.0,
                "Mean IoU": 0.0,
                "Class IoU": {i: 0.0 for i in range(self.n_classes)},
            }

        # 使用 errstate 屏蔽除零 warning，结果里可能会出现 inf/nan，后面再处理
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.diag(hist).sum() / hist.sum()
            acc_cls = np.diag(hist) / hist.sum(axis=1)
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )

        # 平均精度和平均 IoU：用 nanmean 跳过无效类别
        acc_cls = np.nanmean(acc_cls)
        mean_iu = np.nanmean(iu)

        # 频权 IoU
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # 每类 IoU：把 nan 转成 0，避免后面使用时报错
        iu = np.nan_to_num(iu)
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }


    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    # AverageMeter：通用均值统计器（按 id/key 维护 sum 与 count），
    # 常用于统计 loss / 时间等（与 IoU 计算无关，但训练里常用）。
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]

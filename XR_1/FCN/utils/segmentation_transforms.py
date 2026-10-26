from utils import transforms as T


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 计算最小和最大尺寸，用于随机调整图像的大小
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # 创建一个转换列表，首先进行随机缩放
        trans = [T.RandomResize(min_size, max_size)]

        # 如果水平翻转的概率大于0，则加入随机水平翻转操作
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        # 添加其他常见的转换操作，包括随机裁剪、将图像转换为张量和标准化
        trans.extend([
            T.RandomCrop(crop_size),  # 随机裁剪，确保输出尺寸一致
            T.ToTensor(),  # 将图像转换为Tensor（PyTorch所需的数据格式）
            T.Normalize(mean=mean, std=std),  # 图像标准化
        ])

        # 使用Compose将所有转换组合成一个操作流水线
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        # 当调用该类时，执行转换操作
        return self.transforms(img, target)


class SegmentationPresetEval:
    # 初始化时设置预处理操作
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),  # 将图像调整到统一的尺寸
            T.ToTensor(),  # 将图像转换为Tensor
            T.Normalize(mean=mean, std=std),  # 对图像进行标准化
        ])

    def __call__(self, img, target):
        # 调用时进行图像预处理
        return self.transforms(img, target)


def get_transform(train):
    # WHU 子集瓦片本身就是 512×512，缩放与裁剪都对齐到 512，
    # 与 U-Net / DeepLabV3+ / SegFormer 的输入口径保持一致（消融要求同 crop）
    base_size = 512  # 基本尺寸，图像将被缩放至至少这个大小
    crop_size = 512  # 裁剪尺寸，图像最终会被裁剪到这个尺寸（在训练时）

    # 如果是训练模式，返回训练时使用的预处理（包括缩放、裁剪等）
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)

"""
SegFormer 模型包。

    mit.py           MiT 编码器（B0/B1/B2/B3），不依赖 timm，可直接加载 mmseg 命名的主干权重
    segformer.py     All-MLP 解码器 + 完整 SegFormer（含可选的 1/2 分辨率精修）
    unet_training.py 损失函数与学习率调度（与 U-Net / DeepLabV3+ 项目共用同一份实现）

刻意不在这里做 re-export：train.py / val.py / predict.py 统一写
`from model.segformer import SegFormer`，路径一目了然，也避免包初始化时引入循环导入。
"""

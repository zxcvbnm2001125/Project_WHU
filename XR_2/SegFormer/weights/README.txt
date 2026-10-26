这个目录放 MiT 主干的 ImageNet 预训练权重（train.py / val.py / predict.py 不会自动下载）。

推荐的文件名（train.py 的 --backbone-weights 留空时会自动按这个名字去找）：

    mit_b0_imagenet.pth
    mit_b1_imagenet.pth
    mit_b2_imagenet.pth

获取方式（二选一）：

1) 自动下载并转换（需要能连上 huggingface.co 或 hf-mirror.com）
       python download_pretrained.py --variants b1
       python download_pretrained.py --variants b0 b1 b2 --endpoint https://hf-mirror.com

   它会把 HuggingFace 的 nvidia/mit-bN 权重转成 mmseg 风格的命名，
   并把拆开的 key/value 投影合并成本项目使用的融合 kv，最后存成本目录下的
   mit_bN_imagenet.pth。

2) 手动下载 NVlabs 官方权重后本地转换
   https://github.com/NVlabs/SegFormer  →  README 里的 mit_b0.pth / mit_b1.pth / mit_b2.pth
       python download_pretrained.py --variants b1 --from-local mit_b1.pth

不下载预训练权重也能训（--backbone-weights 指向不存在的文件时会提示并随机初始化），
但小数据集从零训 Transformer 收敛慢、精度也会掉，建议还是把权重补齐。

注意：训练产物（best_model_2.pth / last_model_2.pth / checkpoint_2.pth）
不在这个目录，而在 run/train/exp*/weights/ 下；这里的 mit_*.pth 是输入不是输出。

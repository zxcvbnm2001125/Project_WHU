# 导入标准库和第三方库
import torch  # 导入PyTorch库
from torch.utils.data import DataLoader  # 导入数据加载器模块
import time

# 导入自定义模块和模型
from model.unet_resnet import Unet  # 导入U-Net模型
from utils.dataloader import UnetDataset, unet_dataset_collate  # 导入U-Net数据集及其合并函数
from utils.train_and_eval import pixel_accuracy, mean_accuracy, mean_iou, frequency_weighted_iou

class LogColor:
    # 定义终端输出的颜色常量，用于日志信息的彩色显示
    GREEN = "\033[1;32m"  # 绿色
    YELLOW = "\033[1;33m"  # 黄色
    RED = "\033[1;31m"  # 红色
    RESET = "\033[0m"  # 重置颜色
    BLUE = "\033[1;34m"  # 蓝色


def evaluate(model, val_loader, device, num_classes):
    # 设置模型为验证模式
    model_eval = model.eval()
    model_eval = model_eval.cuda()

    # 初始化累积变量
    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_iou = 0
    total_fw_iou = 0
    num_batches = len(val_loader)

    # 遍历验证数据，前向传播
    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels = batch  # 获取验证数据

            imgs = imgs.to(device)
            pngs = pngs.to(device)
            outputs = model_eval(imgs)

            # 计算各个指标
            pixel_acc = pixel_accuracy(outputs, pngs)
            mean_acc = mean_accuracy(outputs, pngs, num_classes)
            mean_iou_value = mean_iou(outputs, pngs, num_classes)
            fw_iou = frequency_weighted_iou(outputs, pngs, num_classes)

            # 累加到总结果
            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou

            # 打印标题（每个epoch开始时打印一次）
            if iteration == 0:  # 只在第一个 batch 打印标题
                data_num_len = len("data_num") - len("data_num") + 12
                Pixelacc_len = len("GPU Mem") - len("Pixelacc") + 12
                Meanacc_len = len("Loss") - len("Meanacc") + 12
                Meaniou_len = len("LR") - len("Meaniou") + 12

                print(
                      f"{LogColor.RED}data_num{LogColor.RESET}{' ' * data_num_len}"
                      f"{LogColor.RED}Pixelacc{LogColor.RESET}{' ' * Pixelacc_len}"
                      f"{LogColor.RED}Meanacc{LogColor.RESET}{' ' * Meanacc_len}"
                      f"{LogColor.RED}Meaniou{LogColor.RESET}{' ' * Meaniou_len}"
                      f"{LogColor.RED}Fwiou{LogColor.RESET}")

    # 计算平均值
    avg_pixel_acc = total_pixel_acc / num_batches
    avg_mean_acc = total_mean_acc / num_batches
    avg_mean_iou = total_mean_iou / num_batches
    avg_fw_iou = total_fw_iou / num_batches


    batch_len = data_num_len + len("data_num") - len(str(f"{len(val_loader.dataset)}"))
    avg_pixel_acc_len = Pixelacc_len + len("Pixelacc") - len(str(f"{avg_pixel_acc:.2f}"))
    avg_mean_acc_len = Meanacc_len + len("Meanacc") - len(str(f"{avg_mean_acc:.2f}"))
    avg_Mean_iou_len = Meaniou_len + len("Meaniou") - len(str(f"{avg_mean_iou:.2f}"))

    # 使用 \r 在同一行更新输出
    print(
          f"{len(val_loader.dataset)}{' ' * batch_len}"
          f"{avg_pixel_acc:.2f}{' ' * avg_pixel_acc_len}"
          f"{avg_mean_acc:.2f}{' ' * avg_mean_acc_len}"
          f"{avg_mean_iou:.2f}{' ' * avg_Mean_iou_len}"
          f"{avg_fw_iou:.2f}", end='', flush=True)
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)  # 加一点延迟，防止输出闪烁过快


def val(args):
    num_classes = args.num_classes + 1  # 类别加上背景类

    # 选择设备（GPU 如果可用，否则使用 CPU）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    input_shape = [480, 512]  # 一定要是32的整数倍

    # 创建验证数据集对象（直接读 WUH 原生 val/image|label）
    # args.data_path: 数据集根路径， input_shape: 输入图像尺寸，num_classes: 类别数
    # augmentation=False: 验证阶段不做增强，split="val": 指定划分
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, split="val")

    # 加载验证集的DataLoader
    val_loader = DataLoader(val_dataset,
                        shuffle=True,
                        batch_size=1,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False, collate_fn=unet_dataset_collate, sampler=None)

    # 创建一个Unet模型实例，num_classes指定了分类的数量（输出通道数）
    model = Unet(num_classes=num_classes)

    # 删除与辅助分类器相关的权重
    weights_dict = torch.load(args.weights, map_location=device)  # 加载模型权重

    # 加载权重
    model.load_state_dict(weights_dict)
    model.to(device)  # 将模型移到相应的设备（GPU 或 CPU）

    # 在验证集上评估模型
    evaluate(model, val_loader, device, num_classes)



def parse_args():
    import argparse
    # 创建 ArgumentParser 对象，用于处理命令行输入
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    # 添加数据路径参数
    parser.add_argument("--data-path",
                        default=r"D:/[C]Remote_Sencing_Projection/Project_WUD/消融实验/WUH_sub25",
                        help="WUH 数据集根目录（内含 train|val，各含 image 与 label）")
    # 添加模型权重路径参数
    parser.add_argument("--weights", default="run/train/exp10/weights/best_model_20.pth")
    # 添加类别数量参数（前景类数，不含背景）
    parser.add_argument("--num-classes", default=1, type=int)
    # 添加训练设备选择的参数，默认为 "cuda"（即使用 GPU）
    parser.add_argument("--device", default="cuda", help="training device")

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 返回解析后的参数对象
    return args


if __name__ == '__main__':
    args = parse_args()
    val(args)

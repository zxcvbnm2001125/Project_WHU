# -*- coding: utf-8 -*-
"""控制台输出格式，与 U-Net / FCN 两个工程保持一致。

移植自 XR_1/U-Net/utils/train_and_eval.py 与 XR_1/FCN/utils/train_and_eval.py，
列宽计算逐字照搬，保证三份日志在同一终端里对齐、配色一致：

    训练期  绿色 "Epoch" + 黄色表头（data_num / GPU Mem / Loss / LR / Image_size）
            \r 原地刷新，字段顺序与 U-Net / FCN 完全相同
    验证期  红色表头（data_num / Pixelacc / Meanacc / Meaniou / Fwiou / Fgiou）
            指标统一 4 位小数，末尾换行并置绿色

原来 DeepLab 用的是 tqdm 进度条 + format_seg_metrics_line 的彩色单行，
字段里没有 GPU Mem / LR / Fgiou，与另外两个工程对不上，故整体替换。
"""


class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"


# U-Net / FCN 里这几个宽度常量写成 "len(字面量) + 偏移" 的形式，
# 这里保留同样的算式，改格式时三边一起改才不会错位。
_EPOCH_HEAD_W = len("Epoch") + 12
_DATA_HEAD_W = len("data_num") - len("data_num") + 12
_PIXELACC_W = len("GPU Mem") - len("Pixelacc") + 12
_MEANACC_W = len("Loss") - len("Meanacc") + 12
_MEANIOU_W = len("LR") - len("Meaniou") + 12


def print_train_header():
    """训练期表头，配色：Epoch 绿、其余黄。"""
    print(f"{LogColor.GREEN}Epoch{LogColor.RESET}{' ' * 12}"
          f"{LogColor.YELLOW}data_num{LogColor.RESET}{' ' * 12}"
          f"{LogColor.YELLOW}GPU Mem{LogColor.RESET}{' ' * 12}"
          f"{LogColor.YELLOW}Loss{LogColor.RESET}{' ' * 12}"
          f"{LogColor.YELLOW}LR{LogColor.RESET}{' ' * 12}"
          f"{LogColor.YELLOW}Image_size{LogColor.RESET}{' ' * 12}"
          )


def print_train_line(epoch, total_epochs, it, total_it, gpu_used, loss, lr, img_size):
    """训练期逐 batch 原地刷新一行（\\r，不换行）。"""
    a = total_it if total_it < 1 else 1

    Epoch_len = len("Epoch") + 12 - len(str(f"{epoch}/{total_epochs}"))
    batch_len = len("data_num") + 12 - len(str(f"{it}/{total_it}"))
    GPU_len = len("GPU Mem") + 12 - len(str(f"{gpu_used:.2f} MB"))
    Loss_len = len("Loss") + 12 - len(str(f"{loss:.8f}"))
    LR_len = len("LR") + 12 - len(str(f"{lr:.8f}"))

    print(f"\r{epoch}/{total_epochs}{' ' * Epoch_len}"
          f"{it}/{total_it}{' ' * batch_len}"
          f"{gpu_used:.2f} MB{' ' * GPU_len}"
          f"{loss:.8f}{' ' * Loss_len}"
          f"{lr:.8f}{' ' * LR_len}"
          f"{img_size}", end='', flush=True)


def print_train_end():
    """训练期收尾：换行并置绿色（与 U-Net / FCN 的 time.sleep 前后效果一致）。"""
    print(f"{LogColor.GREEN}")


def print_val_header():
    """验证期表头，配色：全部红色。"""
    print(f"{' ' * _EPOCH_HEAD_W}"
          f"{LogColor.RED}data_num{LogColor.RESET}{' ' * _DATA_HEAD_W}"
          f"{LogColor.RED}Pixelacc{LogColor.RESET}{' ' * _PIXELACC_W}"
          f"{LogColor.RED}Meanacc{LogColor.RESET}{' ' * _MEANACC_W}"
          f"{LogColor.RED}Meaniou{LogColor.RESET}{' ' * _MEANIOU_W}"
          f"{LogColor.RED}Fwiou{LogColor.RESET}{' ' * 6}"
          f"{LogColor.RED}Fgiou{LogColor.RESET}")


def print_val_line(n_done, n_total, pixel_acc, mean_acc, mean_iou, fw_iou, fg_iou):
    """验证期结果行：一次打印一行，指标 4 位小数，末尾换行并置绿色。"""
    batch_len = _DATA_HEAD_W + len("data_num") - len(str(f"{n_done}/{n_total}"))
    avg_pixel_acc_len = _PIXELACC_W + len("Pixelacc") - len(str(f"{pixel_acc:.4f}"))
    avg_mean_acc_len = _MEANACC_W + len("Meanacc") - len(str(f"{mean_acc:.4f}"))
    avg_mean_iou_len = _MEANIOU_W + len("Meaniou") - len(str(f"{mean_iou:.4f}"))

    print(f"{' ' * _EPOCH_HEAD_W}"
          f"{n_done}/{n_total}{' ' * batch_len}"
          f"{pixel_acc:.4f}{' ' * avg_pixel_acc_len}"
          f"{mean_acc:.4f}{' ' * avg_mean_acc_len}"
          f"{mean_iou:.4f}{' ' * avg_mean_iou_len}"
          f"{fw_iou:.4f}{' ' * 8}"
          f"{fg_iou:.4f}", end='', flush=True)
    print(f"\n{LogColor.GREEN}")

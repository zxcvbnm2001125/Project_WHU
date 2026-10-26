import colorsys
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.unet_resnet import Unet
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.create_exp_folder import create_val_exp_folder

def time_synchronized():
    """
    该函数同步 CUDA 操作，并返回当前时间。

    如果使用 GPU，`torch.cuda.synchronize()` 将会等待所有 CUDA 操作完成，
    以确保测量时间时没有潜在的异步操作。

    Returns:
        float: 当前时间戳，单位为秒。
    """
    # 如果 CUDA 可用，则等待所有 CUDA 操作完成
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # 返回当前时间戳（单位为秒）
    return time.time()

def load_model(model_path, num_classes, device):
    # 创建模型并加载权重
    net = Unet(num_classes=num_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 设置为评估模式
    net.to(device)  # 移动到指定设备
    return net


def detect_image(file_path, model, num_classes, exp_folder, mix_type=True):
    try:
        image = Image.open(file_path)
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening image: {e}")
        return

    # 自动选择设备（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将图像转换为 RGB 格式
    image = cvtColor(image)
    old_img = image.copy()  # 直接复制原图

    input_shape = [480, 480] # 设置输出图片的大小
    orininal_h, orininal_w = np.array(image).shape[:2]
    image_data, nw, nh = resize_image(image, (input_shape[1], input_shape[0]))

    # 预处理图像
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    # 生成颜色映射
    if num_classes <= 21:
        colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                  (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                  (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 128)]
    else:
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    with torch.no_grad():
        # 将图像数据转换为 PyTorch 张量
        images = torch.from_numpy(image_data).to(device)

        # 进行推理
        pr = model(images)[0]

        # Softmax 后处理
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        # 恢复图像尺寸
        pr = pr[int((input_shape[0] - nh) // 2): int((input_shape[0] - nh) // 2 + nh),
             int((input_shape[1] - nw) // 2): int((input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

        # 获取预测类别
        pr = pr.argmax(axis=-1)

    # 根据不同的混合类型，生成不同的结果
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    if mix_type :
        # 将 PIL 图像转换为 NumPy 数组
        old_img_np = np.array(old_img)
        alpha = 0.7
        # 使用 OpenCV 进行加权混合
        blended_img = cv2.addWeighted(old_img_np, 1 - alpha, seg_img, alpha, 0)
        # 将混合后的图像转换回 PIL 图像
        image = Image.fromarray(blended_img)
    else:
        image = Image.fromarray(np.uint8(seg_img))

    # 保存结果
    img_name = os.path.basename(file_path)
    mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
    save_path = os.path.join(exp_folder, mask_filename)
    image.save(save_path)
    print(f"Mask saved at: {save_path}")


def predict(args):
    exp_folder = create_val_exp_folder()
    num_classes = args.num_classes + 1

    # 确保路径存在
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = load_model(args.weights, num_classes, device)

    # 处理输入路径
    if os.path.isdir(args.data_path):
        file_paths = [str(p) for p in Path(args.data_path).rglob("*") if p.suffix in [".jpg", ".png", ".jpeg"]]
    elif os.path.isfile(args.data_path):
        file_paths = [args.data_path]
    else:
        raise ValueError(f"Unsupported input path: {args.data_path}")

    # 记录推理开始时间
    t_start = time_synchronized()

    # 对每个文件进行推理
    for file_path in file_paths:
        if file_path.endswith((".jpg", ".png", ".jpeg", ".tif")):
            detect_image(file_path, model, num_classes, exp_folder, mix_type=args.mix_type)

    # 记录推理结束时间
    t_end = time_synchronized()

    # 输出推理所花费的时间
    print(f"inference time for: {t_end - t_start}")


def parse_args():
    import argparse
    # 创建 ArgumentParser 对象，用于处理命令行输入
    parser = argparse.ArgumentParser(description="pytorch unet predict")

    # 添加数据路径参数
    parser.add_argument("--data_path", default=r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25\val\image\val_562.tif",
                        help="data root")
    # 添加模型权重路径参数
    parser.add_argument("--weights", default=r"run\train\exp\weights\best_model_1.pth")
    # 添加类别数量参数，默认为 3
    parser.add_argument("--num-classes", default=1, type=int)

    # 添加是否保存并排显示图像的参数，默认为 False
    parser.add_argument("--mix_type", default=True, action='store_true',
                        help="Save original and segmentation result side by side")

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 返回解析后的参数对象
    return args


if __name__ == "__main__":
    args = parse_args()
    predict(args)


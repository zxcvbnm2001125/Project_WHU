import os
import time
import json

import torch
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from model import fcn_resnet50
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
    # 创建模型：这里使用 FCN 模型，使用 ResNet-50 作为 backbone
    net = fcn_resnet50(num_classes=num_classes)  # num_classes + 1 是因为背景也要考虑

    # 删除与辅助分类器相关的权重
    weights_dict = torch.load(model_path, map_location=device)  # 加载模型权重
    for k in list(weights_dict.keys()):
        if "aux" in k:  # 删除与辅助分类器相关的权重
            del weights_dict[k]
    # 加载权重
    net.load_state_dict(weights_dict)
    net.eval()  # 设置为评估模式
    net.to(device)   # 将模型移到相应的设备（GPU 或 CPU）

    return net


def detect_image(image_path, model, data_transform, device, exp_folder, pallette,  mix_type=False):

    # 加载原始图像 (RGB)
    original_img = Image.open(image_path).convert("RGB")
    original_cv_img = np.array(original_img)[:, :, ::-1]  # 转为 BGR 用于 cv2

    # 将原始图像转换为 Tensor 并标准化
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    # 模型推理
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)  # 预热
        output = model(img.to(device))
        prediction = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ================================
    # 使用 PIL + putpalette 保存彩色 mask
    # ================================
    mask_pil = Image.fromarray(prediction, mode='P')
    mask_pil.putpalette(pallette)

    # resize 回原图大小，保持可视化一致
    mask_resized_pil = mask_pil.resize((original_cv_img.shape[1], original_cv_img.shape[0]), resample=Image.NEAREST)
    img_name = os.path.basename(image_path)

    #   mix_type = True的时候代表原图与生成的图进行混合
    if mix_type :
        mask_rgb = np.array(mask_resized_pil.convert("RGB"))[:, :, ::-1]  # PIL->RGB->BGR
        alpha = 0.7
        blended = cv2.addWeighted(original_cv_img, 1 - alpha, mask_rgb, alpha, 0)
        # 保存 overlay
        overlay_filename = os.path.splitext(img_name)[0] + "_mask.png"
        overlay_path = os.path.join(exp_folder, overlay_filename)
        cv2.imwrite(overlay_path, blended)
        print(f"Overlay saved at: {overlay_path}")

    #   mix_type = 1的的时候代表仅保留生成的图
    else :
        # 保存 mask
        mask_filename = os.path.splitext(img_name)[0] + "_mask.png"
        mask_path = os.path.join(exp_folder, mask_filename)
        mask_resized_pil.save(mask_path)
        print(f"Mask saved at: {mask_path}")



def predict(args):
    # 调用函数获取新的exp文件夹和weights文件夹路径
    exp_folder = create_val_exp_folder()

    # 定义类别数量，语义分割任务中，num_classes 是类别数 + 1（包含背景）
    num_classes = args.num_classes + 1

    # 调色板文件路径（用于将分割结果映射到颜色）
    palette_path = "utils/palette.json"

    # 确保路径中的文件存在
    assert os.path.exists(args.weights), f"weights {args.weights} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."

    # 读取调色板文件（JSON 格式），获取每个类别的颜色
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)  # 加载调色板数据
        pallette = []
        for v in pallette_dict.values():
            pallette += v  # 将颜色添加到列表中


    # 选择设备（GPU 或 CPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 确保路径存在
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # 加载模型
    model = load_model(args.weights, num_classes, device)

    # 处理不同的输入类型（单张图片，文件夹）
    if os.path.isdir(args.data_path):  # 如果输入是文件夹
        file_paths = [str(p) for p in Path(args.data_path).rglob("*") if
                      p.suffix in [".jpg", ".png", ".jpeg", ".tif"]]
    elif os.path.isfile(args.data_path):  # 如果是单一文件
        file_paths = [args.data_path]
    else:
        raise ValueError(f"Unsupported input path: {args.data_path}")

    # 定义图像预处理的转换操作
    data_transform = transforms.Compose([transforms.Resize(512),  # 将图像调整为至少 512的大小
                                         transforms.ToTensor(),   # 将图像转换为 Tensor 格式
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])  # 标准化

    # 记录推理开始时间
    t_start = time_synchronized()
    # 处理文件夹中的每个文件
    for file_path in file_paths:
        if file_path.endswith((".jpg", ".png", ".jpeg", ".tif")):  # 如果是图片文件
            detect_image(file_path, model, data_transform, device, exp_folder, pallette, mix_type=args.mix_type)

    # 记录推理结束时间
    t_end = time_synchronized()

    # 输出推理所花费的时间
    print(f"inference time for: {t_end - t_start}")


def parse_args():
    import argparse
    # 创建 ArgumentParser 对象，用于处理命令行输入
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    # 添加数据路径参数
    parser.add_argument("--data_path", default=r"D:\[C]Remote_Sencing_Projection\Project_WUD\消融实验\WUH_sub25\test\image", help="data root")
    # 添加模型权重路径参数
    parser.add_argument("--weights", default=r"run\train\exp3\weights\best_model_1.pth")
    # 添加类别数量参数，默认为 3
    parser.add_argument("--num-classes", default=1, type=int)

    # 添加是否保存并排显示图像的参数，默认为 False
    parser.add_argument("--mix_type", default=True, action='store_true',
                        help="Save original and segmentation result side by side")

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 返回解析后的参数对象
    return args


if __name__ == '__main__':
    args = parse_args()
    predict(args)

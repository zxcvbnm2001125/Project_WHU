import matplotlib.pyplot as plt
import numpy as np
import os

# 类别名（用于逐类 IoU 曲线的图例）；未列出的类别数用 class{i} 兜底
CLASS_NAMES_BY_N = {
    2: ["background", "building"],
    7: ["background", "building", "road", "water", "barren", "forest", "agriculture"],
}


def plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder):
    # 准备数据
    epochs = range(1, len(train_losses) + 1)
    pixel_acc_list = [m["Pixel Accuracy"] for m in val_metrics_history]
    mean_acc_list = [m["Mean Accuracy"] for m in val_metrics_history]
    mean_iou_list = [m["Mean IoU"] for m in val_metrics_history]
    fw_iou_list = [m["Frequency Weighted IoU"] for m in val_metrics_history]

    # ========================
    # 📈 绘制 Loss 曲线
    # ========================
    plt.figure(figsize=(8,6))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=14, fontname='Times New Roman')
    plt.ylabel("Loss", fontsize=14, fontname='Times New Roman')
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(prop={'family':'Times New Roman', 'size':12})
    plt.tight_layout()
    plt.savefig(os.path.join(weights_folder, "loss_curve.png"), dpi=300)
    plt.close()

    # =========================
    # 📈 绘制指标曲线
    # =========================
    plt.figure(figsize=(8,6))
    plt.plot(epochs, pixel_acc_list, label="Pixel Accuracy", linewidth=2)
    plt.plot(epochs, mean_acc_list, label="Mean Accuracy", linewidth=2)
    plt.plot(epochs, mean_iou_list, label="Mean IoU", linewidth=2)
    plt.plot(epochs, fw_iou_list, label="FWIoU", linewidth=2)

    plt.xlabel("Epoch", fontsize=14, fontname='Times New Roman')
    plt.ylabel("Score", fontsize=14, fontname='Times New Roman')
    plt.xticks(fontsize=12, fontname='Times New Roman')
    plt.yticks(fontsize=12, fontname='Times New Roman')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(prop={'family':'Times New Roman', 'size':12})
    plt.tight_layout()
    plt.savefig(os.path.join(weights_folder, "metrics_curve.png"), dpi=300)
    plt.close()

    # =========================
    # 📈 逐类 IoU 曲线（多类任务的关键图：进步往往只体现在个别类上）
    # =========================
    iou_per_class = [m.get("IoU per class") for m in val_metrics_history]
    if iou_per_class and iou_per_class[0]:
        n_cls = len(iou_per_class[0])
        names = CLASS_NAMES_BY_N.get(n_cls, [f"class{i}" for i in range(n_cls)])

        plt.figure(figsize=(9, 6))
        for c in range(n_cls):
            ys = [row[c] if row is not None and c < len(row) else np.nan for row in iou_per_class]
            plt.plot(epochs, ys, label=names[c], linewidth=1.8)
        plt.xlabel("Epoch", fontsize=14, fontname='Times New Roman')
        plt.ylabel("IoU per class", fontsize=14, fontname='Times New Roman')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(prop={'family': 'Times New Roman', 'size': 10}, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(weights_folder, "iou_per_class_curve.png"), dpi=300)
        plt.close()

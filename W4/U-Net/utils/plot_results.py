import matplotlib.pyplot as plt
import os

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

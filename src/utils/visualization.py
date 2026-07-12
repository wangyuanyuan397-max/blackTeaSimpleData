"""
??: src/utils/visualization.py
??: ???????
????: ????????????????????
????: ??????????????????
"""

import itertools
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_chinese_font():
    system = platform.system()
    if system == "Windows":
        font_names = ["SimHei", "Microsoft YaHei", "SimSun", "Malgun Gothic"]
    elif system == "Darwin":
        font_names = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
    else:
        font_names = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei"]

    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in font_names:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("[Warning] No compatible Chinese font found. Plots may contain garbled text.")


set_chinese_font()


def _format_confusion_labels(classes):
    """
    For numeric labels 0..N-1, render as 0.0h, 0.5h, ..., to match 13 timepoints.
    Otherwise keep original class names.
    """
    try:
        numeric = [int(c) for c in classes]
    except (TypeError, ValueError):
        return [str(c) for c in classes]

    if sorted(numeric) == list(range(len(numeric))):
        # 13-class (0.0h~6.0h, 0.5h step)
        if len(numeric) == 13:
            return [f"{idx * 0.5:.1f}h" for idx in numeric]
        # 4-class reconstructed anchors
        if len(numeric) == 4:
            return ["0.0h", "2.0h", "4.0h", "6.0h"]
    return [str(c) for c in classes]


def plot_confusion_matrix(cm, classes, save_path, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    Plot confusion matrix with standard orientation:
    - row: true label
    - col: predicted label
    """
    plt.figure(figsize=(10, 8))
    cm = np.array(cm)
    labels = _format_confusion_labels(classes)

    try:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
        )
    except Exception as e:
        print(f"[Warning] Seaborn heatmap failed: {e}. Fallback to Matplotlib.")
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"[Error] Failed to save confusion matrix: {e}")
    finally:
        plt.close()


def plot_training_curves(history, save_path):
    """绘制并保存训练历史，兼容早停或异常中断产生的不完整序列。"""
    if not history or not history.get("train_loss"):
        print("[Warning] No history data to plot.")
        return None

    def plot_series(axis, key, label, color, marker=None):
        """按各指标自己的实际长度绘图，避免部分 epoch 缺少某指标时报错。"""
        values = history.get(key) or []
        if not values:
            return False
        epochs = range(1, len(values) + 1)
        axis.plot(
            epochs,
            values,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=3,
            label=label,
        )
        return True

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    plot_series(axes[0, 0], "train_loss", "Train Loss", "#2563eb")
    plot_series(axes[0, 0], "val_loss", "Validation Loss", "#dc2626")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_ylabel("Loss")

    plot_series(axes[0, 1], "train_acc", "Train Accuracy", "#2563eb")
    plot_series(axes[0, 1], "val_acc", "Validation Accuracy", "#dc2626")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].set_ylabel("Accuracy")

    has_mae = plot_series(
        axes[1, 0], "val_mae", "Validation MAE", "#d97706", marker="o"
    )
    axes[1, 0].set_title("Validation MAE")
    axes[1, 0].set_ylabel("MAE")
    if not has_mae:
        axes[1, 0].text(0.5, 0.5, "No MAE history", ha="center", va="center")

    has_qwk = plot_series(
        axes[1, 1], "val_qwk", "Validation QWK", "#059669", marker="o"
    )
    axes[1, 1].set_title("Validation QWK")
    axes[1, 1].set_ylabel("QWK")
    if not has_qwk:
        axes[1, 1].text(0.5, 0.5, "No QWK history", ha="center", va="center")

    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(True, linestyle="--", alpha=0.35)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()

    figure.suptitle("Training Curves", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    try:
        figure.savefig(save_path, dpi=180, bbox_inches="tight")
        return save_path
    except Exception as e:
        print(f"[Error] Failed to save training curves: {e}")
        return None
    finally:
        plt.close(figure)

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
    if not history or not history.get("train_loss"):
        print("[Warning] No history data to plot.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    if "val_loss" in history and len(history["val_loss"]) == len(epochs):
        plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if "train_acc" in history and len(history["train_acc"]) == len(epochs):
        plt.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    if "val_acc" in history and len(history["val_acc"]) == len(epochs):
        plt.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"[Error] Failed to save training curves: {e}")
    finally:
        plt.close()

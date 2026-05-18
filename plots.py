"""
plots.py

Plotting utilities for the Hybrid IFCM-TransUNet-FMRCNN emphysema framework.

This module generates reviewer-facing and manuscript-ready visual outputs:
- Training and validation curves
- Metric comparison bar charts
- ROC curves
- Confusion matrices
- Segmentation overlay grids
- Detection maps
- Error metric comparison charts

All figures are saved at high resolution using matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def ensure_parent_dir(path: str | Path) -> None:
    """
    Create parent directory for a file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_figure(output_path: str | Path, dpi: int = 300) -> None:
    """
    Save current matplotlib figure.
    """
    ensure_parent_dir(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
) -> None:
    """
    Plot training/validation loss, Dice, and IoU curves from training_log.json.

    Parameters
    ----------
    history_path:
        JSON file saved by train.py.
    output_dir:
        Folder where figures will be saved.
    """
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        raise FileNotFoundError(f"Training history file not found: {history_path}")

    history = pd.read_json(history_path)

    if history.empty:
        raise ValueError("Training history is empty.")

    epochs = history["epoch"].values

    if {"train_loss", "val_loss"}.issubset(history.columns):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
        plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(output_dir / "training_validation_loss.png", dpi=dpi)

    if {"train_dice", "val_dice"}.issubset(history.columns):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["train_dice"], marker="o", label="Training Dice")
        plt.plot(epochs, history["val_dice"], marker="o", label="Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score")
        plt.title("Training and Validation Dice Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(output_dir / "training_validation_dice.png", dpi=dpi)

    if {"train_iou", "val_iou"}.issubset(history.columns):
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history["train_iou"], marker="o", label="Training IoU")
        plt.plot(epochs, history["val_iou"], marker="o", label="Validation IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.title("Training and Validation IoU")
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_figure(output_dir / "training_validation_iou.png", dpi=dpi)


def plot_metric_bar(
    metrics: Dict[str, float],
    output_path: str | Path,
    title: str = "Model Performance Metrics",
    dpi: int = 300,
) -> None:
    """
    Plot a bar chart of metric values.
    """
    if not metrics:
        raise ValueError("metrics dictionary is empty.")

    names = list(metrics.keys())
    values = [float(metrics[name]) for name in names]

    plt.figure(figsize=(max(8, len(names) * 0.7), 5))
    plt.bar(names, values)
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_figure(output_path, dpi=dpi)


def plot_metric_comparison(
    comparison: Dict[str, Dict[str, float]],
    output_path: str | Path,
    title: str = "Comparative Performance Analysis",
    dpi: int = 300,
) -> None:
    """
    Plot grouped metric comparison across multiple models.

    Expected format:
    {
        "Proposed": {"accuracy": 0.974, "dice": 0.95},
        "U-Net": {"accuracy": 0.897, "dice": 0.86}
    }
    """
    if not comparison:
        raise ValueError("comparison dictionary is empty.")

    model_names = list(comparison.keys())
    metric_names = list(next(iter(comparison.values())).keys())

    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    plt.figure(figsize=(max(9, len(metric_names) * 1.2), 5.5))

    for index, model_name in enumerate(model_names):
        values = [float(comparison[model_name].get(metric, 0.0)) for metric in metric_names]
        offset = (index - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model_name)

    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(x, metric_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_figure(output_path, dpi=dpi)


def plot_error_comparison(
    error_metrics: Dict[str, Dict[str, float]],
    output_path: str | Path,
    title: str = "Error Metric Comparison",
    dpi: int = 300,
) -> None:
    """
    Plot grouped comparison of MSE, MAE, and RMSE across models.
    """
    plot_metric_comparison(
        comparison=error_metrics,
        output_path=output_path,
        title=title,
        dpi=dpi,
    )


def plot_roc_curve(
    y_true: Sequence[float],
    y_score: Sequence[float],
    output_path: str | Path,
    title: str = "ROC Curve",
    dpi: int = 300,
) -> float:
    """
    Plot ROC curve and return AUC value.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC curve requires at least two classes in y_true.")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_figure(output_path, dpi=dpi)
    return float(auc_value)


def plot_multi_roc_curve(
    roc_items: Dict[str, Dict[str, Sequence[float]]],
    output_path: str | Path,
    title: str = "ROC Curve Comparison",
    dpi: int = 300,
) -> Dict[str, float]:
    """
    Plot multiple ROC curves.

    Expected format:
    {
        "Proposed": {"y_true": [...], "y_score": [...]},
        "U-Net": {"y_true": [...], "y_score": [...]}
    }
    """
    auc_values = {}

    plt.figure(figsize=(7, 5.5))

    for model_name, values in roc_items.items():
        y_true = np.asarray(values["y_true"]).reshape(-1)
        y_score = np.asarray(values["y_score"]).reshape(-1)

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = auc(fpr, tpr)
        auc_values[model_name] = float(auc_value)

        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} AUC = {auc_value:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_figure(output_path, dpi=dpi)
    return auc_values


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[List[str]],
    output_path: str | Path,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    dpi: int = 300,
) -> np.ndarray:
    """
    Plot confusion matrix.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    labels = list(range(len(class_names))) if class_names else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm_display = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        cm_display = cm

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm_display, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    threshold = cm_display.max() / 2.0 if cm_display.size else 0.0

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            plt.text(
                j,
                i,
                format(cm_display[i, j], fmt),
                ha="center",
                va="center",
            )

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    save_figure(output_path, dpi=dpi)
    return cm


def _normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] for visualization.
    """
    image = np.asarray(image, dtype=np.float32)
    minimum = float(image.min())
    maximum = float(image.max())

    if maximum - minimum < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    return (image - minimum) / (maximum - minimum)


def save_segmentation_overlay_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    predictions: List[np.ndarray],
    output_path: str | Path,
    max_samples: int = 6,
    dpi: int = 300,
) -> None:
    """
    Save grid showing CT image, ground-truth mask, and predicted mask.
    """
    count = min(len(images), len(masks), len(predictions), max_samples)

    if count == 0:
        raise ValueError("No samples available for overlay grid.")

    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count))

    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(count):
        image = _normalize_image_for_display(images[idx])
        mask = np.asarray(masks[idx])
        pred = np.asarray(predictions[idx])

        axes[idx, 0].imshow(image, cmap="gray")
        axes[idx, 0].set_title("Input CT")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(image, cmap="gray")
        axes[idx, 1].imshow(mask, alpha=0.45)
        axes[idx, 1].set_title("Ground Truth")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(image, cmap="gray")
        axes[idx, 2].imshow(pred, alpha=0.45)
        axes[idx, 2].set_title("Prediction")
        axes[idx, 2].axis("off")

    save_figure(output_path, dpi=dpi)


def save_detection_map(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    output_path: str | Path,
    title: str = "Detection Map",
    dpi: int = 300,
) -> None:
    """
    Save detection map with bounding boxes.
    """
    image = _normalize_image_for_display(image)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")

    boxes = np.asarray(boxes)

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [float(v) for v in box]
        width = xmax - xmin
        height = ymax - ymin

        rect = plt.Rectangle(
            (xmin, ymin),
            width,
            height,
            fill=False,
            linewidth=2,
        )
        plt.gca().add_patch(rect)

        label_text = ""
        if labels is not None and idx < len(labels):
            label_text += f"C{int(labels[idx])}"
        if scores is not None and idx < len(scores):
            label_text += f" {float(scores[idx]):.2f}"

        if label_text:
            plt.text(xmin, max(0, ymin - 3), label_text, fontsize=8)

    plt.title(title)
    plt.axis("off")
    save_figure(output_path, dpi=dpi)


def plot_ifcm_objective(
    objective_history: Sequence[float],
    output_path: str | Path,
    title: str = "IFCM Objective Convergence",
    dpi: int = 300,
) -> None:
    """
    Plot IFCM objective convergence curve.
    """
    values = np.asarray(objective_history, dtype=np.float32)

    if values.size == 0:
        raise ValueError("objective_history is empty.")

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(values) + 1), values, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    save_figure(output_path, dpi=dpi)


def save_ifcm_result_grid(
    image: np.ndarray,
    label_map: np.ndarray,
    prior_mask: np.ndarray,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Save IFCM qualitative result: input, cluster labels, and prior mask.
    """
    image = _normalize_image_for_display(image)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input CT")
    axes[0].axis("off")

    axes[1].imshow(label_map)
    axes[1].set_title("IFCM Cluster Map")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(prior_mask, alpha=0.45)
    axes[2].set_title("IFCM Prior Mask")
    axes[2].axis("off")

    save_figure(output_path, dpi=dpi)


def plot_table_like_metrics(
    metrics: Dict[str, float],
    output_path: str | Path,
    title: str = "Metric Summary Table",
    dpi: int = 300,
) -> None:
    """
    Save a simple table-like metric figure.
    """
    names = list(metrics.keys())
    values = [f"{float(v):.4f}" for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(names) * 0.35)))
    ax.axis("off")

    table_data = [[name, value] for name, value in zip(names, values)]
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    plt.title(title)
    save_figure(output_path, dpi=dpi)


def generate_default_paper_comparison_figures(output_dir: str | Path, dpi: int = 300) -> None:
    """
    Generate paper-style comparison charts using the reported manuscript values.

    This function is useful only for reproducing manuscript summary charts.
    It does not replace model-based evaluation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_comparison = {
        "Proposed": {"Training Accuracy": 99.7, "Validation Accuracy": 97.4},
        "FCM": {"Training Accuracy": 90.5, "Validation Accuracy": 85.6},
        "U-Net": {"Training Accuracy": 93.8, "Validation Accuracy": 89.7},
        "Mask R-CNN": {"Training Accuracy": 95.2, "Validation Accuracy": 92.3},
        "DeepLabV3+": {"Training Accuracy": 92.7, "Validation Accuracy": 88.5},
    }

    plot_metric_comparison(
        comparison=accuracy_comparison,
        output_path=output_dir / "reported_accuracy_comparison.png",
        title="Reported Training and Validation Accuracy Comparison",
        dpi=dpi,
    )

    error_comparison = {
        "Proposed": {"MSE": 0.024, "MAE": 0.085, "RMSE": 0.153},
        "FCM": {"MSE": 0.068, "MAE": 0.155, "RMSE": 0.260},
        "U-Net": {"MSE": 0.057, "MAE": 0.129, "RMSE": 0.238},
        "Mask R-CNN": {"MSE": 0.046, "MAE": 0.108, "RMSE": 0.212},
        "DeepLabV3+": {"MSE": 0.064, "MAE": 0.140, "RMSE": 0.252},
    }

    plot_error_comparison(
        error_metrics=error_comparison,
        output_path=output_dir / "reported_error_comparison.png",
        title="Reported Error Metric Comparison",
        dpi=dpi,
    )


if __name__ == "__main__":
    generate_default_paper_comparison_figures("outputs/figures")
    print("Default comparison figures generated in outputs/figures.")

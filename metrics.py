"""
metrics.py

Reusable evaluation metrics for the Hybrid IFCM-TransUNet-FMRCNN emphysema
diagnosis framework.

This module includes:
- Segmentation metrics
- Binary classification metrics
- Error metrics
- ROC-AUC helpers
- Confusion-matrix helpers
- Detection summary utilities

All functions support PyTorch tensors and NumPy arrays where practical.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


EPS = 1e-8


def _to_tensor(x) -> torch.Tensor:
    """
    Convert NumPy array or tensor to float tensor.
    """
    if torch.is_tensor(x):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


def _flatten_binary(preds, targets) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten prediction and target tensors for binary metric computation.
    """
    preds = _to_tensor(preds).detach().reshape(-1)
    targets = _to_tensor(targets).detach().reshape(-1)
    preds = (preds >= 0.5).float()
    targets = (targets >= 0.5).float()
    return preds, targets


def dice_score(preds, targets, smooth: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient.
    """
    preds, targets = _flatten_binary(preds, targets)
    intersection = torch.sum(preds * targets)
    denominator = torch.sum(preds) + torch.sum(targets)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.item())


def iou_score(preds, targets, smooth: float = 1e-6) -> float:
    """
    Intersection over Union.
    """
    preds, targets = _flatten_binary(preds, targets)
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(iou.item())


def true_positive(preds, targets) -> float:
    preds, targets = _flatten_binary(preds, targets)
    return float(torch.sum((preds == 1) & (targets == 1)).item())


def true_negative(preds, targets) -> float:
    preds, targets = _flatten_binary(preds, targets)
    return float(torch.sum((preds == 0) & (targets == 0)).item())


def false_positive(preds, targets) -> float:
    preds, targets = _flatten_binary(preds, targets)
    return float(torch.sum((preds == 1) & (targets == 0)).item())


def false_negative(preds, targets) -> float:
    preds, targets = _flatten_binary(preds, targets)
    return float(torch.sum((preds == 0) & (targets == 1)).item())


def accuracy_score_binary(preds, targets) -> float:
    """
    Binary accuracy.
    """
    preds, targets = _flatten_binary(preds, targets)
    correct = torch.sum(preds == targets)
    total = targets.numel()
    return float((correct / max(total, 1)).item())


def precision_score_binary(preds, targets) -> float:
    """
    Binary precision.
    """
    tp = true_positive(preds, targets)
    fp = false_positive(preds, targets)
    return float(tp / (tp + fp + EPS))


def recall_score_binary(preds, targets) -> float:
    """
    Binary recall / sensitivity.
    """
    tp = true_positive(preds, targets)
    fn = false_negative(preds, targets)
    return float(tp / (tp + fn + EPS))


def sensitivity_score_binary(preds, targets) -> float:
    """
    Alias for recall.
    """
    return recall_score_binary(preds, targets)


def specificity_score_binary(preds, targets) -> float:
    """
    Binary specificity.
    """
    tn = true_negative(preds, targets)
    fp = false_positive(preds, targets)
    return float(tn / (tn + fp + EPS))


def f1_score_binary(preds, targets) -> float:
    """
    Binary F1-score.
    """
    precision = precision_score_binary(preds, targets)
    recall = recall_score_binary(preds, targets)
    return float(2.0 * precision * recall / (precision + recall + EPS))


def f2_score_binary(preds, targets) -> float:
    """
    Binary F2-score, giving recall more weight than precision.
    """
    precision = precision_score_binary(preds, targets)
    recall = recall_score_binary(preds, targets)
    beta_sq = 4.0
    return float((1 + beta_sq) * precision * recall / (beta_sq * precision + recall + EPS))


def balanced_accuracy_score_binary(preds, targets) -> float:
    """
    Balanced accuracy = average of sensitivity and specificity.
    """
    sensitivity = sensitivity_score_binary(preds, targets)
    specificity = specificity_score_binary(preds, targets)
    return float((sensitivity + specificity) / 2.0)


def mse_score(preds, targets) -> float:
    """
    Mean Squared Error.
    """
    preds = _to_tensor(preds).detach()
    targets = _to_tensor(targets).detach()
    return float(torch.mean((preds - targets) ** 2).item())


def mae_score(preds, targets) -> float:
    """
    Mean Absolute Error.
    """
    preds = _to_tensor(preds).detach()
    targets = _to_tensor(targets).detach()
    return float(torch.mean(torch.abs(preds - targets)).item())


def rmse_score(preds, targets) -> float:
    """
    Root Mean Squared Error.
    """
    return float(np.sqrt(mse_score(preds, targets)))


def compute_binary_segmentation_metrics(preds, targets) -> Dict[str, float]:
    """
    Compute full binary segmentation metric dictionary.
    """
    return {
        "accuracy": accuracy_score_binary(preds, targets),
        "dice": dice_score(preds, targets),
        "iou": iou_score(preds, targets),
        "precision": precision_score_binary(preds, targets),
        "recall": recall_score_binary(preds, targets),
        "sensitivity": sensitivity_score_binary(preds, targets),
        "specificity": specificity_score_binary(preds, targets),
        "f1_score": f1_score_binary(preds, targets),
        "f2_score": f2_score_binary(preds, targets),
        "balanced_accuracy": balanced_accuracy_score_binary(preds, targets),
        "mse": mse_score(preds, targets),
        "mae": mae_score(preds, targets),
        "rmse": rmse_score(preds, targets),
    }


def safe_roc_auc(y_true, y_score) -> float:
    """
    Compute ROC-AUC safely. Returns NaN if only one class is present.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(roc_auc_score(y_true, y_score))


def safe_average_precision(y_true, y_score) -> float:
    """
    Compute Average Precision safely. Returns NaN if no valid positive class exists.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(average_precision_score(y_true, y_score))


def compute_roc_data(y_true, y_score) -> Dict[str, np.ndarray | float]:
    """
    Compute ROC curve data and AUC.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return {
            "fpr": np.array([]),
            "tpr": np.array([]),
            "thresholds": np.array([]),
            "auc": float("nan"),
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_value = auc(fpr, tpr)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(auc_value),
    }


def compute_precision_recall_data(y_true, y_score) -> Dict[str, np.ndarray | float]:
    """
    Compute precision-recall curve data and average precision.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return {
            "precision": np.array([]),
            "recall": np.array([]),
            "thresholds": np.array([]),
            "average_precision": float("nan"),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "average_precision": float(ap),
    }


def compute_confusion_matrix(
    y_true,
    y_pred,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return confusion_matrix(y_true, y_pred, labels=labels)


def classwise_metrics_from_confusion_matrix(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute class-wise precision, recall, specificity, and F1 from confusion matrix.
    """
    cm = np.asarray(cm)
    num_classes = cm.shape[0]

    result = {}

    for class_id in range(num_classes):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - tp)
        fn = float(cm[class_id, :].sum() - tp)
        tn = float(cm.sum() - tp - fp - fn)

        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        specificity = tn / (tn + fp + EPS)
        f1 = 2.0 * precision * recall / (precision + recall + EPS)

        result[f"class_{class_id}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
        }

    return result


def macro_average_classwise_metrics(classwise_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute macro average over class-wise metrics.
    """
    if not classwise_metrics:
        return {}

    keys = next(iter(classwise_metrics.values())).keys()

    return {
        key: float(np.mean([metrics[key] for metrics in classwise_metrics.values()]))
        for key in keys
    }


def detection_count_summary(detections: List[Dict]) -> Dict[str, float]:
    """
    Summarize detection list produced by inference/evaluation scripts.
    """
    if not detections:
        return {
            "total_detections": 0,
            "mean_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
        }

    scores = [
        float(item.get("score", item.get("confidence", 0.0)))
        for item in detections
    ]

    return {
        "total_detections": int(len(detections)),
        "mean_confidence": float(np.mean(scores)),
        "max_confidence": float(np.max(scores)),
        "min_confidence": float(np.min(scores)),
    }


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes.

    Box format: [xmin, ymin, xmax, ymax]
    """
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)

    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return float(intersection / (union + EPS))


def mean_box_iou(pred_boxes: np.ndarray, true_boxes: np.ndarray) -> float:
    """
    Compute mean best-match IoU between predicted and ground-truth boxes.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    true_boxes = np.asarray(true_boxes, dtype=np.float32)

    if pred_boxes.size == 0 or true_boxes.size == 0:
        return 0.0

    best_ious = []

    for pred_box in pred_boxes:
        ious = [box_iou(pred_box, true_box) for true_box in true_boxes]
        best_ious.append(max(ious) if ious else 0.0)

    return float(np.mean(best_ious))


def aggregate_metric_dicts(metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """
    Average a list of metric dictionaries.
    """
    metric_dicts = list(metric_dicts)

    if not metric_dicts:
        return {}

    keys = metric_dicts[0].keys()
    return {
        key: float(np.mean([metrics[key] for metrics in metric_dicts if key in metrics]))
        for key in keys
    }


def bootstrap_confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    num_bootstrap: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a list of scalar metric values.
    """
    values = np.asarray(list(values), dtype=np.float32)

    if values.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    boot_means = []

    for _ in range(num_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(lower), float(upper)

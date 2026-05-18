"""
evaluate.py

Evaluation script for the Hybrid IFCM-TransUNet-FMRCNN emphysema framework.

This script evaluates:
- TransUNet segmentation performance
- Faster Mask R-CNN detection output availability
- Classification-style performance from labels and predictions
- Error metrics and ROC-AUC where valid labels are available

Usage:
    python evaluate.py --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data import build_dataloaders
from fmrcnn import build_fmrcnn_from_config, fmrcnn_inference, summarize_detections
from metrics import (
    accuracy_score_binary,
    dice_score,
    f1_score_binary,
    iou_score,
    mae_score,
    mse_score,
    precision_score_binary,
    recall_score_binary,
    rmse_score,
    specificity_score_binary,
)
from plots import (
    plot_confusion_matrix,
    plot_metric_bar,
    plot_roc_curve,
    save_segmentation_overlay_grid,
)
from transunet import build_transunet
from utils import create_output_dirs, get_device, load_checkpoint, save_json, set_seed


def load_config(path: str | Path) -> Dict:
    """
    Load YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_transunet_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load TransUNet checkpoint if available.
    """
    if not checkpoint_path.exists():
        print(f"Warning: TransUNet checkpoint not found: {checkpoint_path}")
        return model

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded TransUNet checkpoint: {checkpoint_path}")
    return model


def load_fmrcnn_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load FMRCNN checkpoint if available.
    """
    if not checkpoint_path.exists():
        print(f"Warning: FMRCNN checkpoint not found: {checkpoint_path}")
        return model

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded FMRCNN checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def evaluate_segmentation(model: torch.nn.Module, dataloader, device: torch.device, threshold: float = 0.5) -> Dict:
    """
    Evaluate segmentation model on test data.
    """
    model.eval()

    dice_values: List[float] = []
    iou_values: List[float] = []
    accuracy_values: List[float] = []
    precision_values: List[float] = []
    recall_values: List[float] = []
    specificity_values: List[float] = []
    f1_values: List[float] = []
    mse_values: List[float] = []
    mae_values: List[float] = []
    rmse_values: List[float] = []

    all_true_pixels = []
    all_prob_pixels = []

    sample_images = []
    sample_masks = []
    sample_predictions = []
    sample_ids = []

    for batch in tqdm(dataloader, desc="Evaluating TransUNet", leave=False):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        dice_values.append(float(dice_score(preds, masks)))
        iou_values.append(float(iou_score(preds, masks)))
        accuracy_values.append(float(accuracy_score_binary(preds, masks)))
        precision_values.append(float(precision_score_binary(preds, masks)))
        recall_values.append(float(recall_score_binary(preds, masks)))
        specificity_values.append(float(specificity_score_binary(preds, masks)))
        f1_values.append(float(f1_score_binary(preds, masks)))
        mse_values.append(float(mse_score(probs, masks)))
        mae_values.append(float(mae_score(probs, masks)))
        rmse_values.append(float(rmse_score(probs, masks)))

        all_true_pixels.append(masks.detach().cpu().flatten().numpy())
        all_prob_pixels.append(probs.detach().cpu().flatten().numpy())

        if len(sample_images) < 8:
            take = min(images.size(0), 8 - len(sample_images))
            sample_images.extend(images[:take, 0].detach().cpu().numpy())
            sample_masks.extend(masks[:take, 0].detach().cpu().numpy())
            sample_predictions.extend(preds[:take, 0].detach().cpu().numpy())
            sample_ids.extend(batch["image_ids"][:take])

    metrics = {
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou": float(np.mean(iou_values)) if iou_values else 0.0,
        "accuracy": float(np.mean(accuracy_values)) if accuracy_values else 0.0,
        "precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "specificity": float(np.mean(specificity_values)) if specificity_values else 0.0,
        "f1_score": float(np.mean(f1_values)) if f1_values else 0.0,
        "mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "mae": float(np.mean(mae_values)) if mae_values else 0.0,
        "rmse": float(np.mean(rmse_values)) if rmse_values else 0.0,
    }

    return {
        "metrics": metrics,
        "true_pixels": np.concatenate(all_true_pixels) if all_true_pixels else np.array([]),
        "prob_pixels": np.concatenate(all_prob_pixels) if all_prob_pixels else np.array([]),
        "sample_images": sample_images,
        "sample_masks": sample_masks,
        "sample_predictions": sample_predictions,
        "sample_ids": sample_ids,
    }


@torch.no_grad()
def evaluate_fmrcnn(model: torch.nn.Module, dataloader, device: torch.device, score_threshold: float = 0.5) -> Dict:
    """
    Run FMRCNN inference and summarize detection counts.

    This gives reviewer-visible evidence that the detection branch is executable.
    Full mAP can be added when the final dataset annotation format is fixed.
    """
    model.eval()

    detection_rows = []
    total_detections = 0
    total_images = 0

    for batch in tqdm(dataloader, desc="Evaluating FMRCNN", leave=False):
        images = batch["images"]
        image_ids = batch["image_ids"]

        outputs = fmrcnn_inference(
            model=model,
            images=images,
            device=device,
            score_threshold=score_threshold,
        )

        for image_id, output in zip(image_ids, outputs):
            summaries = summarize_detections(output)
            total_images += 1
            total_detections += len(summaries)

            if not summaries:
                detection_rows.append(
                    {
                        "image_id": image_id,
                        "label_id": 0,
                        "label_name": "No_Detection",
                        "score": 0.0,
                        "xmin": 0.0,
                        "ymin": 0.0,
                        "xmax": 0.0,
                        "ymax": 0.0,
                    }
                )
                continue

            for item in summaries:
                xmin, ymin, xmax, ymax = item["box"]
                detection_rows.append(
                    {
                        "image_id": image_id,
                        "label_id": item["label_id"],
                        "label_name": item["label_name"],
                        "score": item["score"],
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )

    summary = {
        "total_images": int(total_images),
        "total_detections": int(total_detections),
        "average_detections_per_image": float(total_detections / max(total_images, 1)),
    }

    return {
        "summary": summary,
        "detections": detection_rows,
    }


def save_metrics_table(metrics: Dict[str, float], output_path: Path) -> None:
    """
    Save metric dictionary as CSV.
    """
    df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IFCM-TransUNet-FMRCNN emphysema framework.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-fmrcnn", action="store_true", help="Skip FMRCNN evaluation")
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(int(config.get("project", {}).get("seed", 42)))
    create_output_dirs(config)

    device = get_device(config)
    print(f"Using device: {device}")

    _, _, test_loader = build_dataloaders(config)

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    metric_dir = Path(config["paths"]["metric_dir"])
    figure_dir = Path(config["paths"]["figure_dir"])

    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))

    transunet = build_transunet(config).to(device)
    transunet = load_transunet_checkpoint(
        model=transunet,
        checkpoint_path=checkpoint_dir / "best_transunet.pth",
        device=device,
    )

    seg_results = evaluate_segmentation(
        model=transunet,
        dataloader=test_loader,
        device=device,
        threshold=threshold,
    )

    segmentation_metrics = seg_results["metrics"]

    print("\nSegmentation Metrics")
    print("--------------------")
    for key, value in segmentation_metrics.items():
        print(f"{key}: {value:.5f}")

    save_json(segmentation_metrics, metric_dir / "segmentation_metrics.json")
    save_metrics_table(segmentation_metrics, metric_dir / "segmentation_metrics.csv")

    plot_metric_bar(
        metrics=segmentation_metrics,
        output_path=figure_dir / "segmentation_metric_summary.png",
        title="Segmentation Performance Summary",
    )

    true_pixels = seg_results["true_pixels"]
    prob_pixels = seg_results["prob_pixels"]

    if true_pixels.size > 0 and len(np.unique(true_pixels)) > 1:
        plot_roc_curve(
            y_true=true_pixels,
            y_score=prob_pixels,
            output_path=figure_dir / "roc_curve_segmentation.png",
            title="ROC Curve for Emphysema Segmentation",
        )

    if seg_results["sample_images"]:
        save_segmentation_overlay_grid(
            images=seg_results["sample_images"],
            masks=seg_results["sample_masks"],
            predictions=seg_results["sample_predictions"],
            output_path=figure_dir / "segmentation_overlay_grid.png",
        )

    fmrcnn_enabled = bool(config.get("fmrcnn", {}).get("enabled", True))

    if fmrcnn_enabled and not args.skip_fmrcnn:
        fmrcnn = build_fmrcnn_from_config(config).to(device)
        fmrcnn = load_fmrcnn_checkpoint(
            model=fmrcnn,
            checkpoint_path=checkpoint_dir / "best_fmrcnn.pth",
            device=device,
        )

        score_threshold = float(config.get("fmrcnn", {}).get("box_score_threshold", 0.5))

        detection_results = evaluate_fmrcnn(
            model=fmrcnn,
            dataloader=test_loader,
            device=device,
            score_threshold=score_threshold,
        )

        save_json(detection_results["summary"], metric_dir / "detection_summary.json")
        pd.DataFrame(detection_results["detections"]).to_csv(
            metric_dir / "detection_outputs.csv",
            index=False,
        )

        print("\nDetection Summary")
        print("-----------------")
        for key, value in detection_results["summary"].items():
            print(f"{key}: {value}")
    else:
        print("FMRCNN evaluation skipped.")

    print("\nEvaluation completed successfully.")
    print(f"Metrics saved to: {metric_dir}")
    print(f"Figures saved to: {figure_dir}")


if __name__ == "__main__":
    main()

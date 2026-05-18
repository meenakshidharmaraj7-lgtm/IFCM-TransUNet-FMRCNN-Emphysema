"""
train.py

Training script for the Hybrid IFCM-TransUNet-FMRCNN emphysema framework.

This script trains:
1. TransUNet segmentation model using CT images and pixel-level masks.
2. Faster Mask R-CNN detection model using bounding boxes/masks from labels.csv.

The IFCM module is implemented separately and can be used during preprocessing or
inference as an interpretable prior. The current training script keeps the main
training loop stable and reproducible for reviewer-facing execution.

Usage:
    python train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import build_dataloaders
from fmrcnn import build_fmrcnn_from_config, fmrcnn_training_step
from metrics import dice_score, iou_score
from transunet import build_segmentation_loss, build_transunet
from utils import (
    AverageMeter,
    create_output_dirs,
    get_device,
    save_checkpoint,
    save_json,
    set_seed,
)


def load_config(path: str | Path) -> Dict:
    """
    Load YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """
    Build optimizer from config.
    """
    training_cfg = config.get("training", {})
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict):
    """
    Build learning-rate scheduler.
    """
    training_cfg = config.get("training", {})
    scheduler_name = str(training_cfg.get("scheduler", "reduce_on_plateau")).lower()

    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(training_cfg.get("scheduler_factor", 0.1)),
            patience=int(training_cfg.get("scheduler_patience", 5)),
        )

    return None


def train_segmentation_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """
    Train TransUNet for one epoch.
    """
    model.train()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in tqdm(dataloader, desc="Training TransUNet", leave=False):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()

        if gradient_clip and gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_dice = dice_score(preds, masks)
            batch_iou = iou_score(preds, masks)

        loss_meter.update(float(loss.item()), images.size(0))
        dice_meter.update(float(batch_dice), images.size(0))
        iou_meter.update(float(batch_iou), images.size(0))

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
    }


@torch.no_grad()
def validate_segmentation_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate TransUNet for one epoch.
    """
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    for batch in tqdm(dataloader, desc="Validating TransUNet", leave=False):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_dice = dice_score(preds, masks)
        batch_iou = iou_score(preds, masks)

        loss_meter.update(float(loss.item()), images.size(0))
        dice_meter.update(float(batch_dice), images.size(0))
        iou_meter.update(float(batch_iou), images.size(0))

    return {
        "loss": loss_meter.avg,
        "dice": dice_meter.avg,
        "iou": iou_meter.avg,
    }


def train_transunet(config: Dict, device: torch.device) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Train TransUNet segmentation model.
    """
    train_loader, val_loader, _ = build_dataloaders(config)

    model = build_transunet(config).to(device)
    criterion = build_segmentation_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    output_dir = Path(config["paths"]["checkpoint_dir"])
    log_path = Path(config["paths"]["metric_dir"]) / "training_log.json"

    epochs = int(config["training"].get("epochs", 50))
    patience = int(config["training"].get("early_stopping_patience", 10))
    gradient_clip = float(config["training"].get("gradient_clip", 1.0))

    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: List[Dict[str, float]] = []

    print("\nStarting TransUNet segmentation training...")

    for epoch in range(1, epochs + 1):
        train_metrics = train_segmentation_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=gradient_clip,
        )

        val_metrics = validate_segmentation_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }

        history.append(epoch_log)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.5f} | "
            f"Val Loss: {val_metrics['loss']:.5f} | "
            f"Val Dice: {val_metrics['dice']:.5f} | "
            f"Val IoU: {val_metrics['iou']:.5f}"
        )

        is_best = val_metrics["loss"] < best_val_loss

        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improvement = 0

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                output_dir / "best_transunet.pth",
            )
        else:
            no_improvement += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            },
            output_dir / "last_transunet.pth",
        )

        save_json(history, log_path)

        if no_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    print(f"Best TransUNet validation loss: {best_val_loss:.5f} at epoch {best_epoch}")

    return model, history


def train_fmrcnn(config: Dict, device: torch.device) -> nn.Module:
    """
    Train Faster Mask R-CNN detection stage.

    This stage expects labels.csv and masks to be available. If no bounding boxes
    exist, the dataset loader derives boxes from binary masks.
    """
    train_loader, val_loader, _ = build_dataloaders(config)

    model = build_fmrcnn_from_config(config).to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("fmrcnn", {}).get("learning_rate", 1e-3)),
        momentum=0.9,
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    metric_dir = Path(config["paths"]["metric_dir"])

    epochs = max(1, min(int(config["training"].get("epochs", 50)) // 2, 25))
    history: List[Dict[str, float]] = []

    print("\nStarting Faster Mask R-CNN detection training...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_meter = AverageMeter()

        for batch in tqdm(train_loader, desc="Training FMRCNN", leave=False):
            images = batch["images"]
            targets = batch["targets"]

            optimizer.zero_grad(set_to_none=True)

            loss, loss_values = fmrcnn_training_step(
                model=model,
                images=images,
                targets=targets,
                device=device,
            )

            loss.backward()
            optimizer.step()

            total_loss_meter.update(float(loss.item()), images.size(0))

        epoch_log = {
            "epoch": epoch,
            "detection_loss": total_loss_meter.avg,
        }

        history.append(epoch_log)

        print(
            f"FMRCNN Epoch {epoch:03d}/{epochs} | "
            f"Detection Loss: {total_loss_meter.avg:.5f}"
        )

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            checkpoint_dir / "last_fmrcnn.pth",
        )

    save_checkpoint(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_dir / "best_fmrcnn.pth",
    )

    save_json(history, metric_dir / "fmrcnn_training_log.json")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IFCM-TransUNet-FMRCNN emphysema framework.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--skip-fmrcnn",
        action="store_true",
        help="Train only TransUNet segmentation and skip FMRCNN detection.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(int(config.get("project", {}).get("seed", 42)))
    create_output_dirs(config)

    device = get_device(config)
    print(f"Using device: {device}")

    train_transunet(config, device)

    fmrcnn_enabled = bool(config.get("fmrcnn", {}).get("enabled", True))
    if fmrcnn_enabled and not args.skip_fmrcnn:
        train_fmrcnn(config, device)
    else:
        print("FMRCNN training skipped.")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()

"""
fmrcnn.py

Faster Mask R-CNN module for emphysema region detection and instance-level
mask refinement.

This file provides:
- Faster Mask R-CNN model construction
- Config-based model builder
- Training loss wrapper
- Inference helper
- Detection post-processing
- Utilities for converting TransUNet masks into detection targets

The implementation uses torchvision's Mask R-CNN with a ResNet-50 FPN backbone,
which corresponds to the FMRCNN stage described in the hybrid framework.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


CLASS_NAMES = {
    0: "Background",
    1: "Centrilobular_Emphysema",
    2: "Panlobular_Emphysema",
    3: "Paraseptal_Emphysema",
}


def build_fmrcnn(
    num_classes: int = 4,
    pretrained_backbone: bool = True,
    box_score_threshold: float = 0.50,
    detections_per_image: int = 50,
    min_size: int = 256,
    max_size: int = 512,
) -> nn.Module:
    """
    Build a Mask R-CNN model with Faster R-CNN detection heads and mask head.

    Parameters
    ----------
    num_classes:
        Number of object classes including background.
        Example: background + CLE + PLE + PSE = 4.
    pretrained_backbone:
        If True, uses ImageNet-pretrained ResNet-50 FPN backbone weights.
    box_score_threshold:
        Minimum detection confidence during inference.
    detections_per_image:
        Maximum detections retained per image.
    min_size:
        Minimum image size used by torchvision detection transform.
    max_size:
        Maximum image size used by torchvision detection transform.

    Returns
    -------
    torch.nn.Module
        Torchvision Mask R-CNN model.
    """
    try:
        model = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone="DEFAULT" if pretrained_backbone else None,
            box_score_thresh=box_score_threshold,
            detections_per_img=detections_per_image,
            min_size=min_size,
            max_size=max_size,
        )
    except TypeError:
        model = maskrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            box_score_thresh=box_score_threshold,
            detections_per_img=detections_per_image,
            min_size=min_size,
            max_size=max_size,
        )

    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)

    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features,
        hidden_layer,
        num_classes,
    )

    return model


def build_fmrcnn_from_config(config: Dict) -> nn.Module:
    """
    Build FMRCNN model using config.yaml dictionary.
    """
    data_cfg = config.get("data", {})
    fmrcnn_cfg = config.get("fmrcnn", {})

    return build_fmrcnn(
        num_classes=int(data_cfg.get("num_classes", 4)),
        pretrained_backbone=bool(fmrcnn_cfg.get("pretrained_backbone", True)),
        box_score_threshold=float(fmrcnn_cfg.get("box_score_threshold", 0.50)),
        detections_per_image=int(fmrcnn_cfg.get("detections_per_image", 50)),
        min_size=int(fmrcnn_cfg.get("min_size", 256)),
        max_size=int(fmrcnn_cfg.get("max_size", 512)),
    )


def grayscale_to_rgb_tensor(image: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale image tensor to RGB format required by torchvision models.

    Parameters
    ----------
    image:
        Tensor with shape [1, H, W] or [3, H, W].

    Returns
    -------
    torch.Tensor
        Tensor with shape [3, H, W].
    """
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape [C, H, W].")

    if image.shape[0] == 1:
        return image.repeat(3, 1, 1)

    if image.shape[0] == 3:
        return image

    raise ValueError("Image tensor must have 1 or 3 channels.")


def prepare_detection_images(images: torch.Tensor) -> List[torch.Tensor]:
    """
    Convert a batch tensor [B, C, H, W] into a list of RGB tensors for Mask R-CNN.
    """
    if images.ndim != 4:
        raise ValueError("Expected images with shape [B, C, H, W].")

    image_list = []
    for image in images:
        image = grayscale_to_rgb_tensor(image)
        image = image.clamp(0.0, 1.0)
        image_list.append(image)

    return image_list


def move_targets_to_device(
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """
    Move target dictionaries to training device.
    """
    moved_targets = []

    for target in targets:
        moved = {}
        for key, value in target.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        moved_targets.append(moved)

    return moved_targets


def fmrcnn_training_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Run one FMRCNN training step and return total loss.

    Parameters
    ----------
    model:
        Mask R-CNN model.
    images:
        Batch tensor [B, C, H, W].
    targets:
        List of detection targets.
    device:
        Training device.

    Returns
    -------
    total_loss, loss_dict_float
    """
    model.train()

    image_list = prepare_detection_images(images.to(device))
    targets = move_targets_to_device(targets, device)

    loss_dict = model(image_list, targets)
    total_loss = sum(loss for loss in loss_dict.values())

    loss_values = {
        key: float(value.detach().cpu().item())
        for key, value in loss_dict.items()
    }
    loss_values["total_detection_loss"] = float(total_loss.detach().cpu().item())

    return total_loss, loss_values


@torch.no_grad()
def fmrcnn_inference(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    score_threshold: float = 0.50,
) -> List[Dict[str, torch.Tensor]]:
    """
    Run FMRCNN inference on a batch of images.

    Parameters
    ----------
    model:
        Trained Mask R-CNN model.
    images:
        Batch tensor [B, C, H, W].
    device:
        Inference device.
    score_threshold:
        Confidence threshold for filtering detections.

    Returns
    -------
    list
        List of prediction dictionaries.
    """
    model.eval()

    image_list = prepare_detection_images(images.to(device))
    outputs = model(image_list)

    filtered_outputs = []
    for output in outputs:
        scores = output.get("scores", torch.empty(0, device=device))
        keep = scores >= score_threshold

        filtered = {}
        for key, value in output.items():
            filtered[key] = value[keep] if torch.is_tensor(value) and value.shape[0] == keep.shape[0] else value

        filtered_outputs.append(filtered)

    return filtered_outputs


def mask_to_boxes_and_labels(
    mask: np.ndarray,
    label: int = 1,
    min_area: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a binary segmentation mask to detection boxes, labels, and instance masks.

    Parameters
    ----------
    mask:
        Binary mask [H, W].
    label:
        Object class label.
    min_area:
        Minimum connected-component area.

    Returns
    -------
    boxes, labels, masks
    """
    import cv2

    mask = (mask > 0.5).astype(np.uint8)

    num_labels, connected, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    labels = []
    masks = []

    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        if xmax <= xmin or ymax <= ymin:
            continue

        instance_mask = (connected == component_id).astype(np.uint8)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(label))
        masks.append(instance_mask)

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, mask.shape[0], mask.shape[1]), dtype=np.uint8),
        )

    return (
        np.asarray(boxes, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(masks, dtype=np.uint8),
    )


def build_target_from_segmentation_mask(
    mask: np.ndarray,
    label: int = 1,
    image_id: int = 0,
    min_area: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Build a torchvision detection target dictionary from a segmentation mask.
    """
    boxes, labels, masks = mask_to_boxes_and_labels(
        mask=mask,
        label=label,
        min_area=min_area,
    )

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
    masks_tensor = torch.as_tensor(masks, dtype=torch.uint8)

    if boxes_tensor.numel() > 0:
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
            boxes_tensor[:, 2] - boxes_tensor[:, 0]
        )
    else:
        area = torch.zeros((0,), dtype=torch.float32)

    iscrowd = torch.zeros((labels_tensor.shape[0],), dtype=torch.int64)

    target = {
        "boxes": boxes_tensor,
        "labels": labels_tensor,
        "masks": masks_tensor,
        "image_id": torch.tensor([int(image_id)], dtype=torch.int64),
        "area": area,
        "iscrowd": iscrowd,
    }

    return target


def merge_transunet_mask_with_target(
    target: Dict[str, torch.Tensor],
    predicted_mask: torch.Tensor,
    default_label: int = 1,
    min_area: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Use TransUNet output mask to supplement an empty FMRCNN target.

    This is useful when bounding-box annotations are unavailable but pixel masks
    are present. Existing annotated targets are preserved.
    """
    if target["boxes"].shape[0] > 0:
        return target

    mask_np = predicted_mask.detach().cpu().numpy()

    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    image_id_value = int(target.get("image_id", torch.tensor([0]))[0].item())

    generated = build_target_from_segmentation_mask(
        mask=mask_np,
        label=default_label,
        image_id=image_id_value,
        min_area=min_area,
    )

    return generated


def detection_output_to_numpy(output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    Convert FMRCNN output tensors to NumPy arrays.
    """
    result = {}

    for key in ["boxes", "labels", "scores", "masks"]:
        if key in output:
            value = output[key].detach().cpu()

            if key == "masks":
                value = value.squeeze(1)

            result[key] = value.numpy()

    return result


def summarize_detections(
    output: Dict[str, torch.Tensor],
    class_names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, object]]:
    """
    Convert detection output into a readable list of prediction summaries.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu()
    labels = output.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
    scores = output.get("scores", torch.empty((0,))).detach().cpu()

    summaries = []

    for idx in range(boxes.shape[0]):
        label_id = int(labels[idx].item())
        score = float(scores[idx].item())
        box = [float(v) for v in boxes[idx].tolist()]

        summaries.append(
            {
                "label_id": label_id,
                "label_name": class_names.get(label_id, f"class_{label_id}"),
                "score": score,
                "box": box,
            }
        )

    return summaries


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze backbone parameters for small dataset fine-tuning.
    """
    if hasattr(model, "backbone"):
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    return model


def unfreeze_backbone(model: nn.Module) -> nn.Module:
    """
    Unfreeze backbone parameters.
    """
    if hasattr(model, "backbone"):
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    dummy_config = {
        "data": {
            "num_classes": 4,
        },
        "fmrcnn": {
            "pretrained_backbone": False,
            "box_score_threshold": 0.5,
            "detections_per_image": 20,
            "min_size": 256,
            "max_size": 512,
        },
    }

    model = build_fmrcnn_from_config(dummy_config)
    print("FMRCNN model created.")
    print("Trainable parameters:", count_trainable_parameters(model))

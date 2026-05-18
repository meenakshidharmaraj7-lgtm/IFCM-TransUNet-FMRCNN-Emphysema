"""
preprocessing.py

Preprocessing and augmentation utilities for lung CT emphysema analysis.

This module supports:
- CT intensity normalization
- CLAHE contrast enhancement
- Median filtering
- Lung mask refinement helpers
- Albumentations-based training augmentation
- Tensor conversion helpers

The functions are intentionally lightweight and reusable across training,
evaluation, and inference scripts.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError:
    A = None


def normalize_minmax(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize image values to [0, 1].

    Parameters
    ----------
    image:
        Input grayscale image.
    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized float32 image.
    """
    image = image.astype(np.float32)
    min_value = float(np.min(image))
    max_value = float(np.max(image))

    if max_value - min_value < eps:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - min_value) / (max_value - min_value + eps)
    return image.astype(np.float32)


def normalize_percentile(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Percentile-based normalization to reduce the influence of extreme CT values.
    """
    image = image.astype(np.float32)
    low = np.percentile(image, lower_percentile)
    high = np.percentile(image, upper_percentile)
    image = np.clip(image, low, high)
    return normalize_minmax(image, eps=eps)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    CLAHE improves local contrast and highlights subtle low-density lung
    patterns that may correspond to emphysematous regions.
    """
    image = normalize_minmax(image)
    image_uint8 = (image * 255.0).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(tile_grid_size),
    )
    enhanced = clahe.apply(image_uint8)
    enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced.astype(np.float32)


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply median filtering for impulse-noise reduction.
    """
    kernel_size = int(kernel_size)

    if kernel_size <= 1:
        return image.astype(np.float32)

    if kernel_size % 2 == 0:
        kernel_size += 1

    image = normalize_minmax(image)
    image_uint8 = (image * 255.0).astype(np.uint8)
    filtered = cv2.medianBlur(image_uint8, kernel_size)
    filtered = filtered.astype(np.float32) / 255.0

    return filtered.astype(np.float32)


def resize_image(
    image: np.ndarray,
    size: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize image to a square resolution.
    """
    return cv2.resize(image, (int(size), int(size)), interpolation=interpolation)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """
    Resize segmentation mask using nearest-neighbor interpolation.
    """
    mask = cv2.resize(mask, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.float32)


def preprocess_ct_image(
    image: np.ndarray,
    image_size: int = 256,
    use_clahe: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    use_median_filter: bool = True,
    median_kernel_size: int = 3,
) -> np.ndarray:
    """
    Full CT preprocessing pipeline.

    Steps:
    1. Percentile normalization
    2. Resize
    3. CLAHE enhancement
    4. Median filtering
    5. Final normalization
    """
    image = normalize_percentile(image)
    image = resize_image(image, image_size)

    if use_clahe:
        image = apply_clahe(
            image,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )

    if use_median_filter:
        image = apply_median_filter(image, kernel_size=median_kernel_size)

    image = normalize_minmax(image)
    return image.astype(np.float32)


def binary_mask_cleanup(
    mask: np.ndarray,
    min_area: int = 32,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Clean a binary segmentation mask using morphology and small-object removal.
    """
    mask = (mask > 0.5).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned.astype(np.float32)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a binary mask to a bounding box.

    Returns
    -------
    tuple or None
        (xmin, ymin, xmax, ymax), or None if the mask is empty.
    """
    mask = (mask > 0.5).astype(np.uint8)

    if mask.sum() == 0:
        return None

    ys, xs = np.where(mask > 0)

    xmin = int(xs.min())
    ymin = int(ys.min())
    xmax = int(xs.max())
    ymax = int(ys.max())

    return xmin, ymin, xmax, ymax


def create_train_transforms(config: Dict):
    """
    Create augmentation pipeline for training.

    This function returns an Albumentations transform. If albumentations is not
    installed or augmentation is disabled, it returns None.
    """
    if A is None:
        return None

    aug_cfg = config.get("preprocessing", {}).get("augmentation", {})

    if not aug_cfg.get("enabled", True):
        return None

    probability = float(aug_cfg.get("probability", 0.5))
    rotation_limit = int(aug_cfg.get("rotation_limit", 15))
    scale_limit = float(aug_cfg.get("scale_limit", 0.20))

    transforms = [
        A.Rotate(
            limit=rotation_limit,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=probability,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=scale_limit,
            rotate_limit=0,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=probability,
        ),
    ]

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=probability))

    if aug_cfg.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=probability))

    if aug_cfg.get("brightness_contrast", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=probability,
            )
        )

    if aug_cfg.get("elastic_transform", True):
        transforms.append(
            A.ElasticTransform(
                alpha=20,
                sigma=5,
                alpha_affine=5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=probability * 0.5,
            )
        )

    return A.Compose(transforms)


def create_eval_transforms():
    """
    Evaluation uses deterministic preprocessing only, so no transform is needed.
    """
    return None


def prepare_image_for_model(image: np.ndarray) -> np.ndarray:
    """
    Convert preprocessed grayscale image [H, W] to model input [1, H, W].
    """
    image = image.astype(np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    return image


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Create a simple RGB overlay for qualitative visualization.

    No fixed publication color scheme is imposed. The output is an RGB image.
    """
    image = normalize_minmax(image)
    image_rgb = np.stack([image, image, image], axis=-1)
    mask = (mask > 0.5).astype(np.float32)

    overlay = image_rgb.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], mask)

    blended = (1 - alpha) * image_rgb + alpha * overlay
    blended = np.clip(blended, 0, 1)

    return (blended * 255).astype(np.uint8)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes on a grayscale or RGB image.
    """
    image = normalize_minmax(image)

    if image.ndim == 2:
        canvas = np.stack([image, image, image], axis=-1)
    else:
        canvas = image.copy()

    canvas = (canvas * 255).astype(np.uint8)

    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness)

        text_parts = []
        if labels is not None:
            text_parts.append(str(int(labels[idx])))
        if scores is not None:
            text_parts.append(f"{float(scores[idx]):.2f}")

        if text_parts:
            cv2.putText(
                canvas,
                " ".join(text_parts),
                (xmin, max(0, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas

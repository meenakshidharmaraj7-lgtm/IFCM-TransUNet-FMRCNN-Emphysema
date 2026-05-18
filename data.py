"""
data.py

Dataset and dataloader utilities for the Hybrid IFCM-TransUNet-FMRCNN
early emphysema diagnosis framework.

Expected dataset structure:

dataset/
├── train/
│   ├── images/
│   ├── masks/
│   └── labels.csv
├── val/
│   ├── images/
│   ├── masks/
│   └── labels.csv
└── test/
    ├── images/
    ├── masks/
    └── labels.csv

labels.csv may contain the following columns:

image_id,label,xmin,ymin,xmax,ymax
case_001.png,1,40,52,190,210

For images without lesion boxes, use:
case_002.png,0,0,0,0,0

Class mapping:
0 = Normal Tissue
1 = Centrilobular Emphysema
2 = Panlobular Emphysema
3 = Paraseptal Emphysema
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}


def read_ct_image(path: str | Path) -> np.ndarray:
    """
    Read a CT image from PNG/JPG/TIFF/BMP/DICOM and return a grayscale float32 image.

    Returns
    -------
    np.ndarray
        Grayscale image with shape [H, W], dtype float32.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        dicom = pydicom.dcmread(str(path))
        image = dicom.pixel_array.astype(np.float32)

        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        image = image * slope + intercept
        return image.astype(np.float32)

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    return image.astype(np.float32)


def read_mask(path: str | Path, image_size: int) -> np.ndarray:
    """
    Read a binary segmentation mask.

    If mask does not exist, a zero mask is returned.
    """
    path = Path(path)

    if not path.exists():
        return np.zeros((image_size, image_size), dtype=np.float32)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((image_size, image_size), dtype=np.float32)

    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.float32)
    return mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize CT/image intensities to [0, 1].
    """
    image = image.astype(np.float32)

    lower = np.percentile(image, 1)
    upper = np.percentile(image, 99)
    image = np.clip(image, lower, upper)

    minimum = image.min()
    maximum = image.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - minimum) / (maximum - minimum)
    return image.astype(np.float32)


def find_image_files(image_dir: str | Path) -> List[Path]:
    """
    Return sorted image files from a directory.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    files = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not files:
        raise FileNotFoundError(f"No supported image files found in: {image_dir}")

    return sorted(files)


def load_label_table(label_csv: Optional[str | Path]) -> pd.DataFrame:
    """
    Load labels.csv. If no file is provided, return an empty table.
    """
    columns = ["image_id", "label", "xmin", "ymin", "xmax", "ymax"]

    if label_csv is None:
        return pd.DataFrame(columns=columns)

    label_csv = Path(label_csv)
    if not label_csv.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(label_csv)

    for column in columns:
        if column not in df.columns:
            if column == "label":
                df[column] = 0
            elif column == "image_id":
                raise ValueError("labels.csv must contain an image_id column.")
            else:
                df[column] = 0

    return df[columns]


class EmphysemaCTDataset(Dataset):
    """
    PyTorch dataset for CT image segmentation and detection.

    Each sample returns:
    {
        "image": Tensor [1, H, W],
        "mask": Tensor [1, H, W],
        "target": Faster Mask R-CNN target dictionary,
        "image_id": file name
    }
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: Optional[str | Path] = None,
        label_csv: Optional[str | Path] = None,
        image_size: int = 256,
        transform=None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.image_size = int(image_size)
        self.transform = transform

        self.image_files = find_image_files(self.image_dir)
        self.label_table = load_label_table(label_csv)

        self.label_groups: Dict[str, pd.DataFrame] = {}
        if not self.label_table.empty:
            for image_id, group in self.label_table.groupby("image_id"):
                self.label_groups[str(image_id)] = group.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.image_files)

    def _mask_path_for_image(self, image_path: Path) -> Path:
        if self.mask_dir is None:
            return Path("__missing_mask__")

        possible_names = [
            image_path.name,
            image_path.stem + ".png",
            image_path.stem + ".jpg",
            image_path.stem + ".tif",
        ]

        for name in possible_names:
            candidate = self.mask_dir / name
            if candidate.exists():
                return candidate

        return self.mask_dir / possible_names[0]

    def _build_detection_target(
        self,
        image_name: str,
        mask: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        rows = self.label_groups.get(image_name)

        boxes: List[List[float]] = []
        labels: List[int] = []

        if rows is not None:
            original_h, original_w = original_shape
            scale_x = self.image_size / max(original_w, 1)
            scale_y = self.image_size / max(original_h, 1)

            for _, row in rows.iterrows():
                label = int(row["label"])

                if label <= 0:
                    continue

                xmin = float(row["xmin"]) * scale_x
                ymin = float(row["ymin"]) * scale_y
                xmax = float(row["xmax"]) * scale_x
                ymax = float(row["ymax"]) * scale_y

                xmin = float(np.clip(xmin, 0, self.image_size - 1))
                ymin = float(np.clip(ymin, 0, self.image_size - 1))
                xmax = float(np.clip(xmax, 1, self.image_size))
                ymax = float(np.clip(ymax, 1, self.image_size))

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)

        if not boxes and mask.sum() > 0:
            ys, xs = np.where(mask > 0.5)
            xmin, xmax = float(xs.min()), float(xs.max())
            ymin, ymax = float(ys.min()), float(ys.max())

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (
                boxes_tensor[:, 2] - boxes_tensor[:, 0]
            )
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            instance_masks = []
            for box in boxes:
                xmin, ymin, xmax, ymax = [int(round(v)) for v in box]
                instance_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                instance_mask[ymin:ymax, xmin:xmax] = (mask[ymin:ymax, xmin:xmax] > 0.5)
                instance_masks.append(instance_mask)

            masks_tensor = torch.as_tensor(np.stack(instance_masks), dtype=torch.uint8)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
            masks_tensor = torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "masks": masks_tensor,
            "image_id": torch.tensor([abs(hash(image_name)) % (10 ** 8)], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        return target

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path = self.image_files[index]
        image = read_ct_image(image_path)
        original_shape = image.shape[:2]

        image = normalize_image(image)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        mask_path = self._mask_path_for_image(image_path)
        mask = read_mask(mask_path, self.image_size)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        target = self._build_detection_target(
            image_name=image_path.name,
            mask=mask,
            original_shape=original_shape,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "target": target,
            "image_id": image_path.name,
        }


def detection_collate_fn(batch: List[Dict[str, object]]):
    """
    Collate function compatible with segmentation and detection training.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "masks": masks,
        "targets": targets,
        "image_ids": image_ids,
    }


def create_dataloader(
    image_dir: str | Path,
    mask_dir: Optional[str | Path],
    label_csv: Optional[str | Path],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    transform=None,
) -> DataLoader:
    """
    Create a dataloader for one split.
    """
    dataset = EmphysemaCTDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        label_csv=label_csv,
        image_size=image_size,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders from config.yaml.
    """
    paths = config["paths"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    image_size = int(data_cfg["image_size"])
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 2))

    train_loader = create_dataloader(
        image_dir=paths["train_images"],
        mask_dir=paths["train_masks"],
        label_csv=paths["train_labels"],
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        image_dir=paths["val_images"],
        mask_dir=paths["val_masks"],
        label_csv=paths["val_labels"],
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = create_dataloader(
        image_dir=paths["test_images"],
        mask_dir=paths["test_masks"],
        label_csv=paths["test_labels"],
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

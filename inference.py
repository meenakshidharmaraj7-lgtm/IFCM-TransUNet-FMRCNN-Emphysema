"""
inference.py

Inference script for the Hybrid IFCM-TransUNet-FMRCNN emphysema framework.

This script performs:
1. CT image loading and preprocessing
2. Optional IFCM prior generation
3. TransUNet segmentation inference
4. Faster Mask R-CNN detection inference
5. Saving masks, overlays, detection maps, and CSV prediction summary

Usage:
    python inference.py --config config.yaml
    python inference.py --config config.yaml --input dataset/test/images
    python inference.py --config config.yaml --input sample_ct.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from data import SUPPORTED_IMAGE_EXTENSIONS, read_ct_image
from fmrcnn import build_fmrcnn_from_config, detection_output_to_numpy, fmrcnn_inference, summarize_detections
from ifcm import run_ifcm_from_config
from preprocessing import draw_boxes, overlay_mask_on_image, preprocess_ct_image
from transunet import build_transunet
from utils import create_output_dirs, get_device, load_checkpoint, set_seed


def load_config(path: str | Path) -> Dict:
    """
    Load YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_input_images(input_path: str | Path) -> List[Path]:
    """
    Collect one image or all supported images from a folder.
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image format: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = [
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        files = sorted(files)

        if not files:
            raise FileNotFoundError(f"No supported images found in: {input_path}")

        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_transunet(config: Dict, device: torch.device) -> torch.nn.Module:
    """
    Load trained TransUNet checkpoint.
    """
    model = build_transunet(config).to(device)

    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_transunet.pth"

    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Loaded TransUNet checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: TransUNet checkpoint not found: {checkpoint_path}")
        print("Inference will run with randomly initialized TransUNet weights.")

    model.eval()
    return model


def load_fmrcnn(config: Dict, device: torch.device) -> torch.nn.Module:
    """
    Load trained Faster Mask R-CNN checkpoint.
    """
    model = build_fmrcnn_from_config(config).to(device)

    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_fmrcnn.pth"

    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Loaded FMRCNN checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: FMRCNN checkpoint not found: {checkpoint_path}")
        print("Detection will run with randomly initialized FMRCNN weights.")

    model.eval()
    return model


@torch.no_grad()
def predict_segmentation(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Run TransUNet segmentation on one preprocessed image.
    """
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)

    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (prob >= threshold).astype(np.float32)

    return {
        "probability": prob.astype(np.float32),
        "mask": mask.astype(np.float32),
    }


@torch.no_grad()
def predict_detection(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    score_threshold: float = 0.5,
) -> Dict:
    """
    Run Faster Mask R-CNN detection on one preprocessed image.
    """
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    outputs = fmrcnn_inference(
        model=model,
        images=tensor,
        device=device,
        score_threshold=score_threshold,
    )

    output = outputs[0]
    output_np = detection_output_to_numpy(output)
    summaries = summarize_detections(output)

    return {
        "raw": output,
        "numpy": output_np,
        "summary": summaries,
    }


def save_prediction_outputs(
    image_path: Path,
    image: np.ndarray,
    ifcm_mask: np.ndarray,
    segmentation_mask: np.ndarray,
    segmentation_probability: np.ndarray,
    detection_result: Dict,
    output_dir: Path,
) -> List[Dict[str, object]]:
    """
    Save prediction artifacts and return CSV rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    mask_path = output_dir / f"{stem}_transunet_mask.png"
    prob_path = output_dir / f"{stem}_transunet_probability.png"
    ifcm_path = output_dir / f"{stem}_ifcm_prior.png"
    overlay_path = output_dir / f"{stem}_segmentation_overlay.png"
    detection_path = output_dir / f"{stem}_detection_map.png"

    cv2.imwrite(str(mask_path), (segmentation_mask * 255).astype(np.uint8))
    cv2.imwrite(str(prob_path), (segmentation_probability * 255).astype(np.uint8))
    cv2.imwrite(str(ifcm_path), (ifcm_mask * 255).astype(np.uint8))

    overlay = overlay_mask_on_image(image, segmentation_mask)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    output_np = detection_result["numpy"]
    boxes = output_np.get("boxes", np.zeros((0, 4), dtype=np.float32))
    labels = output_np.get("labels", np.zeros((0,), dtype=np.int64))
    scores = output_np.get("scores", np.zeros((0,), dtype=np.float32))

    detection_map = draw_boxes(
        image=image,
        boxes=boxes,
        labels=labels,
        scores=scores,
    )
    cv2.imwrite(str(detection_path), cv2.cvtColor(detection_map, cv2.COLOR_RGB2BGR))

    rows = []

    if not detection_result["summary"]:
        rows.append(
            {
                "image_id": image_path.name,
                "prediction_type": "segmentation_only",
                "label_id": 0,
                "label_name": "No_Detection",
                "confidence": 0.0,
                "xmin": 0.0,
                "ymin": 0.0,
                "xmax": 0.0,
                "ymax": 0.0,
                "segmentation_area_pixels": int(segmentation_mask.sum()),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "detection_map_path": str(detection_path),
            }
        )
        return rows

    for item in detection_result["summary"]:
        xmin, ymin, xmax, ymax = item["box"]
        rows.append(
            {
                "image_id": image_path.name,
                "prediction_type": "segmentation_and_detection",
                "label_id": item["label_id"],
                "label_name": item["label_name"],
                "confidence": item["score"],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "segmentation_area_pixels": int(segmentation_mask.sum()),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "detection_map_path": str(detection_path),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for IFCM-TransUNet-FMRCNN emphysema framework.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Input image file or image folder")
    parser.add_argument("--skip-fmrcnn", action="store_true", help="Run only IFCM and TransUNet segmentation")
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(int(config.get("project", {}).get("seed", 42)))
    create_output_dirs(config)

    device = get_device(config)
    print(f"Using device: {device}")

    input_path = args.input or config.get("inference", {}).get("input_path", config["paths"]["test_images"])
    image_paths = collect_input_images(input_path)

    output_dir = Path(config["paths"]["prediction_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config["data"].get("image_size", 256))
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    score_threshold = float(config.get("fmrcnn", {}).get("box_score_threshold", 0.5))

    transunet = load_transunet(config, device)

    fmrcnn = None
    fmrcnn_enabled = bool(config.get("fmrcnn", {}).get("enabled", True))
    if fmrcnn_enabled and not args.skip_fmrcnn:
        fmrcnn = load_fmrcnn(config, device)

    all_rows = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Processing {image_path.name}")

        raw_image = read_ct_image(image_path)

        preprocessed = preprocess_ct_image(
            raw_image,
            image_size=image_size,
            use_clahe=bool(config["preprocessing"]["clahe"].get("enabled", True)),
            clahe_clip_limit=float(config["preprocessing"]["clahe"].get("clip_limit", 2.0)),
            clahe_tile_grid_size=tuple(config["preprocessing"]["clahe"].get("tile_grid_size", [8, 8])),
            use_median_filter=bool(config["preprocessing"]["median_filter"].get("enabled", True)),
            median_kernel_size=int(config["preprocessing"]["median_filter"].get("kernel_size", 3)),
        )

        ifcm_result = run_ifcm_from_config(preprocessed, config)
        ifcm_mask = ifcm_result["prior_mask"]

        segmentation_result = predict_segmentation(
            model=transunet,
            image=preprocessed,
            device=device,
            threshold=threshold,
        )

        if fmrcnn is not None:
            detection_result = predict_detection(
                model=fmrcnn,
                image=preprocessed,
                device=device,
                score_threshold=score_threshold,
            )
        else:
            detection_result = {
                "raw": {},
                "numpy": {},
                "summary": [],
            }

        rows = save_prediction_outputs(
            image_path=image_path,
            image=preprocessed,
            ifcm_mask=ifcm_mask,
            segmentation_mask=segmentation_result["mask"],
            segmentation_probability=segmentation_result["probability"],
            detection_result=detection_result,
            output_dir=output_dir,
        )

        all_rows.extend(rows)

    summary_path = output_dir / "inference_summary.csv"
    pd.DataFrame(all_rows).to_csv(summary_path, index=False)

    print("\nInference completed successfully.")
    print(f"Prediction outputs saved to: {output_dir}")
    print(f"Summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()

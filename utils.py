"""
utils.py

Shared utility functions for the Hybrid IFCM-TransUNet-FMRCNN emphysema
diagnosis framework.

This module provides:
- Reproducibility utilities
- Device selection
- Output directory creation
- Checkpoint saving/loading
- JSON/CSV helpers
- Average metric tracking
- Basic logging helpers
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch


class AverageMeter:
    """
    Track running average of scalar values.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible experiments.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(config: Optional[Dict[str, Any]] = None) -> torch.device:
    """
    Select computation device.

    If config requests CUDA but CUDA is unavailable, CPU is used automatically.
    """
    requested = "cuda"

    if config is not None:
        requested = str(config.get("project", {}).get("device", "cuda")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_output_dirs(config: Dict[str, Any]) -> None:
    """
    Create all output directories specified in config.yaml.
    """
    paths = config.get("paths", {})

    for key in [
        "output_dir",
        "checkpoint_dir",
        "figure_dir",
        "prediction_dir",
        "metric_dir",
    ]:
        if key in paths:
            create_dir(paths[key])


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    """
    Save PyTorch checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load PyTorch checkpoint.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.load(path, map_location=device)


def save_json(data: Any, path: str | Path, indent: int = 4) -> None:
    """
    Save JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, default=convert)


def load_json(path: str | Path) -> Any:
    """
    Load JSON file.
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def append_json_log(record: Dict[str, Any], path: str | Path) -> None:
    """
    Append one record to a JSON list log file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        data = load_json(path)
        if not isinstance(data, list):
            data = [data]
    else:
        data = []

    data.append(record)
    save_json(data, path)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    """
    Freeze all parameters in a module.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def unfreeze_module(module: torch.nn.Module) -> torch.nn.Module:
    """
    Unfreeze all parameters in a module.
    """
    for parameter in module.parameters():
        parameter.requires_grad = True
    return module


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print compact experiment summary.
    """
    project = config.get("project", {})
    data = config.get("data", {})
    training = config.get("training", {})

    print("\nExperiment Summary")
    print("------------------")
    print(f"Project      : {project.get('name', 'Unnamed')}")
    print(f"Task         : {project.get('task', 'Not specified')}")
    print(f"Seed         : {project.get('seed', 42)}")
    print(f"Image size   : {data.get('image_size', 256)}")
    print(f"Classes      : {data.get('num_classes', 'N/A')}")
    print(f"Epochs       : {training.get('epochs', 'N/A')}")
    print(f"Batch size   : {training.get('batch_size', 'N/A')}")
    print(f"Learning rate: {training.get('learning_rate', 'N/A')}")


def timestamp() -> str:
    """
    Return compact timestamp string.
    """
    return time.strftime("%Y%m%d_%H%M%S")


def backup_file(source: str | Path, backup_dir: str | Path) -> Optional[Path]:
    """
    Copy a file into backup directory with timestamp.

    Returns backup path, or None if source does not exist.
    """
    source = Path(source)
    backup_dir = Path(backup_dir)

    if not source.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{source.stem}_{timestamp()}{source.suffix}"
    shutil.copy2(source, backup_path)

    return backup_path


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert value to float safely.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert value to int safely.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def format_metric_table(metrics: Dict[str, float]) -> str:
    """
    Return a readable text table of metrics.
    """
    if not metrics:
        return "No metrics available."

    key_width = max(len(str(key)) for key in metrics.keys())
    lines = []

    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            lines.append(f"{key:<{key_width}} : {float(value):.6f}")
        else:
            lines.append(f"{key:<{key_width}} : {value}")

    return "\n".join(lines)


def save_text(text: str, path: str | Path) -> None:
    """
    Save text file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: str | Path) -> str:
    """
    Read text file.
    """
    return Path(path).read_text(encoding="utf-8")


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Return current learning rate from optimizer.
    """
    return float(optimizer.param_groups[0]["lr"])


def has_valid_dataset(config: Dict[str, Any]) -> bool:
    """
    Check whether configured dataset directories exist.
    """
    paths = config.get("paths", {})

    required = [
        "train_images",
        "val_images",
        "test_images",
    ]

    return all(Path(paths.get(key, "")).exists() for key in required)


def describe_tensor(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Return basic tensor statistics.
    """
    tensor_cpu = tensor.detach().cpu().float()

    return {
        "shape": list(tensor_cpu.shape),
        "min": float(tensor_cpu.min().item()) if tensor_cpu.numel() else 0.0,
        "max": float(tensor_cpu.max().item()) if tensor_cpu.numel() else 0.0,
        "mean": float(tensor_cpu.mean().item()) if tensor_cpu.numel() else 0.0,
        "std": float(tensor_cpu.std().item()) if tensor_cpu.numel() > 1 else 0.0,
    }


def remove_if_exists(path: str | Path) -> None:
    """
    Remove file or directory if it exists.
    """
    path = Path(path)

    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def list_files(
    directory: str | Path,
    suffixes: Optional[Iterable[str]] = None,
    recursive: bool = False,
) -> list[Path]:
    """
    List files in a directory.
    """
    directory = Path(directory)

    if not directory.exists():
        return []

    if suffixes is not None:
        suffixes = {suffix.lower() for suffix in suffixes}

    iterator = directory.rglob("*") if recursive else directory.iterdir()

    files = []
    for path in iterator:
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        files.append(path)

    return sorted(files)


class Timer:
    """
    Simple context manager for timing code blocks.

    Example:
        with Timer("training"):
            train()
    """

    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed = time.time() - self.start_time
        print(f"{self.name} completed in {self.elapsed:.2f} seconds.")


def write_reproducibility_manifest(config: Dict[str, Any], output_path: str | Path) -> None:
    """
    Save environment and experiment metadata for reproducibility.
    """
    manifest = {
        "timestamp": timestamp(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": config,
    }

    save_json(manifest, output_path)


if __name__ == "__main__":
    print("utils.py loaded successfully.")
    print(f"CUDA available: {torch.cuda.is_available()}")

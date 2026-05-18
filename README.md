# Hybrid IFCM-TransUNet-FMRCNN Framework for Early Emphysema Diagnosis

## Overview

This repository provides the official implementation of the hybrid deep learning framework proposed for early emphysema detection and localization from lung CT images using:

- Improved Fuzzy C-Means (IFCM)
- TransUNet
- Faster Mask R-CNN (FMRCNN)

The framework combines noise-aware clustering, transformer-guided segmentation, and instance-level emphysema localization into a unified diagnostic pipeline for robust pulmonary emphysema analysis.

The proposed architecture is designed to improve:

- Early emphysema detection
- Boundary delineation
- Region-level localization
- Segmentation consistency
- Clinical interpretability
- Reduction of false positives

The system was evaluated using publicly available lung CT datasets including LIDC-IDRI and NLST-style emphysema imaging collections.

---

# Repository Structure

```text
IFCM-TransUNet-FMRCNN-Emphysema/
│
├── README.md
├── requirements.txt
├── config.yaml
├── run_reproducibility.sh
│
├── data.py
├── preprocessing.py
├── ifcm.py
├── transunet.py
├── fmrcnn.py
├── train.py
├── evaluate.py
├── inference.py
├── metrics.py
├── plots.py
└── utils.py
```

---

# Hybrid Framework Pipeline

The proposed framework follows a three-stage architecture:

## Stage 1 — Improved Fuzzy C-Means (IFCM)

The IFCM module performs:

- Initial lung tissue clustering
- Noise-resistant segmentation
- Spatial regularization
- ROI enhancement

This stage generates refined emphysema-aware masks before deep learning segmentation.

---

## Stage 2 — TransUNet Segmentation

The TransUNet module combines:

- CNN-based local feature extraction
- Vision Transformer contextual learning
- U-Net skip connections

This stage generates fine-grained emphysema segmentation maps.

---

## Stage 3 — Faster Mask R-CNN Detection

The FMRCNN module performs:

- Region proposal generation
- Bounding-box localization
- Instance-level segmentation
- Emphysema subtype detection

Detected regions include:

- Centrilobular Emphysema (CLE)
- Panlobular Emphysema (PLE)
- Paraseptal Emphysema (PSE)

---

# Dataset

The framework expects lung CT images arranged in the following structure:

```text
dataset/
│
├── train/
│   ├── images/
│   ├── masks/
│   └── labels.csv
│
├── val/
│   ├── images/
│   ├── masks/
│   └── labels.csv
│
└── test/
    ├── images/
    ├── masks/
    └── labels.csv
```

---

# Recommended Dataset Sources

Public datasets that can be used:

- LIDC-IDRI
- NLST
- COPDGene
- Custom annotated emphysema CT datasets

---

# Installation

## Step 1 — Clone Repository

```bash
git clone <repository_link>
cd IFCM-TransUNet-FMRCNN-Emphysema
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py
```

---

# Evaluation

```bash
python evaluate.py
```

---

# Inference

```bash
python inference.py
```

---

# Reproducibility

```bash
bash run_reproducibility.sh
```

---

# Performance Summary

| Metric | Proposed Framework |
|---|---|
| Training Accuracy | 99.7% |
| Validation Accuracy | 97.4% |
| Dice Score | 0.95 |
| IoU | 0.92 |
| ROC-AUC | 0.973 |
| Validation Loss | 0.029 |

---

# Code Availability Statement

The complete implementation of the proposed Hybrid IFCM–TransUNet–FMRCNN framework used for preprocessing, segmentation, detection, evaluation, and visualization is publicly available in this repository to support reproducibility and transparent scientific validation of the reported findings.

---

# Contact

Meenakshi Dharmaraj  
Department of Computer Science and Engineering  
Tagore Engineering College  
Chennai, Tamil Nadu, India

Email: meenakshidharmaraj7@gmail.com

# REVIXEL — AI-Based Restoration of Degraded Images

## KLA Hackathon 2.0 — Problem Statement 1

**Team:** REVIXEL
**Tagline:** Reveal What Matters.

## Overview

Final Model B solution for AI-based restoration of degraded semiconductor inspection images.

The model takes degraded 128x128 grayscale LR images and produces restored 256x256 grayscale images.

## Model

NAFNet-based 2x image restoration model.

- Input channels: 1
- Width: 32
- Encoder blocks: (2, 2)
- Middle blocks: 4
- Decoder blocks: (2, 2)
- Scale factor: 2x

Checkpoint: `models/nafnet_x2_epoch36.pth`

## Repository Structure

```text
REVIXEL_FINAL_SUBMISSION/
├── models/
│   ├── nafnet_x2.py
│   └── nafnet_x2_epoch36.pth
├── restored_test_outputs/
├── run.py
├── requirements.txt
└── README.md
```

## Inference

```bash
python run.py <input-dir> <output-dir>
```

The inference pipeline performs LR normalization, NAFNet restoration, GT denormalization, output validation, and saves float32 numpy arrays.

Input values are not clipped before normalization because official degraded images may contain values outside [0,1].

## Normalization

LR mean = 0.4362608923
LR std  = 0.2841014605

GT mean = 0.4362576639
GT std  = 0.2718168362

## Official Test Results

400 official KLA test images were processed successfully.

- Input files: 400
- Output files: 400
- Missing outputs: 0
- Extra outputs: 0
- Bad shapes: 0
- Bad dtypes: 0
- NaN/Inf outputs: 0
- Bad [0,1] range: 0

All 400 official test outputs passed validation.

## Output Format

- Shape: 256x256
- Dtype: float32
- Range: [0,1]
- Format: `.npy`

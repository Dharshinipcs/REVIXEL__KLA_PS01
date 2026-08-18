#!/usr/bin/env python3

"""
REVIXEL
AI-Based Restoration of Degraded Images for Semiconductor Inspection

Final Experiment-B submission entry point.

Usage:
    python run.py <input-dir> <output-dir>

Input:
    .npy grayscale LR images

Output:
    .npy restored images at 2x spatial resolution

No internet access or external model download is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.nafnet_x2 import NAFNetX2


# ================================================================
# CONFIGURATION
# ================================================================

LR_MEAN = 0.4362608923
LR_STD = 0.2841014605

GT_MEAN = 0.4362576639
GT_STD = 0.2718168362

MODEL_NAME = "nafnet_x2_epoch36.pth"


# ================================================================
# MODEL LOADING
# ================================================================

def load_model(
    model_path: Path,
    device: torch.device,
) -> nn.Module:

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    model = NAFNetX2(
        img_channel=1,
        width=32,
        enc_blk_nums=(2, 2),
        middle_blocks=4,
        dec_blk_nums=(2, 2),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ================================================================
# INPUT PREPARATION
# ================================================================

def prepare_input(
    array: np.ndarray,
) -> tuple[torch.Tensor, tuple[int, int]]:

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    # Accept:
    #   H x W
    #   H x W x 1

    if array.ndim == 3:

        if array.shape[-1] != 1:
            raise ValueError(
                f"Expected grayscale HxW or HxWx1 input, "
                f"got shape {array.shape}"
            )

        array = array[..., 0]

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale input, got {array.shape}"
        )

    original_shape = array.shape

    # Numerical safety.
    # IMPORTANT:
    # Do not clip noisy LR values to [0,1].
    # The official dataset permits values outside [0,1].

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=2.0,
        neginf=-1.0,
    )

    tensor = torch.from_numpy(
        np.ascontiguousarray(array)
    ).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Exact Experiment-B training normalization.
    tensor = (
        tensor - LR_MEAN
    ) / LR_STD

    return tensor, original_shape


# ================================================================
# OUTPUT POSTPROCESSING
# ================================================================

def postprocess_output(
    prediction: torch.Tensor,
) -> np.ndarray:

    prediction = prediction.detach().float().cpu()

    # Undo GT normalization.
    prediction = (
        prediction * GT_STD
    ) + GT_MEAN

    prediction = torch.nan_to_num(
        prediction,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    # Ground truth is defined in [0,1].
    prediction = torch.clamp(
        prediction,
        0.0,
        1.0,
    )

    output = prediction[0, 0].numpy()

    output = np.asarray(
        output,
        dtype=np.float32,
    )

    return output


# ================================================================
# SINGLE IMAGE INFERENCE
# ================================================================

@torch.inference_mode()
def restore_image(
    model: nn.Module,
    array: np.ndarray,
    device: torch.device,
) -> np.ndarray:

    tensor, input_shape = prepare_input(array)

    tensor = tensor.to(
        device,
        non_blocking=True,
    )

    prediction = model(tensor)

    output = postprocess_output(
        prediction
    )

    expected_shape = (
        input_shape[0] * 2,
        input_shape[1] * 2,
    )

    if output.shape != expected_shape:

        raise RuntimeError(
            f"Incorrect output shape: "
            f"expected {expected_shape}, "
            f"got {output.shape}"
        )

    if not np.isfinite(output).all():
        raise RuntimeError(
            "Output contains NaN or Inf values."
        )

    if output.min() < 0.0 or output.max() > 1.0:
        raise RuntimeError(
            "Output values are outside [0,1]."
        )

    return output


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="REVIXEL NAFNet image restoration inference"
    )

    parser.add_argument(
        "input_dir",
        type=Path,
    )

    parser.add_argument(
        "output_dir",
        type=Path,
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("REVIXEL — NAFNET FINAL INFERENCE")
    print("=" * 70)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device          : {device}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    script_dir = Path(__file__).resolve().parent

    model_path = (
        script_dir
        / "models"
        / MODEL_NAME
    )

    print(f"Model           : {model_path}")

    model = load_model(
        model_path,
        device,
    )

    print("Model loaded successfully.")

    # ------------------------------------------------------------
    # Find inputs
    # ------------------------------------------------------------

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if not input_files:
        raise RuntimeError(
            f"No .npy files found in {input_dir}"
        )

    print(f"Input files     : {len(input_files)}")
    print("-" * 70)

    # ------------------------------------------------------------
    # Process every image
    # ------------------------------------------------------------

    for index, input_path in enumerate(
        input_files,
        start=1,
    ):

        array = np.load(
            input_path,
            allow_pickle=False,
        )

        output = restore_image(
            model,
            array,
            device,
        )

        output_path = (
            output_dir
            / input_path.name
        )

        np.save(
            output_path,
            output,
        )

        print(
            f"[{index:04d}/{len(input_files):04d}] "
            f"{input_path.name} "
            f"{array.shape} -> {output.shape} "
            f"range=[{output.min():.6f}, {output.max():.6f}]"
        )

    print("-" * 70)

    print(
        f"Completed: {len(input_files)} / "
        f"{len(input_files)}"
    )

    print("All outputs saved successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

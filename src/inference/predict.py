"""
predict.py — Inference pipeline for the retina vasculature segmentation UNet.

Supports:
  • Single image prediction
  • Batch prediction from a directory
  • Saving binary mask outputs to disk
"""

import os
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import torch

from src.config.config import (
    IMG_SIZE,
    PRED_THRESHOLD,
    CHECKPOINT_DIR,
    BEST_CKPT_NAME,
    PLOTS_DIR,
)
from src.data.dataset import preprocess_image
from src.models.model import build_model
from src.utils.helpers import get_device, load_model


# ── Core prediction helpers ───────────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    threshold: float = PRED_THRESHOLD,
) -> tuple:
    """
    Predict the vessel mask for a single fundus image.

    Args:
        model     : trained UNet in eval mode
        img_path  : path to a .png fundus image
        device    : torch.device
        threshold : binarisation threshold

    Returns:
        (prob_map, binary_mask) both as (H, W) numpy float32 arrays
    """
    model.eval()

    # Preprocess — returns (1, H, W) channel-first
    img_np = preprocess_image(img_path)                    # (1, H, W)
    img_t  = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, 1, H, W)

    prob_map    = model(img_t).squeeze().cpu().numpy()     # (H, W)
    binary_mask = (prob_map > threshold).astype(np.float32)

    return prob_map, binary_mask


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = PRED_THRESHOLD,
) -> tuple:
    """
    Run inference over an entire DataLoader.

    Returns:
        (predictions, ground_truth) as (N, 1, H, W) numpy float32 arrays.
        ground_truth may be all-zeros if the DataLoader has no mask labels.
    """
    model.eval()
    all_preds = []
    all_masks = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).cpu().numpy()
        all_preds.append(preds)
        all_masks.append(masks.numpy())

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_masks, axis=0),
    )


# ── Save utilities ────────────────────────────────────────────────────────────

def save_mask(binary_mask: np.ndarray, save_path: str) -> None:
    """
    Save a (H, W) binary mask (values 0/1 float) as an 8-bit PNG.

    Args:
        binary_mask : (H, W) float32 with values in {0, 1}
        save_path   : destination file path (must end in .png)
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, mask_uint8)


def predict_directory(
    model: torch.nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    threshold: float = PRED_THRESHOLD,
    pattern: str = "*.png",
) -> None:
    """
    Run inference on all images in `input_dir` and save masks to `output_dir`.

    Args:
        model      : trained UNet
        input_dir  : directory containing fundus images
        output_dir : directory where predicted masks will be saved
        device     : torch.device
        threshold  : binarisation threshold
        pattern    : glob pattern for image files
    """
    os.makedirs(output_dir, exist_ok=True)

    img_paths = sorted(glob(os.path.join(input_dir, pattern)))
    if not img_paths:
        print(f"No images matching '{pattern}' found in: {input_dir}")
        return

    print(f"Predicting {len(img_paths)} images → {output_dir}")

    for img_path in img_paths:
        stem      = Path(img_path).stem
        _, mask   = predict_single(model, img_path, device, threshold)
        save_path = os.path.join(output_dir, f"{stem}_pred.png")
        save_mask(mask, save_path)

    print("Done.")


# ── Entry-point helper ────────────────────────────────────────────────────────

def load_trained_model(
    checkpoint_path: str = None,
    device: torch.device = None,
) -> tuple:
    """
    Convenience function: build model, load checkpoint, return (model, device).

    Args:
        checkpoint_path : path to .pth file; defaults to best checkpoint.
        device          : torch.device; auto-selects if None.
    """
    if device is None:
        device = get_device()

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, BEST_CKPT_NAME)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run scripts/run_train.py first to train the model."
        )

    model = build_model(device)
    model = load_model(model, checkpoint_path, device)
    print(f"Model loaded from: {checkpoint_path}")

    return model, device

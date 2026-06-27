"""
helpers.py — General utility functions for the vasculature segmentation project.
"""

import os
import random
import logging
import numpy as np
import torch

from src.config.config import SEED, FINAL_MODEL_DIR, FINAL_MODEL_NAME


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    """Return the best available device (CUDA GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    return device


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_final_model(model: torch.nn.Module) -> str:
    """Save the final model weights to FINAL_MODEL_DIR."""
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    path = os.path.join(FINAL_MODEL_DIR, FINAL_MODEL_NAME)
    torch.save(model.state_dict(), path)
    print(f"Final model saved → {path}")
    return path


def load_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Load model weights from a checkpoint file.

    Supports both:
      • plain state-dict files (saved with torch.save(model.state_dict(), ...))
      • full checkpoint dicts (saved with epoch, optimizer, etc.)

    Args:
        model           : un-initialised UNet instance
        checkpoint_path : path to .pth checkpoint file
        device          : target device
        strict          : passed to load_state_dict

    Returns:
        model with loaded weights, moved to `device`, in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle both plain state-dict and full checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch      = ckpt.get("epoch", "?")
        val_dice   = ckpt.get("val_dice", "?")
        print(f"Loaded checkpoint: epoch={epoch}, val_dice={val_dice}")
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they do not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

#!/usr/bin/env python
"""
run_compare_models.py — Compare Baseline UNet vs MDRG-UNet on the test set.

Usage:
    python scripts/run_compare_models.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from src.utils.helpers import get_device, setup_logging, load_model
from src.data.dataloader import get_dataloaders
from src.models.model import build_model
from src.models.mdrg_unet import build_mdrg_model
from src.training.validate import evaluate_test_set
from src.config.config import CHECKPOINT_DIR


def main():
    setup_logging()
    device = get_device()

    _, _, test_loader = get_dataloaders(num_workers=2)

    # ── Test Baseline UNet ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Testing BASELINE UNet...")
    print("="*55)
    baseline_ckpt = os.path.join(CHECKPOINT_DIR, "best_retina_unet.pth")
    baseline_model = build_model(device)
    baseline_model = load_model(baseline_model, baseline_ckpt, device)
    baseline_results = evaluate_test_set(baseline_model, test_loader, device)

    # ── Test MDRG-UNet ────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Testing MDRG-UNet...")
    print("="*55)
    mdrg_ckpt = os.path.join(CHECKPOINT_DIR, "best_mdrg_unet.pth")
    mdrg_model = build_mdrg_model(device)
    mdrg_model = load_model(mdrg_model, mdrg_ckpt, device)
    mdrg_results = evaluate_test_set(mdrg_model, test_loader, device)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("         BASELINE UNet  vs  MDRG-UNet — COMPARISON")
    print("="*60)
    print(f"  {'Metric':<22} {'Baseline':>10}  {'MDRG':>10}  {'Improvement':>12}")
    print("-"*60)

    metrics = [
        ("Dice Coefficient",  "test_dice"),
        ("clDice",            "test_cldice"),
        ("Hausdorff (px)",    "hausdorff"),
        ("Skeleton Recall",   "skel_recall"),
    ]

    for name, key in metrics:
        base_val = baseline_results[key]
        mdrg_val = mdrg_results[key]

        # For Hausdorff lower is better
        if key == "hausdorff":
            diff = base_val - mdrg_val
            symbol = "↓ better" if diff > 0 else "↑ worse"
        else:
            diff = mdrg_val - base_val
            symbol = "↑ better" if diff > 0 else "↓ worse"

        print(f"  {name:<22} {base_val:>10.4f}  {mdrg_val:>10.4f}  "
              f"{diff:>+8.4f} {symbol}")

    print("="*60)


if __name__ == "__main__":
    main()
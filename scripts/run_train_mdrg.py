#!/usr/bin/env python
"""
run_train_mdrg.py — Train the MDRG-UNet for retina vasculature segmentation.

The MDRG module is inserted after each encoder ConvBlock, adding
deformable convolutions + ASPP + channel attention to enhance
vessel topology preservation (targets clDice > 0.95).

Usage:
    python scripts/run_train_mdrg.py
    python scripts/run_train_mdrg.py --epochs 50 --batch-size 4 --lr 1e-4
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    CHECKPOINT_DIR,
    LOGS_DIR,
)
from src.utils.helpers import (
    set_seed, get_device, setup_logging,
    count_parameters, load_model
)
from src.data.dataloader import get_dataloaders
from src.models.mdrg_unet import build_mdrg_model
from src.training.train import train
from src.training.validate import evaluate_test_set
from src.utils.visualization import plot_training_curves, visualize_predictions
import torch
import os

# MDRG-specific checkpoint name (separate from baseline UNet)
MDRG_CKPT_NAME   = "best_mdrg_unet.pth"
MDRG_FINAL_NAME  = "mdrg_unet_final.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MDRG-UNet")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--workers",    type=int,   default=2)
    parser.add_argument("--no-eval",    action="store_true")
    return parser.parse_args()


def main():
    setup_logging()
    args   = parse_args()
    device = get_device()
    set_seed()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_mdrg_model(device)
    total = count_parameters(model)
    print(f"MDRG-UNet parameters: {total:,}")
    print(f"  (Baseline UNet was 7,784,577 — MDRG adds deformable conv + ASPP + attention)")

    # ── Override checkpoint name so it doesn't overwrite baseline ────────────
    import src.config.config as cfg
    original_ckpt = cfg.BEST_CKPT_NAME
    cfg.BEST_CKPT_NAME = MDRG_CKPT_NAME

    # ── Training ──────────────────────────────────────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        ckpt_name="best_mdrg_unet.pth",     # MDRG UNet checkpoint
    )

    # Restore original checkpoint name
    cfg.BEST_CKPT_NAME = original_ckpt

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(history, save=True)

    # ── Save final model ──────────────────────────────────────────────────────
    os.makedirs(os.path.join(CHECKPOINT_DIR, "..","final_model"), exist_ok=True)
    final_path = os.path.join(CHECKPOINT_DIR, "..", "final_model", MDRG_FINAL_NAME)
    torch.save(model.state_dict(), final_path)
    print(f"Final MDRG model saved → {final_path}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    if not args.no_eval:
        best_ckpt = os.path.join(CHECKPOINT_DIR, MDRG_CKPT_NAME)
        model = load_model(model, best_ckpt, device)

        results = evaluate_test_set(model, test_loader, device)

        visualize_predictions(
            images=results["masks"],
            image_paths=results["image_paths"],
            masks=results["masks"],
            preds=results["predictions"],
            indices=list(range(min(5, len(results["predictions"])))),
            save=True,
            filename="mdrg_test_predictions.png",
        )

        print("\n" + "=" * 55)
        print("  MDRG-UNet vs Baseline UNet Comparison")
        print("=" * 55)
        print(f"  Baseline clDice : ~0.9181  (from earlier run)")
        print(f"  MDRG     clDice : {results['test_cldice']:.4f}")
        print(f"  Baseline Dice   : ~0.8739")
        print(f"  MDRG     Dice   : {results['test_dice']:.4f}")
        print("=" * 55)


if __name__ == "__main__":
    main()
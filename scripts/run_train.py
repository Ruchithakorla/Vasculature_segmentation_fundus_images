#!/usr/bin/env python
"""
run_train.py — Entry point to train the retina vasculature UNet.

Usage:
    python scripts/run_train.py
    python scripts/run_train.py --epochs 30 --batch-size 8 --lr 5e-5
"""

import sys
import os
import argparse
import torch

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    FINAL_MODEL_DIR,
    FINAL_MODEL_NAME,
    CHECKPOINT_DIR,
)
from src.utils.helpers import set_seed, get_device, setup_logging, save_final_model, count_parameters
from src.data.dataloader import get_dataloaders
from src.models.model import build_model
from src.training.train import train
from src.training.validate import evaluate_test_set
from src.utils.visualization import plot_training_curves, visualize_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Train retina vasculature UNet")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS,    help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--workers",    type=int,   default=2,             help="DataLoader worker processes")
    parser.add_argument("--no-eval",    action="store_true",               help="Skip test set evaluation")
    return parser.parse_args()


def main():
    setup_logging()
    args   = parse_args()
    device = get_device()

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed()

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # ── Training ──────────────────────────────────────────────────────────────
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        ckpt_name="best_retina_unet.pth",   # baseline UNet checkpoint
    )

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(history, save=True)

    # ── Save final model ──────────────────────────────────────────────────────
    save_final_model(model)

    # ── Test set evaluation ───────────────────────────────────────────────────
    if not args.no_eval:
        # Reload best checkpoint for evaluation
        from src.utils.helpers import load_model
        best_ckpt = os.path.join(CHECKPOINT_DIR, "best_retina_unet.pth")
        model = load_model(model, best_ckpt, device)

        results = evaluate_test_set(model, test_loader, device)

        visualize_predictions(
            images=results["masks"],
            image_paths=results["image_paths"],
            masks=results["masks"],
            preds=results["predictions"],
            indices=list(range(min(5, len(results["predictions"])))),
            save=True,
            filename="test_predictions.png",
        )


if __name__ == "__main__":
    main()

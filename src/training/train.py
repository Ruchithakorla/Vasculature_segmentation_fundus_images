"""
train.py — Training loop for the retina vasculature segmentation UNet.

Features:
  • Hybrid loss (clDice + Dice + BCE)
  • ReduceLROnPlateau scheduler
  • Best-checkpoint saving
  • CSV training log
  • Early stopping
"""

import os
import csv
import time
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    LR_PATIENCE,
    LR_FACTOR,
    LR_MIN,
    EARLY_STOP_PATIENCE,
    CHECKPOINT_DIR,
    LOGS_DIR,
    BEST_CKPT_NAME,
)
from src.utils.metrics import hybrid_loss, dice_coef, cldice
from src.training.validate import validate_one_epoch

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """
    Run one full training epoch.

    Returns:
        dict with keys: loss, dice, cldice  (averaged over all batches)
    """
    model.train()
    total_loss  = 0.0
    total_dice  = 0.0
    total_cldice = 0.0
    n_batches   = len(loader)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        preds = model(images)

        loss = hybrid_loss(masks, preds)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss   += loss.item()
            total_dice   += dice_coef(masks, preds).item()
            total_cldice += cldice(masks, preds).item()

    return {
        "loss":   total_loss   / n_batches,
        "dice":   total_dice   / n_batches,
        "cldice": total_cldice / n_batches,
    }


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    ckpt_name: str = None,
) -> dict:
    """
    Full training procedure with early stopping and LR scheduling.

    Args:
        model         : UNet model (already on `device`)
        train_loader  : training DataLoader
        val_loader    : validation DataLoader
        device        : torch.device
        num_epochs    : maximum number of epochs
        learning_rate : initial learning rate

    Returns:
        history dict with lists of per-epoch metrics.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    save_name      = ckpt_name if ckpt_name else BEST_CKPT_NAME
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, save_name)
    log_csv_path   = os.path.join(LOGS_DIR, "training_log.csv")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN
    )

    history = {
        "train_loss": [], "train_dice": [], "train_cldice": [],
        "val_loss":   [], "val_dice":   [], "val_cldice":   [],
    }
    

    best_val_cldice  = -1.0
    early_stop_count = 0

    # ── CSV header ───────────────────────────────────────────────────────────
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_cldice",
            "val_loss",   "val_dice",   "val_cldice",
            "lr",
        ])

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics   = validate_one_epoch(model, val_loader, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_cldice"].append(train_metrics["cldice"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_cldice"].append(val_metrics["cldice"])

        # LR scheduler steps on val loss
        scheduler.step(val_metrics["loss"])

        # Best checkpoint
        if val_metrics["cldice"] > best_val_cldice:
            best_val_cldice = val_metrics["cldice"]
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_cldice":  best_val_cldice,
                    "val_dice":    val_metrics["dice"],
                },
                best_ckpt_path,
            )
            early_stop_count = 0
            improved_marker  = " ← best cl dice"
        else:
            early_stop_count += 1
            improved_marker  = ""

        # Log to console
        print(
            f"Epoch [{epoch:3d}/{num_epochs}] "
            f"| train loss={train_metrics['loss']:.4f}  dice={train_metrics['dice']:.4f}  cldice={train_metrics['cldice']:.4f}"
            f"| val   loss={val_metrics['loss']:.4f}  dice={val_metrics['dice']:.4f}  cldice={val_metrics['cldice']:.4f}"
            f"| best_cldice={best_val_cldice:.4f}"
            f"| lr={current_lr:.2e}  time={elapsed:.1f}s{improved_marker}"
        )

        # Append CSV row
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_metrics["loss"],   6),
                round(train_metrics["dice"],   6),
                round(train_metrics["cldice"], 6),
                round(val_metrics["loss"],     6),
                round(val_metrics["dice"],     6),
                round(val_metrics["cldice"],   6),
                current_lr,
            ])

        # Early stopping
        if early_stop_count >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(no val_dice improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"\nTraining complete. Best val clDice: {best_val_cldice:.4f}")
    print(f"Best checkpoint saved at: {best_ckpt_path}")

    return history

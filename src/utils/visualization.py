"""
visualization.py — Plotting helpers for training curves and prediction overlays.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server environments
import matplotlib.pyplot as plt

from src.config.config import PLOTS_DIR


def plot_training_curves(history: dict, save: bool = True) -> None:
    """
    Plot train/val loss, Dice, and clDice curves.

    Args:
        history : dict returned by src.training.train.train()
        save    : if True, save to PLOTS_DIR/training_curves.png
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # ── Dice ─────────────────────────────────────────────────────────────────
    axes[0].plot(history["train_dice"], label="Train Dice")
    axes[0].plot(history["val_dice"],   label="Val Dice")
    axes[0].set_title("Dice Coefficient")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(True)

    # ── clDice ───────────────────────────────────────────────────────────────
    axes[1].plot(history["train_cldice"], label="Train clDice")
    axes[1].plot(history["val_cldice"],   label="Val clDice")
    axes[1].set_title("clDice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True)

    # ── Loss ─────────────────────────────────────────────────────────────────
    axes[2].plot(history["train_loss"], label="Train Loss")
    axes[2].plot(history["val_loss"],   label="Val Loss")
    axes[2].set_title("Hybrid Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save:
        save_path = os.path.join(PLOTS_DIR, "training_curves.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")

    plt.close(fig)
def visualize_predictions(
    images: np.ndarray,
    masks: np.ndarray,
    preds: np.ndarray,
    image_paths: list = None,
    indices: list = None,
    threshold: float = 0.3,
    save: bool = True,
    filename: str = "predictions.png",
) -> None:
    """
    Side-by-side visualisation:
    Original RGB fundus image | Ground truth mask | Predicted mask

    Args:
        images      : (N, 1, H, W) fallback images if no paths given
        masks       : (N, 1, H, W) ground-truth masks
        preds       : (N, 1, H, W) predicted probabilities
        image_paths : list of original image file paths for real RGB display
        indices     : list of sample indices to display (default: first 5)
        threshold   : binarisation threshold for predicted masks
        save        : save to PLOTS_DIR/<filename>
        filename    : output filename
    """
    if indices is None:
        indices = list(range(min(5, len(images))))

    os.makedirs(PLOTS_DIR, exist_ok=True)

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):

        # ── Column 1: Original RGB fundus image ──────────────────────────────
        if image_paths is not None and idx < len(image_paths):
            # Load real original color fundus image from disk
            raw = cv2.imread(image_paths[idx])
            raw = cv2.resize(raw, (images.shape[3], images.shape[2]))
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)  # BGR → RGB
            axes[row, 0].imshow(raw)
            axes[row, 0].set_title(f"Original Fundus Image [{idx}]")
        else:
            # Fallback: show CLAHE green channel if no paths given
            axes[row, 0].imshow(images[idx, 0], cmap="gray")
            axes[row, 0].set_title(f"Input Image [{idx}]")
        axes[row, 0].axis("off")

        # ── Column 2: Ground truth mask ───────────────────────────────────────
        axes[row, 1].imshow(masks[idx, 0], cmap="gray")
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 1].axis("off")

        # ── Column 3: Predicted mask ──────────────────────────────────────────
        pred_mask = (preds[idx, 0] > threshold).astype(np.float32)
        axes[row, 2].imshow(pred_mask, cmap="gray")
        axes[row, 2].set_title(f"Predicted Mask (thr={threshold})")
        axes[row, 2].axis("off")

    plt.tight_layout()

    if save:
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Prediction visualisation saved → {save_path}")

    plt.close(fig)



    


def plot_per_image_metrics(
    metric_dict: dict,
    save: bool = True,
    filename: str = "per_image_metrics.png",
) -> None:
    """
    Plot per-image metric curves (Dice, clDice, Skeleton Recall, etc.).

    Args:
        metric_dict : {metric_name: np.ndarray} of per-image scores
        save        : save to PLOTS_DIR/<filename>
        filename    : output filename
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    n = len(metric_dict)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    colors = [
        "cornflowerblue", "mediumseagreen", "darkorange",
        "orchid", "tomato", "goldenrod",
    ]

    for i, (name, values) in enumerate(metric_dict.items()):
        ax = axes[i]
        ax.plot(values, color=colors[i % len(colors)], linewidth=0.8, alpha=0.7)
        ax.axhline(
            float(np.mean(values)), color="black", linestyle="--",
            linewidth=1.2, label=f"mean={np.mean(values):.3f}",
        )
        ax.set_title(name)
        ax.set_xlabel("Test Image Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save:
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Per-image metrics saved → {save_path}")

    plt.close(fig)

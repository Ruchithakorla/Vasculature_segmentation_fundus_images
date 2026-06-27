#!/usr/bin/env python
"""
run_full_evaluation.py — Compute and visualise the complete metric suite
(Dice, clDice, soft-clDice, smooth-clDice, Betti numbers) on train and
validation/test sets, and plot everything.

Usage:
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --betti-sample 20
    python scripts/run_full_evaluation.py --checkpoint models/checkpoints/best_retina_unet.pth
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.config import CHECKPOINT_DIR, BEST_CKPT_NAME, PLOTS_DIR, BETTI_SAMPLE
from src.utils.helpers import setup_logging, get_device
from src.inference.predict import load_trained_model
from src.data.dataloader import get_dataloaders
from src.utils.metrics import (
    dice_coef,
    cldice,
    soft_cldice_metric,
    smooth_cldice_metric,
    compute_gt_skeleton_batch,
    skeleton_recall,
    betti_error_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Full metric-suite evaluation")
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(CHECKPOINT_DIR, BEST_CKPT_NAME),
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--betti-sample", type=int, default=BETTI_SAMPLE,
        help="Number of images to use for Betti number computation (slow metric)",
    )
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference over a loader, return (images, masks, preds) as numpy arrays."""
    all_images, all_masks, all_preds = [], [], []
    for images, masks in loader:
        images_dev = images.to(device, non_blocking=True)
        preds = model(images_dev)
        all_images.append(images.numpy())
        all_masks.append(masks.numpy())
        all_preds.append(preds.cpu().numpy())
    return (
        np.concatenate(all_images, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_preds, axis=0),
    )


def compute_all_metrics(masks_np, preds_np, device, betti_sample, tag=""):
    """
    Compute Dice, clDice, soft-clDice, smooth-clDice (per-image) plus
    Skeleton Recall and Betti numbers (on a sample) for one split.
    """
    n = len(masks_np)
    masks_t = torch.from_numpy(masks_np).to(device)
    preds_t = torch.from_numpy(preds_np).to(device)

    print(f"[{tag}] Computing per-image Dice / clDice / soft-clDice / smooth-clDice "
          f"for {n} images...")

    dice_scores, cldice_scores = [], []
    soft_cl_scores, smooth_cl_scores = [], []

    for i in range(n):
        m = masks_t[i:i + 1]
        p = preds_t[i:i + 1]
        dice_scores.append(dice_coef(m, p).item())
        cldice_scores.append(cldice(m, p).item())
        soft_cl_scores.append(soft_cldice_metric(m, p).item())
        smooth_cl_scores.append(smooth_cldice_metric(m, p).item())

    # ── Skeleton recall (numpy, whole split) ─────────────────────────────────
    print(f"[{tag}] Computing GT skeletons + skeleton recall...")
    gt_skels = compute_gt_skeleton_batch(masks_np)
    mean_srl, per_img_srl = skeleton_recall(gt_skels, preds_np)

    # ── Betti numbers (slow — only on a sample) ──────────────────────────────
    sample_n = min(betti_sample, n)
    print(f"[{tag}] Computing Betti numbers on a sample of {sample_n} images "
          f"(this is the slow step)...")
    try:
        betti_results = betti_error_batch(
            masks_np[:sample_n], preds_np[:sample_n],
            threshold=0.5, min_persistence=0.0,
        )
    except ImportError:
        print(f"[{tag}] gudhi not installed — skipping Betti numbers. "
              f"Install with: pip install gudhi")
        betti_results = None

    return {
        "dice":        np.array(dice_scores),
        "cldice":      np.array(cldice_scores),
        "soft_cldice": np.array(soft_cl_scores),
        "smooth_cldice": np.array(smooth_cl_scores),
        "skel_recall": per_img_srl,
        "mean_skel_recall": mean_srl,
        "betti": betti_results,
    }


def print_summary(name, results):
    print("\n" + "=" * 60)
    print(f"  {name.upper()} SET — METRIC SUMMARY")
    print("=" * 60)
    print(f"  Dice Coefficient   : {results['dice'].mean():.4f}  ± {results['dice'].std():.4f}")
    print(f"  clDice             : {results['cldice'].mean():.4f}  ± {results['cldice'].std():.4f}")
    print(f"  soft-clDice        : {results['soft_cldice'].mean():.4f}  ± {results['soft_cldice'].std():.4f}")
    print(f"  smooth-clDice      : {results['smooth_cldice'].mean():.4f}  ± {results['smooth_cldice'].std():.4f}")
    print(f"  Skeleton Recall    : {results['mean_skel_recall']:.4f}  ± {results['skel_recall'].std():.4f}")
    if results["betti"] is not None:
        b = results["betti"]
        print(f"  |Δβ₀| (components) : {b['mean_delta_beta0']:.2f}")
        print(f"  |Δβ₁| (loops)      : {b['mean_delta_beta1']:.2f}")
    print("=" * 60)


def plot_train_val_comparison(train_results, val_results, save_path):
    """Bar chart comparing mean metric values between train and val/test sets."""
    metrics = ["dice", "cldice", "soft_cldice", "smooth_cldice"]
    labels  = ["Dice", "clDice", "Soft-clDice", "Smooth-clDice"]

    train_means = [train_results[m].mean() for m in metrics]
    val_means   = [val_results[m].mean()   for m in metrics]

    train_means.append(train_results["mean_skel_recall"])
    val_means.append(val_results["mean_skel_recall"])
    labels.append("Skeleton Recall")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_means, width, label="Train", color="cornflowerblue")
    bars2 = ax.bar(x + width / 2, val_means,   width, label="Validation/Test", color="darkorange")

    ax.set_ylabel("Score")
    ax.set_title("Train vs Validation/Test — Metric Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved train vs val/test comparison → {save_path}")


def plot_per_image_grid(results, split_name, save_path):
    """Per-image line plots for all 4 clDice-family metrics + skeleton recall + Betti."""
    metric_data = [
        (results["dice"],          "Dice Coefficient",   "cornflowerblue"),
        (results["cldice"],        "clDice",             "mediumseagreen"),
        (results["soft_cldice"],   "Soft-clDice",         "darkorange"),
        (results["smooth_cldice"], "Smooth-clDice",       "orchid"),
        (results["skel_recall"],   "Skeleton Recall (SRL)", "teal"),
    ]

    if results["betti"] is not None:
        metric_data.append((results["betti"]["per_img_db0"], "|Δβ₀| (components)", "tomato"))
        metric_data.append((results["betti"]["per_img_db1"], "|Δβ₁| (loops)",       "goldenrod"))

    n = len(metric_data)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle(f"Per-Image Metrics — {split_name}", fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    for i, (values, title, color) in enumerate(metric_data):
        ax = axes[i]
        ax.plot(values, color=color, linewidth=0.8, alpha=0.7)
        ax.axhline(float(np.mean(values)), color="black", linestyle="--",
                   linewidth=1.2, label=f"mean={np.mean(values):.3f}")
        ax.set_title(title)
        ax.set_xlabel("Image Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-image metric grid ({split_name}) → {save_path}")


def main():
    setup_logging()
    args = parse_args()
    device = get_device()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    model, device = load_trained_model(checkpoint_path=args.checkpoint, device=device)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(num_workers=args.workers)

    # ── Collect predictions ─────────────────────────────────────────────────
    print("\nRunning inference on TRAIN set...")
    _, train_masks, train_preds = collect_predictions(model, train_loader, device)

    print("\nRunning inference on TEST set...")
    _, test_masks, test_preds = collect_predictions(model, test_loader, device)

    # ── Compute full metric suite ───────────────────────────────────────────
    train_results = compute_all_metrics(
        train_masks, train_preds, device, args.betti_sample, tag="TRAIN"
    )
    test_results = compute_all_metrics(
        test_masks, test_preds, device, args.betti_sample, tag="TEST"
    )

    # ── Print summaries ──────────────────────────────────────────────────────
    print_summary("train", train_results)
    print_summary("test", test_results)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_train_val_comparison(
        train_results, test_results,
        os.path.join(PLOTS_DIR, "train_vs_test_metrics.png"),
    )
    plot_per_image_grid(
        train_results, "Train Set",
        os.path.join(PLOTS_DIR, "train_per_image_metrics.png"),
    )
    plot_per_image_grid(
        test_results, "Test Set",
        os.path.join(PLOTS_DIR, "test_per_image_metrics.png"),
    )

    print("\nAll evaluation plots saved to:", PLOTS_DIR)
    print("  - train_vs_test_metrics.png      (bar chart comparison)")
    print("  - train_per_image_metrics.png    (per-image grid, train set)")
    print("  - test_per_image_metrics.png     (per-image grid, test set)")


if __name__ == "__main__":
    main()

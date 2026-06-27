#!/usr/bin/env python
"""
run_crosschecking_dataset.py — Evaluate the FIVES-trained UNet on all 8 retinal
vessel datasets in the retinal-vessel-fundus-dataset-collection.

Datasets supported:
    CHASEDB1, DRIVE, FIVES, HRF, LES-AV, RETA, STARE, TRENDS

Usage:
    # Evaluate on ALL datasets
    python scripts/ run_crosschecking_dataset.py

    # Evaluate on specific datasets only
    python scripts/run_crosschecking_dataset.py --datasets DRIVE STARE CHASEDB1

    # Custom collection path
    python scripts/run_crosshecking_dataset.py --collection /path/to/retinal-vessel-fundus-dataset-collection
"""

import sys
import os
import argparse
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.config import (
    CHECKPOINT_DIR,
    BEST_CKPT_NAME,
    PLOTS_DIR,
    IMG_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    PRED_THRESHOLD,
)
from src.utils.helpers import setup_logging, get_device
from src.inference.predict import load_trained_model
from src.utils.metrics import (
    dice_coef,
    cldice,
    hausdorff_distance,
    compute_gt_skeleton_batch,
    skeleton_recall,
)


# ── Dataset configs — images/masks folder + supported extensions ─────────────

DATASET_CONFIGS = {
    "CHASEDB1": {
        "images_dir": "images",
        "masks_dir":  "masks_1",          # use first annotator masks
        "img_exts":   [".jpg", ".png"],
        "mask_exts":  [".png", ".jpg"],
    },
    "DRIVE": {
        "images_dir": "images",
        "masks_dir":  "masks",
        "img_exts":   [".png", ".jpg", ".tif"],
        "mask_exts":  [".png", ".gif", ".tif"],
    },
    "FIVES": {
        "images_dir": "images",
        "masks_dir":  "masks",
        "img_exts":   [".png", ".jpg"],
        "mask_exts":  [".png"],
    },
    "HRF": {
        "images_dir": "images",
        "masks_dir":  "masks",
        "img_exts":   [".jpg", ".png", ".tif"],
        "mask_exts":  [".png", ".tif"],
    },
    "LES-AV": {
        "images_dir": "images",
        "masks_dir":  "masks",            # combined vessel mask
        "img_exts":   [".png", ".jpg"],
        "mask_exts":  [".png"],
    },
    "RETA": {
        "images_dir": "images",
        "masks_dir":  "masks",
        "img_exts":   [".jpg", ".png"],
        "mask_exts":  [".png", ".jpg"],
    },
    "STARE": {
        "images_dir": "images",
        "masks_dir":  "masks_1",          # use first annotator masks
        "img_exts":   [".png", ".jpg", ".ppm"],
        "mask_exts":  [".png", ".pgm", ".ppm"],
    },
    "TRENDS": {
        "images_dir": "images",
        "masks_dir":  "masks",
        "img_exts":   [".png", ".jpg"],
        "mask_exts":  [".png"],
    },
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())


# ── File helpers ─────────────────────────────────────────────────────────────

def collect_files(directory: str, extensions: list) -> list:
    """Return sorted list of files matching any of the given extensions."""
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(set(files))

def match_image_mask_pairs(img_files: list, mask_files: list) -> list:
    """
    Match images to masks using 3 strategies in order:

    Strategy 1 — Exact stem match
        image: 01.png        mask: 01.png

    Strategy 2 — Image stem is a prefix of mask stem
        image: Image_01L.jpg     mask: Image_01L_1stHO.png
        image: im0001.png        mask: im0001_1stHO.png
        image: IDRiD_01.jpg      mask: IDRiD_01_vessel.png
        image: 10_L.png          mask: 10_L_SEG.png

    Strategy 3 — Match by leading number prefix
        image: 21_training.png   mask: 21_manual1.png

    Returns list of (img_path, mask_path) tuples.
    """
    import re

    # ── Strategy 1: exact stem match ─────────────────────────────────────────
    mask_stems = {Path(m).stem: m for m in mask_files}
    pairs = []
    for img_path in img_files:
        stem = Path(img_path).stem
        if stem in mask_stems:
            pairs.append((img_path, mask_stems[stem]))
    if pairs:
        return pairs

    # ── Strategy 2: image stem is a prefix of mask stem ──────────────────────
    pairs = []
    for img_path in img_files:
        img_stem = Path(img_path).stem
        candidates = [
            m for m in mask_files
            if Path(m).stem.startswith(img_stem + "_") or Path(m).stem == img_stem
        ]
        if len(candidates) == 1:
            pairs.append((img_path, candidates[0]))
        elif len(candidates) > 1:
            best = min(candidates, key=lambda m: len(Path(m).stem))
            pairs.append((img_path, best))
    if pairs:
        return pairs

    # ── Strategy 3: match by leading number ──────────────────────────────────
    def leading_number(path):
        nums = re.findall(r'^\d+', Path(path).stem)
        return nums[0] if nums else None

    mask_by_num = {}
    for m in mask_files:
        num = leading_number(m)
        if num and num not in mask_by_num:
            mask_by_num[num] = m

    pairs = []
    for img_path in img_files:
        num = leading_number(img_path)
        if num and num in mask_by_num:
            pairs.append((img_path, mask_by_num[num]))
    return pairs




# ── PyTorch Dataset for cross-dataset inference ──────────────────────────────

class CrossDataset(Dataset):
    """
    Generic dataset that loads image-mask pairs from any folder structure.
    Applies the same CLAHE + green-channel preprocessing as the FIVES training pipeline.
    """

    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # ── Raw image for visualization (real color fundus) ─────────────────
        raw = cv2.imread(img_path)
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        raw = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw_t = torch.from_numpy(raw_rgb)                       # (H, W, 3) uint8

        # ── Image: green channel + CLAHE (same as training) ─────────────────
        green = raw[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
        enhanced = clahe.apply(green).astype(np.float32) / 255.0
        image_t = torch.from_numpy(enhanced[np.newaxis, ...])   # (1, H, W)

        # ── Mask: grayscale, resize, normalise ───────────────────────────────
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Try PIL for formats OpenCV can't handle (e.g. .gif, .ppm)
            from PIL import Image
            mask = np.array(Image.open(mask_path).convert("L"))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        mask_t = torch.from_numpy(mask[np.newaxis, ...])        # (1, H, W)

        return image_t, mask_t, raw_t, img_path


# ── Per-dataset evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(
    dataset_name: str,
    collection_root: str,
    model: torch.nn.Module,
    device: torch.device,
) -> dict | None:
    """
    Load images/masks for one dataset, run model, compute all metrics.
    Returns a results dict or None if the dataset folder is missing.
    """
    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        print(f"  [SKIP] No config for dataset: {dataset_name}")
        return None

    dataset_root = os.path.join(collection_root, dataset_name)
    images_dir   = os.path.join(dataset_root, cfg["images_dir"])
    masks_dir    = os.path.join(dataset_root, cfg["masks_dir"])

    if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
        print(f"  [SKIP] {dataset_name}: folder not found at {dataset_root}")
        return None

    img_files  = collect_files(images_dir,  cfg["img_exts"])
    mask_files = collect_files(masks_dir,   cfg["mask_exts"])

    if not img_files:
        print(f"  [SKIP] {dataset_name}: no images found in {images_dir}")
        return None

    pairs = match_image_mask_pairs(img_files, mask_files)
    if not pairs:
        print(f"  [SKIP] {dataset_name}: could not match any image-mask pairs")
        return None

    print(f"  [{dataset_name}] Found {len(pairs)} image-mask pairs → running inference...")

    loader = DataLoader(
        CrossDataset(pairs),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    # ── Collect predictions ──────────────────────────────────────────────────
    all_masks, all_preds, all_raws = [], [], []
    model.eval()

    for images, masks, raws, _ in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images)
        all_masks.append(masks.numpy())
        all_preds.append(preds.cpu().numpy())
        all_raws.append(raws.numpy())

    all_masks = np.concatenate(all_masks, axis=0)   # (N, 1, H, W)
    all_preds = np.concatenate(all_preds, axis=0)   # (N, 1, H, W)
    all_raws  = np.concatenate(all_raws, axis=0)    # (N, H, W, 3) uint8



    # ── Compute metrics ──────────────────────────────────────────────────────
    masks_t = torch.from_numpy(all_masks).to(device)
    preds_t = torch.from_numpy(all_preds).to(device)

    dice   = dice_coef(masks_t, preds_t).item()
    cl     = cldice(masks_t, preds_t).item()

    # Per-image Hausdorff (averaged)
    hd_scores = [
        hausdorff_distance(all_masks[i], all_preds[i])
        for i in range(len(all_preds))
    ]
    mean_hd = float(np.nanmean(hd_scores))

    # Skeleton recall
    gt_skels = compute_gt_skeleton_batch(all_masks)
    mean_srl, per_srl = skeleton_recall(gt_skels, all_preds)

    # Per-image Dice and clDice
    per_dice   = [dice_coef(masks_t[i:i+1], preds_t[i:i+1]).item() for i in range(len(all_preds))]
    per_cldice = [cldice(masks_t[i:i+1],    preds_t[i:i+1]).item() for i in range(len(all_preds))]

    print(f"  [{dataset_name}] Dice={dice:.4f}  clDice={cl:.4f}  "
          f"Hausdorff={mean_hd:.1f}px  SRL={mean_srl:.4f}")

    return {
        "dataset":    dataset_name,
        "n_images":   len(pairs),
        "dice":       dice,
        "cldice":     cl,
        "hausdorff":  mean_hd,
        "skel_recall": mean_srl,
        "per_dice":   np.array(per_dice),
        "per_cldice": np.array(per_cldice),
        "per_srl":    per_srl,
        "per_hd":     np.array(hd_scores),
        "masks":      all_masks,
        "preds":      all_preds,
        "raw_images": all_raws,
    }

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison_table(results: list, save_path: str) -> None:
    """Bar chart comparing all metrics across all datasets."""
    names      = [r["dataset"]   for r in results]
    dice_vals  = [r["dice"]      for r in results]
    cl_vals    = [r["cldice"]    for r in results]
    srl_vals   = [r["skel_recall"] for r in results]
    hd_vals    = [r["hausdorff"] for r in results]

    x     = np.arange(len(names))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cross-Dataset Evaluation — FIVES-Trained UNet", fontsize=14, fontweight="bold")

    # ── Left: Dice, clDice, SRL (all 0-1) ────────────────────────────────────
    ax = axes[0]
    ax.bar(x - width, dice_vals, width, label="Dice",           color="cornflowerblue")
    ax.bar(x,         cl_vals,   width, label="clDice",         color="mediumseagreen")
    ax.bar(x + width, srl_vals,  width, label="Skeleton Recall", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Dice / clDice / Skeleton Recall")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    # ── Right: Hausdorff Distance ─────────────────────────────────────────────
    ax2 = axes[1]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax2.bar(names, hd_vals, color=colors)
    ax2.set_xticklabels(names, rotation=20, ha="right")
    ax2.set_ylabel("Distance (pixels)")
    ax2.set_title("Hausdorff Distance (lower = better)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison chart saved → {save_path}")


def plot_per_image_metrics(results: list, save_path: str) -> None:
    """Per-image Dice curves for all datasets in one figure."""
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle("Per-Image Dice — Cross Dataset", fontsize=14, fontweight="bold")
    axes = np.array(axes).flatten()

    colors = ["cornflowerblue", "mediumseagreen", "darkorange",
              "orchid", "teal", "tomato", "goldenrod", "slategray"]

    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r["per_dice"], color=colors[i % len(colors)], linewidth=0.9, alpha=0.8)
        ax.axhline(r["dice"], color="black", linestyle="--",
                   linewidth=1.2, label=f"mean={r['dice']:.3f}")
        ax.set_title(f"{r['dataset']} (n={r['n_images']})")
        ax.set_xlabel("Image Index")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-image Dice chart saved → {save_path}")


def plot_sample_predictions(results: list, save_path: str) -> None:
    """One sample prediction per dataset — real fundus image | GT | prediction."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Sample Predictions — Cross Dataset", fontsize=14, fontweight="bold")

    for row, r in enumerate(results):
        # Pick the image closest to mean Dice (most representative)
        mean_idx = int(np.argmin(np.abs(r["per_dice"] - r["dice"])))
        fundus = r["raw_images"][mean_idx]       # (H, W, 3) real color fundus image
        gt     = r["masks"][mean_idx, 0]
        pred   = (r["preds"][mean_idx, 0] > PRED_THRESHOLD).astype(np.float32)

        axes[row, 0].imshow(fundus); axes[row, 0].set_title(f"{r['dataset']} — Fundus Image")
        axes[row, 1].imshow(gt,   cmap="gray"); axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].set_title(f"Predicted (Dice={r['per_dice'][mean_idx]:.3f})")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample predictions saved → {save_path}")

def print_summary_table(results: list) -> None:
    """Print a clean summary table to the terminal."""
    print("\n" + "=" * 72)
    print(f"  {'Dataset':<12} {'N':>5}  {'Dice':>7}  {'clDice':>7}  "
          f"{'Hausdorff':>10}  {'SRL':>7}")
    print("=" * 72)
    for r in results:
        print(f"  {r['dataset']:<12} {r['n_images']:>5}  "
              f"{r['dice']:>7.4f}  {r['cldice']:>7.4f}  "
              f"{r['hausdorff']:>10.2f}  {r['skel_recall']:>7.4f}")
    print("=" * 72)

    # Best/Worst cases of Cldice
    best_cldice = max(results, key=lambda x: x["cldice"])
    worst_cldice = min(results, key=lambda x: x["cldice"])
    print(f"\n  Best  clDice : {best_cldice['dataset']}  ({best_cldice['cldice']:.4f})")
    print(f"  Worst clDice : {worst_cldice['dataset']}  ({worst_cldice['cldice']:.4f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset retinal vessel evaluation")
    parser.add_argument(
        "--collection",
        type=str,
        default="/media/mmk/DATA-1/Ruchitha/vasculature-segmentation/retinal-vessel-fundus-dataset-collection",
        help="Path to the retinal-vessel-fundus-dataset-collection folder",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Datasets to evaluate (default: all). Choose from: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(CHECKPOINT_DIR, BEST_CKPT_NAME),
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    setup_logging()
    args   = parse_args()
    device = get_device()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, device = load_trained_model(checkpoint_path=args.checkpoint, device=device)

    # ── Evaluate each dataset ─────────────────────────────────────────────────
    print(f"\nEvaluating on {len(args.datasets)} datasets from:")
    print(f"  {args.collection}\n")

    all_results = []
    for ds_name in args.datasets:
        result = evaluate_dataset(ds_name, args.collection, model, device)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No datasets could be evaluated. Check the collection path and folder structure.")
        return

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_comparison_table(
        all_results,
        os.path.join(PLOTS_DIR, "cross_dataset_comparison.png"),
    )
    plot_per_image_metrics(
        all_results,
        os.path.join(PLOTS_DIR, "cross_dataset_per_image_dice.png"),
    )
    plot_sample_predictions(
        all_results,
        os.path.join(PLOTS_DIR, "cross_dataset_predictions.png"),
    )

    print("\nAll cross-dataset plots saved to:", PLOTS_DIR)
    print("  - cross_dataset_comparison.png      ← bar chart: all metrics vs all datasets")
    print("  - cross_dataset_per_image_dice.png  ← per-image Dice curves per dataset")
    print("  - cross_dataset_predictions.png     ← sample prediction per dataset")


if __name__ == "__main__":
    main()
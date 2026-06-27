#!/usr/bin/env python
"""
run_inference.py — Run inference with a trained retina vasculature UNet.

Usage:
    # Predict all images in a directory and save masks:
    python scripts/run_inference.py --input /path/to/images --output /path/to/masks

    # Evaluate the test set and save visualisations:
    python scripts/run_inference.py --evaluate

    # Use a custom checkpoint:
    python scripts/run_inference.py --checkpoint models/checkpoints/best_retina_unet.pth --input ...
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.config import PRED_THRESHOLD, CHECKPOINT_DIR, BEST_CKPT_NAME
from src.utils.helpers import setup_logging, get_device
from src.inference.predict import load_trained_model,predict_directory
from src.models.mdrg_unet import build_mdrg_model
from src.data.dataloader import get_dataloaders
from src.training.validate import evaluate_test_set
from src.utils.visualization import visualize_predictions, plot_per_image_metrics
from src.utils.metrics import compute_gt_skeleton_batch, skeleton_recall

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for retina vasculature UNet")
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(CHECKPOINT_DIR, BEST_CKPT_NAME),
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--input",     type=str,   default=None,          help="Directory of fundus images to predict")
    parser.add_argument("--output",    type=str,   default="outputs/preds", help="Directory to save predicted masks")
    parser.add_argument("--threshold", type=float, default=PRED_THRESHOLD, help="Binarisation threshold")
    parser.add_argument("--evaluate",  action="store_true",                help="Evaluate on the test set")
    parser.add_argument("--workers",   type=int,   default=2,             help="DataLoader workers")
    return parser.parse_args()


def main():
    setup_logging()
    args   = parse_args()
    device = get_device()
    # Auto-detect which model to load based on checkpoint filename
    if "mdrg" in args.checkpoint:
        from src.utils.helpers import load_model
        model = build_mdrg_model(device)
        model = load_model(model, args.checkpoint, device)
        print(f"Loaded MDRG-UNet from: {args.checkpoint}")
    else:
        model, device = load_trained_model(
            checkpoint_path=args.checkpoint,
            device=device,
        )
        print(f"Loaded Baseline UNet from: {args.checkpoint}")
    

    # ── Directory-level batch prediction ─────────────────────────────────────
    if args.input:
        predict_directory(
            model=model,
            input_dir=args.input,
            output_dir=args.output,
            device=device,
            threshold=args.threshold,
        )

    # ── Full test-set evaluation ──────────────────────────────────────────────
    if args.evaluate:
        _, _, test_loader = get_dataloaders(num_workers=args.workers)
        results = evaluate_test_set(model, test_loader, device)

        predictions = results["predictions"]   # (N, 1, H, W)
        masks       = results["masks"]         # (N, 1, H, W)

        # ── Visualise sample predictions ─────────────────────────────────────
        visualize_predictions(
    images=results["images"],      # ✅ real preprocessed images now, not masks
    image_paths=results["image_paths"],
    masks=masks,
    preds=predictions,
    indices=list(range(min(5, len(predictions)))),
    threshold=args.threshold,
    save=True,
    filename="inference_predictions.png",
)

        # ── Skeleton recall per image ─────────────────────────────────────────
        print("Computing skeleton recall per image...")
        gt_skels = compute_gt_skeleton_batch(masks)
        mean_srl, per_img_srl = skeleton_recall(gt_skels, predictions)

        # ── Per-image metric visualisation ────────────────────────────────────
        from src.utils.metrics import dice_coef, cldice
        import torch

        preds_t = torch.from_numpy(predictions).to(device)
        masks_t = torch.from_numpy(masks).to(device)

        per_dice   = [
            dice_coef(masks_t[i:i+1], preds_t[i:i+1]).item()
            for i in range(len(predictions))
        ]
        per_cldice = [
            cldice(masks_t[i:i+1], preds_t[i:i+1]).item()
            for i in range(len(predictions))
        ]

        metric_dict = {
            "Dice Coefficient":    np.array(per_dice),
            "clDice":              np.array(per_cldice),
            "Skeleton Recall (SRL)": per_img_srl,
        }
        plot_per_image_metrics(metric_dict, save=True, filename="per_image_metrics.png")

    if not args.input and not args.evaluate:
        print("Nothing to do. Use --input <dir> and/or --evaluate.")


if __name__ == "__main__":
    main()

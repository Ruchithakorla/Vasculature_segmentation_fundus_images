"""
validate.py — Validation loop for the retina vasculature segmentation UNet.
"""

import torch

from src.utils.metrics import hybrid_loss, dice_coef, cldice


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss   = 0.0
    total_dice   = 0.0
    total_cldice = 0.0
    n_batches    = len(loader)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        preds = model(images)

        total_loss   += hybrid_loss(masks, preds).item()
        total_dice   += dice_coef(masks, preds).item()
        total_cldice += cldice(masks, preds).item()

    return {
        "loss":   total_loss   / n_batches,
        "dice":   total_dice   / n_batches,
        "cldice": total_cldice / n_batches,
    }


def _resolve_image_paths(dataset) -> list:
    from torch.utils.data import Subset

    for attr in ("image_paths", "img_paths", "images_list"):
        if hasattr(dataset, attr):
            return list(getattr(dataset, attr))

    if hasattr(dataset, "pairs"):
        return [p[0] for p in dataset.pairs]
    if hasattr(dataset, "samples"):
        sample0 = dataset.samples[0]
        if isinstance(sample0, (tuple, list)):
            return [s[0] for s in dataset.samples]
        return list(dataset.samples)

    if isinstance(dataset, Subset):
        base_paths = _resolve_image_paths(dataset.dataset)
        if base_paths:
            return [base_paths[i] for i in dataset.indices]

    return []


@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    import numpy as np
    from src.utils.metrics import (
        hausdorff_distance,
        compute_gt_skeleton_batch,
        skeleton_recall,
    )

    model.eval()

    all_preds  = []
    all_masks  = []
    all_images = []

    for images, masks in test_loader:
        all_images.append(images.numpy())
        images = images.to(device, non_blocking=True)
        preds  = model(images)
        all_preds.append(preds.cpu().numpy())
        all_masks.append(masks.numpy())

    all_preds  = np.concatenate(all_preds, axis=0)
    all_masks  = np.concatenate(all_masks, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    all_img_paths = _resolve_image_paths(test_loader.dataset)
    if len(all_img_paths) != len(all_images):
        if all_img_paths:
            print(
                f"Warning: resolved {len(all_img_paths)} image paths but "
                f"have {len(all_images)} samples — paths discarded to avoid "
                f"misalignment. Check that test_loader uses shuffle=False."
            )
        all_img_paths = []

    preds_t = torch.from_numpy(all_preds).to(device)
    masks_t = torch.from_numpy(all_masks).to(device)

    test_loss   = hybrid_loss(masks_t, preds_t).item()
    test_dice   = dice_coef(masks_t, preds_t).item()
    test_cldice = cldice(masks_t, preds_t).item()

    hd_scores = [
        hausdorff_distance(all_masks[i], all_preds[i])
        for i in range(len(all_preds))
    ]
    mean_hd = float(np.nanmean(hd_scores))

    print("Computing GT skeletons for test set...")
    gt_skels = compute_gt_skeleton_batch(all_masks)
    mean_srl, per_img_srl = skeleton_recall(gt_skels, all_preds)

    results = {
        "test_loss":   test_loss,
        "test_dice":   test_dice,
        "test_cldice": test_cldice,
        "hausdorff":   mean_hd,
        "skel_recall": mean_srl,
        "predictions": all_preds,
        "masks":       all_masks,
        "images":      all_images,
        "image_paths": all_img_paths,
    }

    print("\n" + "=" * 55)
    print("               TEST SET EVALUATION")
    print("=" * 55)
    print(f"  Hybrid Loss          : {test_loss:.4f}")
    print(f"  Dice Coefficient     : {test_dice:.4f}")
    print(f"  clDice               : {test_cldice:.4f}")
    print(f"  Hausdorff Distance   : {mean_hd:.2f} px")
    print(f"  Skeleton Recall (SRL): {mean_srl:.4f}")
    print("=" * 55)

    return results
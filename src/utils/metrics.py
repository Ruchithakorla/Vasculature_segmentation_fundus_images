"""
metrics.py — Loss functions and evaluation metrics for vasculature segmentation.

Implements:
  • dice_coef              — pixel-overlap Dice coefficient
  • soft_erode / soft_dilate / soft_skel — differentiable morphology primitives
  • cldice                 — skeleton-aware clDice metric (Shit et al., CVPR 2021)
  • cldice_loss            — 1 − clDice (minimisable)
  • hybrid_loss            — 0.4·clDice + 0.3·Dice + 0.3·BCE
  • hausdorff_distance     — numpy-based post-hoc metric
  • skeleton_recall        — skeleton recall with dilation tube (numpy)
  • compute_gt_skeleton_batch — CPU skeletonisation of GT masks (numpy)• soft_cldice_metric     — skeleton-based clDice variant (num_iter=5)
  • smooth_cldice_metric   — label-smoothed clDice variant
  • compute_betti_numbers / betti_error_batch — topological correctness (gudhi)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.config import (
    ALPHA_CLDICE,
    BETA_DICE,
    GAMMA_BCE,
    CLDICE_SMOOTH,
    CLDICE_ITERS,
    HAUSDORFF_THRESH,
    SKEL_TUBE_RADIUS,
)


# ── Dice coefficient ─────────────────────────────────────────────────────────

def dice_coef(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Soft Dice coefficient.

    Args:
        y_true : (B, 1, H, W) ground-truth binary mask
        y_pred : (B, 1, H, W) predicted probability map
        smooth : Laplace smoothing to avoid division by zero

    Returns:
        Scalar tensor in [0, 1]; higher = better overlap.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """1 − dice_coef (minimisable loss form)."""
    return 1.0 - dice_coef(y_true, y_pred)


# ── Soft morphological primitives (differentiable) ──────────────────────────

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """
    Differentiable erosion using min-pooling along H and W separately
    (structuring element: cross).
    """
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Differentiable dilation using 3×3 max-pooling."""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    """Morphological opening: erode then dilate."""
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, num_iter: int = CLDICE_ITERS) -> torch.Tensor:
    """
    Iterative soft skeletonisation.

    Produces a differentiable approximation of the skeleton by accumulating
    residuals between successive morphological openings.  `num_iter=10` is
    appropriate for 512×512 retinal vessel images.

    Reference: Shit et al., "clDice", CVPR 2021.
    """
    img  = img.float()
    skel = torch.zeros_like(img)
    for _ in range(num_iter):
        opened = _soft_open(img)
        delta  = F.relu(img - opened)   # skeleton contribution at this scale
        skel   = skel + delta
        img    = _soft_erode(img)       # thin further for next iteration
    return skel


# ── clDice metric & loss ─────────────────────────────────────────────────────

def cldice(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = CLDICE_SMOOTH,
    num_iter: int = CLDICE_ITERS,
) -> torch.Tensor:
    """
    Skeleton-aware clDice metric.

    Tprec penalises false skeleton branches; Tsens penalises skeleton gaps
    (the dominant cause of vessel connectivity breakdown).

    Returns a score in [0, 1]; higher = better vessel continuity.
    """
    skel_pred   = soft_skel(y_pred, num_iter)
    skel_target = soft_skel(y_true, num_iter)

    # Precision on centrelines: pred-skeleton covered by GT
    tprec = (torch.sum(skel_pred * y_true) + smooth) / (torch.sum(skel_pred) + smooth)
    # Sensitivity on centrelines: GT-skeleton covered by prediction
    tsens = (torch.sum(skel_target * y_pred) + smooth) / (torch.sum(skel_target) + smooth)

    return 2.0 * tprec * tsens / (tprec + tsens + 1e-8)


def cldice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = CLDICE_SMOOTH,
    num_iter: int = CLDICE_ITERS,
) -> torch.Tensor:
    """1 − clDice (minimisable loss form)."""
    return 1.0 - cldice(y_true, y_pred, smooth, num_iter)


# ── Hybrid loss ──────────────────────────────────────────────────────────────

_bce_fn = nn.BCELoss()


def hybrid_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = ALPHA_CLDICE,
    beta: float  = BETA_DICE,
    gamma: float = GAMMA_BCE,
) -> torch.Tensor:
    # Clamp both tensors to valid [0,1] range before any loss computation
    y_true = torch.clamp(y_true, 0.0, 1.0)
    y_pred = torch.clamp(y_pred, 0.0, 1.0)

    cl  = cldice_loss(y_true, y_pred)
    dc  = dice_loss(y_true, y_pred)
    bce = _bce_fn(y_pred, y_true)
    return alpha * cl + beta * dc + gamma * bce


# ── Post-hoc numpy metrics ───────────────────────────────────────────────────

def hausdorff_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = HAUSDORFF_THRESH,
) -> float:
    """
    Symmetric Hausdorff distance between GT and predicted binary masks.

    Args:
        y_true    : (H, W) or (1, H, W) numpy float32 ground-truth mask
        y_pred    : (H, W) or (1, H, W) numpy float32 predicted probability
        threshold : binarisation threshold for y_pred

    Returns:
        Hausdorff distance in pixels, or np.nan if either mask is empty.
    """
    from scipy.spatial.distance import directed_hausdorff

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    y_pred = (y_pred > threshold).astype(np.uint8)

    true_pts = np.argwhere(y_true > 0)
    pred_pts = np.argwhere(y_pred > 0)

    if len(true_pts) == 0 or len(pred_pts) == 0:
        return np.nan

    hd1 = directed_hausdorff(true_pts, pred_pts)[0]
    hd2 = directed_hausdorff(pred_pts, true_pts)[0]
    return float(max(hd1, hd2))


def compute_gt_skeleton_batch(
    y_true_np: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    CPU skeletonisation of a batch of GT masks.

    Args:
        y_true_np : (N, 1, H, W) numpy float32
        threshold : binarisation threshold

    Returns:
        (N, 1, H, W) numpy float32 skeleton masks
    """
    from skimage.morphology import skeletonize

    N = y_true_np.shape[0]
    skels = np.zeros_like(y_true_np)
    for i in range(N):
        mask = (y_true_np[i, 0] > threshold).astype(np.uint8)
        skel = skeletonize(mask).astype(np.float32)
        skels[i, 0] = skel
    return skels


def skeleton_recall(
    y_true_skel_np: np.ndarray,
    y_pred_np: np.ndarray,
    tube_radius: int = SKEL_TUBE_RADIUS,
    threshold: float = 0.5,
):
    """
    Skeleton Recall metric with dilation tube.

    Args:
        y_true_skel_np : (N, 1, H, W) pre-computed GT skeletons
        y_pred_np      : (N, 1, H, W) model probability outputs
        tube_radius    : dilation half-size in pixels (≈1 vessel width = 3)
        threshold      : binarisation threshold for predictions

    Returns:
        (mean_recall, per_image_recall_array)
    """
    from scipy.ndimage import binary_dilation

    struct  = np.ones((2 * tube_radius + 1, 2 * tube_radius + 1), dtype=bool)
    recalls = []

    for i in range(len(y_pred_np)):
        skel      = y_true_skel_np[i, 0]
        pred      = y_pred_np[i, 0]
        pred_bin  = (pred > threshold)
        pred_tube = binary_dilation(pred_bin, structure=struct).astype(np.float32)

        skel_sum = skel.sum()
        if skel_sum == 0:
            recalls.append(1.0)
            continue
        recalls.append(float((skel * pred_tube).sum() / skel_sum))

    return float(np.mean(recalls)), np.array(recalls)
# ── Soft-clDice & Smooth-clDice (skeleton-based variants, Shit et al. CVPR 2021) ─
 
def soft_cldice_metric(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = CLDICE_SMOOTH,
    num_iter: int = 5,
) -> torch.Tensor:
    """
    Soft-clDice score using differentiable soft skeletonisation.
 
    Functionally equivalent to `cldice()` but exposed separately to match
    the original notebook's naming and default `num_iter=5` (lighter-weight
    skeletonisation used for fast per-image evaluation).
 
    Returns a score in [0, 1]; higher = better topology overlap.
    """
    return cldice(y_true, y_pred, smooth=smooth, num_iter=num_iter)
 
 
def smooth_cldice_metric(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = CLDICE_SMOOTH,
    label_smooth: float = 0.1,
    num_iter: int = 5,
) -> torch.Tensor:
    """
    Smooth-clDice: applies label smoothing ε to the skeleton masks before
    computing precision/sensitivity. Reduces gradient vanishing on very
    thin vessel structures.
 
    Args:
        label_smooth : ε in [0, 0.2]; 0.1 is a safe default.
 
    Returns a score in [0, 1]; higher = better.
    """
    skel_pred   = soft_skel(y_pred, num_iter)
    skel_target = soft_skel(y_true, num_iter)
 
    # Label-smooth the hard skeleton masks
    skel_pred_s   = skel_pred   * (1.0 - label_smooth) + label_smooth * 0.5
    skel_target_s = skel_target * (1.0 - label_smooth) + label_smooth * 0.5
 
    tprec = (torch.sum(skel_pred_s * y_true) + smooth) / (torch.sum(skel_pred_s) + smooth)
    tsens = (torch.sum(skel_target_s * y_pred) + smooth) / (torch.sum(skel_target_s) + smooth)
 
    return 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
 
 
# ── Betti numbers via persistent homology (topological correctness) ──────────
 
def compute_betti_numbers(binary_mask_2d: np.ndarray, min_persistence: float = 0.0):
    """
    Compute β₀ (connected components) and β₁ (loops/holes) for a 2-D binary
    mask using cubical persistent homology.
 
    Requires the `gudhi` package:  pip install gudhi
 
    Args:
        binary_mask_2d  : (H, W) numpy array, values 0 or 1
        min_persistence : filter out features with persistence ≤ this value
 
    Returns:
        (beta0, beta1) integers
    """
    import gudhi
 
    filtration = 1.0 - binary_mask_2d.astype(np.float64)
 
    cc = gudhi.CubicalComplex(
        dimensions=list(filtration.shape),
        top_dimensional_cells=filtration.flatten().tolist(),
    )
    cc.compute_persistence()
    pairs = cc.persistence()
 
    beta0 = sum(
        1 for dim, (b, d) in pairs
        if dim == 0 and (d - b) > min_persistence and d != float("inf")
    )
    beta0 += sum(1 for dim, (b, d) in pairs if dim == 0 and d == float("inf"))
 
    beta1 = sum(
        1 for dim, (b, d) in pairs
        if dim == 1 and (d - b) > min_persistence
    )
 
    return beta0, beta1
 
 
def betti_error_batch(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    threshold: float = 0.5,
    min_persistence: float = 0.0,
) -> dict:
    """
    Compute mean |Δβ₀| and |Δβ₁| between GT and prediction over a batch.
    Lower = better topological correctness.
 
    Args:
        y_true_np, y_pred_np : (N, 1, H, W) numpy float32
        threshold             : binarisation threshold for predictions
        min_persistence       : noise filter for persistence pairs
 
    Returns:
        dict with mean absolute Betti errors and per-image arrays.
    """
    db0_list, db1_list = [], []
    gt_b0_list, gt_b1_list = [], []
    pr_b0_list, pr_b1_list = [], []
 
    for i in range(len(y_true_np)):
        gt_mask   = (y_true_np[i, 0] > 0.5).astype(np.uint8)
        pred_mask = (y_pred_np[i, 0] > threshold).astype(np.uint8)
 
        b0_gt,   b1_gt   = compute_betti_numbers(gt_mask,   min_persistence)
        b0_pred, b1_pred = compute_betti_numbers(pred_mask, min_persistence)
 
        db0_list.append(abs(b0_pred - b0_gt))
        db1_list.append(abs(b1_pred - b1_gt))
        gt_b0_list.append(b0_gt);   gt_b1_list.append(b1_gt)
        pr_b0_list.append(b0_pred); pr_b1_list.append(b1_pred)
 
    return {
        "mean_delta_beta0": float(np.mean(db0_list)),
        "mean_delta_beta1": float(np.mean(db1_list)),
        "mean_gt_beta0":    float(np.mean(gt_b0_list)),
        "mean_gt_beta1":    float(np.mean(gt_b1_list)),
        "mean_pred_beta0":  float(np.mean(pr_b0_list)),
        "mean_pred_beta1":  float(np.mean(pr_b1_list)),
        "per_img_db0":      np.array(db0_list),
        "per_img_db1":      np.array(db1_list),
    }
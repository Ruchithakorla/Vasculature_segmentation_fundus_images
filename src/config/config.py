"""
config.py — Central configuration for the retina vasculature segmentation project.
All hyperparameters, paths, and training settings live here.
"""

import os
from pathlib import Path

# ── Project root (two levels up from this file) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_ROOT = "/media/mmk/DATA-1/Ruchitha/MDRG_segmentation/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"

TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train", "Original")
TRAIN_MASKS_DIR  = os.path.join(DATA_ROOT, "train", "Ground truth")
TEST_IMAGES_DIR  = os.path.join(DATA_ROOT, "test",  "Original")
TEST_MASKS_DIR   = os.path.join(DATA_ROOT, "test",  "Ground truth")

# ── Output paths ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = str(PROJECT_ROOT / "models" / "checkpoints")
FINAL_MODEL_DIR = str(PROJECT_ROOT / "models" / "final_model")
LOGS_DIR        = str(PROJECT_ROOT / "outputs" / "logs")
PLOTS_DIR       = str(PROJECT_ROOT / "outputs" / "plots")

# ── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE    = 512          # resize all images to IMG_SIZE × IMG_SIZE
IN_CHANNELS = 1            # single green-channel input after CLAHE

# ── CLAHE settings ───────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_GRID    = (8, 8)

# ── Model settings ───────────────────────────────────────────────────────────
ENCODER_FILTERS = [64, 128, 256]   # encoder stage filter counts
BOTTLENECK_FILTERS = 512

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE      = 4
NUM_EPOCHS      = 50
LEARNING_RATE   = 1e-4
SEED            = 42

# ── Loss weights (hybrid_loss = α·clDice + β·Dice + γ·BCE) ──────────────────
ALPHA_CLDICE = 0.4          # clDice weight
BETA_DICE    = 0.3          # Dice loss weight
GAMMA_BCE    = 0.3          # BCE weight
CLDICE_SMOOTH = 1.0
CLDICE_ITERS  = 10          # soft-skeleton iterations

# ── LR scheduler ─────────────────────────────────────────────────────────────
LR_PATIENCE   = 5
LR_FACTOR     = 0.5
LR_MIN        = 1e-7

# ── Early stopping ───────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE = 15

# ── Inference settings ───────────────────────────────────────────────────────
PRED_THRESHOLD   = 0.3     # binarisation threshold for predicted masks
HAUSDORFF_THRESH = 0.5
BETTI_SAMPLE     = 20      # number of test images for Betti evaluation
SKEL_TUBE_RADIUS = 3       # dilation radius for skeleton recall metric

# ── Data augmentation ────────────────────────────────────────────────────────
AUG_ROT_DEGREES  = 20
AUG_ZOOM         = 0.1     # scale jitter ±10 %
AUG_HFLIP        = True
AUG_VFLIP        = True

# ── Validation split (fraction of train used for validation) ─────────────────
VAL_SPLIT = 0.1

# ── Checkpoint filename template ─────────────────────────────────────────────
BEST_CKPT_NAME   = "best_retina_unet.pth"
FINAL_MODEL_NAME = "retina_vessel_model_final.pth"

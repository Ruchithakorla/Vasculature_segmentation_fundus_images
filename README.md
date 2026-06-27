# Retina Vasculature Segmentation — PyTorch UNet

Vessel segmentation in fundus images using a UNet with a topology-preserving
hybrid loss (clDice + Dice + BCE).  Converted from TensorFlow/Keras to PyTorch.

---

## Project Structure

```
vasculature-segmentation/
├── data/
│   ├── raw/          ← place the FIVES dataset here
│   └── processed/    ← (optional) pre-computed tensors
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── config/config.py          ← all hyperparameters & paths
│   ├── data/
│   │   ├── dataset.py            ← VasculatureDataset (PyTorch Dataset)
│   │   └── dataloader.py         ← train / val / test DataLoader factory
│   ├── models/model.py           ← UNet nn.Module
│   ├── training/
│   │   ├── train.py              ← training loop + early stopping
│   │   └── validate.py           ← validation & test-set evaluation
│   ├── utils/
│   │   ├── helpers.py            ← seed, device, checkpoint helpers
│   │   ├── metrics.py            ← Dice, clDice, Hausdorff, SRL
│   │   └── visualization.py      ← training curves & prediction plots
│   └── inference/predict.py      ← single-image & batch inference
│
├── scripts/
│   ├── run_train.py              ← training entry point
│   └── run_inference.py          ← inference entry point
│
├── models/
│   ├── checkpoints/              ← best_retina_unet.pth saved here
│   └── final_model/              ← retina_vessel_model_final.pth
│
├── outputs/
│   ├── logs/                     ← training_log.csv
│   └── plots/                    ← training curves & prediction grids
│
├── requirements.txt
└── README.md
```

---

## Dataset

[FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation](https://www.kaggle.com/datasets/ruchithakorla/fives-dataset)

Expected layout under `data/raw/`:

```
FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/
├── train/
│   ├── Original/          ← fundus images (.png)
│   └── Ground truth/      ← binary masks  (.png)
└── test/
    ├── Original/
    └── Ground truth/
```

Set the `DATA_ROOT` environment variable if your dataset lives elsewhere:

```bash
export DATA_ROOT=/path/to/your/fives/dataset
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU (replace `cu121` with your CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Train

```bash
python scripts/run_train.py
# Options:
python scripts/run_train.py --epochs 50 --batch-size 4 --lr 1e-4
```

Training artefacts:
- `models/checkpoints/best_retina_unet.pth` — best val-Dice checkpoint
- `models/final_model/retina_vessel_model_final.pth` — final weights
- `outputs/logs/training_log.csv` — per-epoch metrics
- `outputs/plots/training_curves.png` — loss/Dice/clDice curves

### 3. Inference

```bash
# Predict a directory of images
python scripts/run_inference.py --input data/raw/.../test/Original --output outputs/preds

# Evaluate on the test set (metrics + visualisations)
python scripts/run_inference.py --evaluate

# Custom checkpoint
python scripts/run_inference.py --checkpoint models/checkpoints/best_retina_unet.pth --evaluate
```

---

## Architecture

```
Input (1, 512, 512)
    │
    ├─ ConvBlock(1→64)  ──────────────────────────────────── skip₁
    │  MaxPool
    ├─ ConvBlock(64→128) ─────────────────────────────────── skip₂
    │  MaxPool
    ├─ ConvBlock(128→256) ────────────────────────────────── skip₃
    │  MaxPool
    │
    ├─ Bottleneck: ConvBlock(256→512)
    │
    ├─ Upsample + cat(skip₃) → ConvBlock(512+256→256)
    ├─ Upsample + cat(skip₂) → ConvBlock(256+128→128)
    ├─ Upsample + cat(skip₁) → ConvBlock(128+64→64)
    │
    └─ Conv1×1 → Sigmoid → Output (1, 512, 512)
```

---

## Loss Function

**Hybrid Loss** = 0.4 × clDice + 0.3 × Dice + 0.3 × BCE

- **clDice** (Shit et al., CVPR 2021): skeleton-aware, topology-preserving.
  Penalises both false skeleton branches and connectivity gaps.
- **Dice**: pixel-overlap overlap.
- **BCE**: standard binary cross-entropy.

Weights are configurable in `src/config/config.py`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Dice Coefficient | Pixel-overlap F1 score |
| clDice | Skeleton-aware vessel continuity |
| Hausdorff Distance | Worst-case boundary error (pixels) |
| Skeleton Recall (SRL) | GT-skeleton coverage with dilation tube |

---

## Configuration

All hyperparameters live in `src/config/config.py`.  Key settings:

| Parameter | Default | Description |
|---|---|---|
| `IMG_SIZE` | 512 | Input image resolution |
| `BATCH_SIZE` | 4 | Training batch size |
| `NUM_EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | Initial Adam LR |
| `ALPHA_CLDICE` | 0.4 | clDice loss weight |
| `PRED_THRESHOLD` | 0.3 | Binarisation threshold |
| `CLDICE_ITERS` | 10 | Soft-skeleton iterations |

---

## License

For research and educational use.  The FIVES dataset has its own licence —
please refer to the original dataset page.

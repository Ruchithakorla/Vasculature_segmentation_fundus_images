# Retina Vasculature Segmentation — PyTorch UNet

Vessel segmentation in fundus images using a UNet with a topology-preserving
hybrid loss (clDice + Dice + BCE)

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
│   │   └── dataloader.py         ← train / val / test DataLoader factory (here we do preprocessing,normalization )
│   ├── models/
│   │   ├── model.py              ← UNet nn.Module
│   │   └── mdrg_unet.py          ← mdrg module unet
│   ├── train/
│   │   ├── train.py              ← training loop + early stopping
│   │   └── validate.py           ← validation & test-set evaluation
│   ├── utils/
│   │   ├── helpers.py            ← seed, device, checkpoint helpers
│   │   ├── metrics.py            ← Dice, clDice, Hausdorff, SRL
│   │   └── visualization.py      ← training curves & prediction plots
│   └── inference/predict.py      ← single-image & batch inference
│
├── scripts/
│   ├── run_compare_models.py      ← comapre the baseline unet and mdrg module model
│   ├── run_crosschecking_dataset.py ← datasets  checking
│   ├── run_full_evaluation_all metrices.py ←all metrices(dice,cldece,soft cldice,bettinumbers)
│   ├── run_trsain_mdrg.py        ← mdrg module model
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

## Architecture

The proposed model extends the standard UNet by inserting a 
Morphology-Driven Region Growing (MDRG) module after every encoder convolution block. 
The MDRG module enhances vessel morphology using deformable convolutions,
 multi-scale context aggregation, and channel attention before spatial downsampling.

Network Architecture
Input (1, 512, 512)
    │
    ├─ ConvBlock(1→64)
    │      │
    │      └── MDRG Module(64)
    │              │
    │              ├── Skip₁ = concat(ConvBlock, MDRG)
    │              └── MaxPool
    │
    ├─ ConvBlock(64→128)
    │      │
    │      └── MDRG Module(128)
    │              │
    │              ├── Skip₂ = concat(ConvBlock, MDRG)
    │              └── MaxPool
    │
    ├─ ConvBlock(128→256)
    │      │
    │      └── MDRG Module(256)
    │              │
    │              ├── Skip₃ = concat(ConvBlock, MDRG)
    │              └── MaxPool
    │
    ├─ Bottleneck: ConvBlock(256→512)
    │
    ├─ Upsample + concat(Skip₃) → ConvBlock(512+512→256)
    ├─ Upsample + concat(Skip₂) → ConvBlock(256+256→128)
    ├─ Upsample + concat(Skip₁) → ConvBlock(128+128→64)
    │
    └─ 1×1 Convolution → Sigmoid → Output (1, 512, 512)
MDRG Module

Each encoder stage contains one Morphology-Driven Region Growing (MDRG) module that refines encoder features before downsampling.

Encoder Feature
      │
      ├───────────────┬───────────────┬───────────────┬
      │               │               │               │
      ▼               ▼               ▼               ▼
Deform Branch X  Deform Branch Y  Deform Branch Z    ASPP
      │               │               │               │
      └───────────────┴───────────────┴───────────────┘
                      │
                 Concatenation
                      │
                 1×1 Pointwise Conv
                      │
                   F_fusion
                      │
            ┌─────────┴─────────┐
            │                   │
           GMP                 GAP
            │                   │
            └────── Addition ───┘
                      │
                     MLP
                      │
                  Sigmoid
                      │
          Channel-wise Multiplication
                      │
     Concatenate(F_fusion, Attention Output)
                      │
                 1×1 Convolution
                      │
                 MDRG Output



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

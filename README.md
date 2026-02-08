# Line2Former

A transformer-based architecture for extracting individual lines from line plot images as polylines. Line2Former uses learnable line queries with a DETR-style decoder to predict a set of polylines, their widths, and objectness scores directly from a plot image.

## Architecture

```
Input Image (H, W, 3)
        |
  Pretrained Backbone (HRNet / Dilated ResNet-50)
        |
  Line-Aware Feature Extractor (oriented convolutions + ridge detection)
        |
  Positional Encoding (2D sinusoidal)
        |
  Transformer Decoder (6 layers, 8 heads)
   - Learnable line queries (DETR-style per-layer position injection)
   - Cross-attention to backbone features
        |
  Prediction Heads
   |         |            |
Centerline  Width     Objectness
(B,N,P,2)  (B,N,P)    (B,N)
```

**Key design choices:**
- **Set prediction**: Hungarian matching pairs predicted lines to GT lines during training
- **Chamfer distance**: Symmetric polyline distance metric for centerline loss
- **Differentiable renderer**: Memory-efficient soft rasterization for optional mask supervision
- **Zero-init content queries**: Content queries start as zeros; positional embeddings injected at every decoder layer

## Supported Backbones

| Backbone | Source | Pretrained | Notes |
|---|---|---|---|
| `hrnet_w32` (default) | timm | ImageNet | Best accuracy/speed tradeoff |
| `hrnet_w18` | timm | ImageNet | Lighter, faster |
| `hrnet_w48` | timm | ImageNet | Highest capacity |
| `dilated_resnet50` | torchvision | ImageNet | Dilated layers 3-4, maintains 1/4 resolution |
| `hrnet` | custom | No | Custom implementation, no pretrained weights |
| `lightweight` | custom | No | Simple CNN for quick experiments |

## Project Structure

```
PloXtractor/
├── line2former/
│   ├── models/
│   │   ├── line2former.py       # Main model (Line2Former, Line2FormerLite)
│   │   ├── backbone.py          # PretrainedHRNetBackbone, DilatedResNetBackbone
│   │   ├── decoder.py           # Transformer decoder + prediction heads
│   │   ├── line_aware_conv.py   # Oriented convolutions + ridge detection
│   │   └── renderer.py          # Differentiable line renderer
│   ├── losses/
│   │   ├── composite.py         # Line2FormerLoss (Hungarian matching)
│   │   ├── chamfer.py           # Chamfer distance
│   │   └── matcher.py           # Hungarian matcher
│   ├── dataset/
│   │   ├── dataset.py           # Line2FormerDataset (COCO-style annotations)
│   │   ├── transforms.py        # Augmentations for images + polylines
│   │   └── utils.py             # Polyline resampling, normalization
│   ├── configs/
│   │   └── default.yaml         # Default hyperparameters
│   ├── tests/                   # Unit tests
│   └── train.py                 # PyTorch Lightning training script
├── synthetic_dataset_generator/
│   ├── dataset_generator.py     # Main generator (COCO + LineFormer formats)
│   ├── equation_bank.py         # 17 equation families, 30+ subtypes
│   ├── style_bank.py            # Colors, line styles, annotations, legends
│   ├── config.yaml              # Generation configuration
│   └── coco_utils.py            # COCO format utilities
├── evaluate.py                  # Evaluation with Chamfer/IoU/AP metrics
└── inference.py                 # Single image or batch inference
```

## Installation

```bash
# Core dependencies
pip install torch torchvision
pip install pytorch-lightning
pip install timm                   # For pretrained HRNet backbones
pip install opencv-python numpy scipy

# For synthetic data generation
pip install matplotlib tqdm pyyaml

# For evaluation
pip install pycocotools
```

## Quick Start

### 1. Generate Synthetic Data

```bash
cd synthetic_dataset_generator

# Generate training + validation data (with visual noise)
python dataset_generator.py \
    --output-dir ../data \
    --num-train 10000 \
    --num-val 1000 \
    --seed 42 \
    --format lineformer \
    --save-metadata

# Generate clean data (no noise)
python dataset_generator.py \
    --output-dir ../data_clean \
    --num-train 5000 \
    --num-val 500 \
    --no-noise
```

The generator creates:
```
data/
├── train/
│   ├── images/            # PNG plot images
│   ├── annotations.json   # Polyline annotations
│   └── stats.json
└── val/
    ├── images/
    └── annotations.json
```

### 2. Train

```bash
cd line2former

# Train with default settings (HRNet-W32, 6 decoder layers)
python train.py \
    --data_root ../data \
    --backbone hrnet_w32 \
    --batch_size 2 \
    --max_epochs 100 \
    --learning_rate 1e-4

# Train lightweight model for quick experiments
python train.py \
    --data_root ../data \
    --backbone lightweight \
    --d_model 128 \
    --num_decoder_layers 3 \
    --batch_size 8

# Resume from checkpoint
python train.py \
    --data_root ../data \
    --resume_from outputs/checkpoints/best.ckpt
```

### 3. Evaluate

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/best.ckpt \
    --data-dir data \
    --split val \
    --output-dir eval_results \
    --visualize
```

Reported metrics:
- **Chamfer Distance** (mean, median, std) - polyline accuracy
- **Width MAE** - line width prediction error
- **Precision / Recall / F1** at IoU thresholds 0.25, 0.50, 0.75
- **mAP** - mean Average Precision across IoU thresholds

### 4. Inference

```bash
# Single image
python inference.py \
    --model outputs/checkpoints/best.ckpt \
    --input path/to/plot.png \
    --output-dir results \
    --save-vis --save-json

# Batch inference on a directory
python inference.py \
    --model outputs/checkpoints/best.ckpt \
    --input path/to/images/ \
    --output-dir results \
    --confidence-threshold 0.5

# Save binary masks
python inference.py \
    --model outputs/checkpoints/best.ckpt \
    --input path/to/plot.png \
    --output-dir results \
    --save-masks
```

Output formats:
- **JSON**: Polyline coordinates + confidence scores per line
- **Visualization**: Color-coded overlay with labeled polylines
- **Masks**: Binary PNG masks (combined and per-line)

## Model Configuration

Default hyperparameters (`line2former/configs/default.yaml`):

```yaml
model:
  backbone: hrnet_w32
  num_queries: 20          # Max lines per image
  max_points: 50           # Points per polyline
  d_model: 256             # Hidden dimension
  nhead: 8                 # Attention heads
  num_decoder_layers: 6    # Decoder depth

data:
  image_size: [480, 640]   # Input resolution (H, W)
  max_lines: 10

loss:
  weight_centerline: 5.0   # Chamfer distance weight
  weight_width: 1.0        # Width L1 weight
  weight_objectness: 2.0   # Focal loss weight
  weight_mask: 0.0         # Mask loss (disabled by default)

training:
  learning_rate: 1e-4
  weight_decay: 1e-4
  lr_scheduler: cosine
```

## Synthetic Data Generator

The generator creates realistic line plot images with pixel-perfect ground truth.

**17 equation families:**

| Category | Examples |
|---|---|
| Linear | `y = mx + b` |
| Polynomial | Degree 2-4 polynomials |
| Trigonometric | sin, cos, sin+cos, tanh |
| Exponential | Growth, decay, shifted |
| Logarithmic | `a*ln(bx + c)` |
| Power | `ax^n` (integer + fractional) |
| Sigmoid | Logistic function |
| Composite | Linear+sin, damped oscillation, etc. |
| Rational | `a/(x-h)`, quadratic/linear ratios |
| Piecewise | Step, sawtooth, square wave, triangle wave |
| Gaussian | Single, bimodal, skewed bell curves |
| Absolute Value | V-shape, ramp (ReLU), piecewise-linear |
| Logistic Growth | Logistic curve, Gompertz growth |
| Fourier | Sum of 2-4 harmonics |
| Spline Noise | Random cubic spline (simulates real data) |
| Sqrt / Cbrt | Square root, cube root curves |
| Reciprocal Trig | Clamped sec(x), csc(x) |

**Visual noise pipeline** (enabled by default, disable with `--no-noise`):

*Matplotlib-level:* text annotations with arrows, horizontal/vertical reference lines, shaded confidence regions, data callouts, error bars, scatter overlays, statistics textboxes, secondary y-axis, inset zoom

*Post-rendering:* JPEG compression artifacts, Gaussian blur, salt-and-pepper noise, brightness/contrast jitter, background textures, watermark overlays, border padding, resolution degradation

## Loss Function

The composite loss combines four components via Hungarian matching:

```
L = w_c * L_chamfer + w_w * L_width + w_o * L_objectness + w_m * L_mask
```

- **L_chamfer**: Symmetric Chamfer distance between predicted and GT polylines
- **L_width**: L1 loss on predicted vs GT line widths
- **L_objectness**: Focal loss for line detection (handles class imbalance)
- **L_mask**: Dice + BCE on rendered masks (optional, weight=0 by default)

Hungarian matching uses a cost matrix of `chamfer + width_L1 + objectness_BCE` to find optimal bipartite assignment.

## Tests

```bash
cd line2former
python -m pytest tests/ -v
```

## License

See LICENSE file for details.

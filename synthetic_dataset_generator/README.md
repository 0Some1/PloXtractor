# Synthetic Line Plot Dataset Generator

A tool for generating synthetic line plot images with perfect ground truth instance segmentation masks, compatible with Detectron2 and Mask2Former.

## Overview

This generator creates diverse line plot images by combining:
- **Equation Bank**: Mathematical functions (linear, polynomial, trigonometric, exponential, etc.)
- **Style Bank**: Visual styles (colors, line styles, legends, titles, grids)
- **Automatic Mask Generation**: Pixel-perfect binary masks for each line instance

## Output Format

The dataset is generated in **COCO instance segmentation format**, directly compatible with:
- Detectron2
- Mask2Former
- MMDetection
- Any framework supporting COCO format

## Quick Start

### 1. Test the Generator

```bash
python test_quick.py
```

This generates 10 sample images and visualizations to verify everything works.

### 2. Generate Full Dataset

```bash
python dataset_generator.py \
    --output-dir ./SyntheticLinePlotDataset \
    --num-train 10000 \
    --num-val 1000 \
    --num-test 1000 \
    --seed 42 \
    --save-metadata
```

### 3. Visualize Results

```bash
# Grid visualization
python visualize.py --dataset-dir ./SyntheticLinePlotDataset --split train --mode grid --num-samples 16

# Single sample detailed view
python visualize.py --dataset-dir ./SyntheticLinePlotDataset --split train --mode detailed --image-id 1

# Print statistics
python visualize.py --dataset-dir ./SyntheticLinePlotDataset --split train --mode stats
```

## Output Structure

```
SyntheticLinePlotDataset/
├── train/
│   ├── images/
│   │   ├── image_000000.png
│   │   ├── image_000001.png
│   │   └── ...
│   ├── annotations.json      # COCO format annotations
│   └── stats.json            # Generation statistics
├── val/
│   ├── images/
│   └── annotations.json
├── test/
│   ├── images/
│   └── annotations.json
├── metadata/                  # Optional: equation/style info per image
│   ├── train/
│   ├── val/
│   └── test/
└── debug_masks/              # Optional: individual mask images
    ├── train/
    ├── val/
    └── test/
```

## COCO Annotation Format

```json
{
  "info": {...},
  "categories": [{"id": 1, "name": "line", "supercategory": "chart_element"}],
  "images": [
    {"id": 1, "file_name": "image_000001.png", "width": 640, "height": 480}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],  // polygon format
      "area": 1234,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ]
}
```

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./output` | Output directory |
| `--num-train` | 1000 | Number of training samples |
| `--num-val` | 200 | Number of validation samples |
| `--num-test` | 200 | Number of test samples |
| `--seed` | 42 | Random seed for reproducibility |
| `--mask-thickness` | 3 | Line thickness in masks (pixels) |
| `--save-debug-masks` | False | Save individual mask images |
| `--save-metadata` | False | Save generation metadata per image |
| `--use-rle` | False | Use RLE encoding instead of polygon |

### Equation Types

The generator includes these mathematical functions:

- **Linear**: `y = mx + b`
- **Polynomial**: Degree 2-4 polynomials
- **Trigonometric**: sin, cos, tanh, combinations
- **Exponential**: Growth/decay functions
- **Logarithmic**: Log functions with shifts
- **Power**: `y = ax^n`
- **Sigmoid**: Logistic curves
- **Composite**: Combinations (e.g., `x² + sin(x)`)
- **Rational**: Ratio of polynomials

### Style Variations

- Multiple color palettes (standard, vibrant, pastel, dark)
- Line styles: solid, dashed, dotted, dash-dot
- Random figure sizes and DPI
- Optional: titles, legends, axis labels, grids
- Various background colors

## Training with Detectron2

```python
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Register the dataset
register_coco_instances(
    "synthetic_lines_train",
    {},
    "SyntheticLinePlotDataset/train/annotations.json",
    "SyntheticLinePlotDataset/train/images"
)

register_coco_instances(
    "synthetic_lines_val",
    {},
    "SyntheticLinePlotDataset/val/annotations.json",
    "SyntheticLinePlotDataset/val/images"
)

# Configure training
cfg = get_cfg()
cfg.DATASETS.TRAIN = ("synthetic_lines_train",)
cfg.DATASETS.TEST = ("synthetic_lines_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "line" class

# Train!
trainer = DefaultTrainer(cfg)
trainer.train()
```

## Training with Mask2Former

```python
# Use the COCO annotations directly with Mask2Former config
# The format is already compatible

from detectron2.config import get_cfg
from mask2former import add_maskformer2_config

cfg = get_cfg()
add_maskformer2_config(cfg)

cfg.DATASETS.TRAIN = ("synthetic_lines_train",)
cfg.DATASETS.TEST = ("synthetic_lines_val",)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1

# Continue with Mask2Former training...
```

## Key Design Decisions

### Mask Generation (per user specifications)

- **No markers in masks**: Only the line path is included
- **No legend boxes**: Legend entries are not part of masks
- **Axes ignored**: X/Y axis lines are not included in masks
- **50% threshold**: Anti-aliased edges use 50% threshold for binarization
- **3px line thickness**: Default mask line width

### Diversity

- Equations are sampled with weighted probabilities
- Diversity mode ensures different equation types within same plot
- Styles are selected to be visually distinguishable

## Files

| File | Description |
|------|-------------|
| `dataset_generator.py` | Main generator script |
| `equation_bank.py` | Mathematical function definitions |
| `style_bank.py` | Visual styling options |
| `coco_utils.py` | COCO format conversion utilities |
| `visualize.py` | Visualization tools |
| `config.yaml` | Default configuration |
| `test_quick.py` | Quick test script |

## Requirements

```
numpy
matplotlib
opencv-python (cv2)
tqdm
pycocotools
pyyaml
```

## Examples

### Sample Generated Images

After running `test_quick.py`, check:
- `test_output/test_grid.png` - Grid of samples with mask overlays
- `test_output/debug_masks/` - Individual binary masks
- `test_output/metadata/` - JSON files with equation details

### Metadata Example

```json
{
  "image_filename": "image_000002.png",
  "num_lines": 7,
  "x_range": [-4.35, 4.35],
  "y_range": [-3.96, 11.74],
  "equations": [
    {
      "name": "decay_exp",
      "category": "exponential",
      "latex": "y = 0.54e^{-0.04x}",
      "params": {"a": 0.54, "b": -0.04, "c": -1.60}
    },
    ...
  ]
}
```

## License

MIT License - Feel free to use for research and commercial purposes.

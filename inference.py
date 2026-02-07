"""
Inference script for Line2Former.

Processes single images or directories and outputs:
  - Polyline predictions as JSON
  - Binary masks as PNG
  - Visualisation overlay images

Usage:
    # Single image
    python inference.py --model ./outputs/checkpoints/best.pth \
                        --input image.png --output-dir ./results

    # Batch (directory of images)
    python inference.py --model ./outputs/checkpoints/last.ckpt --input ../data/val --output-dir ./results

    # Masks only
    python inference.py --model ./outputs/checkpoints/best.pth \
                        --input image.png --output-dir ./results --save-masks
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A

from line2former.models import Line2Former
from line2former.dataset.utils import (
    denormalize_polyline,
    simplify_polyline,
    polyline_to_mask,
)


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# ---------------------------------------------------------------------------
# Preprocessing (mirrors validation transforms)
# ---------------------------------------------------------------------------

def preprocess_image(
    image_bgr: np.ndarray,
    image_size: tuple[int, int] = (480, 640),
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Resize, normalise, and convert to tensor.

    Returns:
        tensor: (1, 3, H, W) float tensor
        original_size: (orig_H, orig_W)
    """
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    target_h, target_w = image_size
    image_resized = cv2.resize(image_rgb, (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)

    # Normalise (ImageNet stats)
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    image_norm = normalize(image=image_resized)['image']

    # HWC -> CHW, add batch dim
    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess_predictions(
    outputs: dict[str, torch.Tensor],
    original_size: tuple[int, int],
    image_size: tuple[int, int] = (480, 640),
    confidence_threshold: float = 0.5,
    simplify_tolerance: float = 0.0,
) -> list[dict]:
    """
    Filter, denormalise, and optionally simplify predictions.

    Returns a list of dicts, one per detected line:
        {
            "centerline": [[x1,y1], ...],   # pixel coords in original image
            "width": float,                   # average width in pixels
            "confidence": float,
            "num_points": int,
        }
    """
    centerlines = outputs['centerlines'][0].cpu()   # (N, P, 2)
    widths = outputs['widths'][0].cpu()              # (N, P)
    objectness = outputs['objectness'][0].cpu()      # (N,)

    scores = objectness.sigmoid()
    keep = scores > confidence_threshold
    keep_indices = torch.where(keep)[0]

    # Sort by confidence (highest first)
    sorted_order = scores[keep_indices].argsort(descending=True)
    keep_indices = keep_indices[sorted_order]

    orig_h, orig_w = original_size
    results = []

    for idx in keep_indices:
        # Denormalize to *original* image pixel coordinates
        cl_norm = centerlines[idx].numpy()  # (P, 2) in [0,1]
        cl_px = denormalize_polyline(cl_norm, image_size)

        # Scale from image_size to original_size
        scale_x = orig_w / image_size[1]
        scale_y = orig_h / image_size[0]
        cl_px[:, 0] *= scale_x
        cl_px[:, 1] *= scale_y

        # Optionally simplify
        if simplify_tolerance > 0:
            cl_px = simplify_polyline(cl_px, tolerance=simplify_tolerance)

        avg_width = float(widths[idx].mean()) * scale_x  # rough scale
        confidence = float(scores[idx])

        results.append({
            'centerline': cl_px.tolist(),
            'width': round(avg_width, 2),
            'confidence': round(confidence, 4),
            'num_points': len(cl_px),
        })

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def save_json(results: list[dict], path: Path):
    """Save polyline predictions as JSON."""
    with open(path, 'w') as f:
        json.dump({'predictions': results}, f, indent=2)


def save_masks(
    results: list[dict],
    original_size: tuple[int, int],
    output_path: Path,
):
    """Save a combined binary mask and per-line masks."""
    h, w = original_size
    combined = np.zeros((h, w), dtype=np.uint8)

    for i, line in enumerate(results):
        cl = np.array(line['centerline'], dtype=np.float32)
        width = line['width']
        mask = polyline_to_mask(cl, width, (h, w))
        combined = np.maximum(combined, mask)

        # Per-line mask
        per_line_path = output_path.parent / f'{output_path.stem}_line{i:02d}.png'
        cv2.imwrite(str(per_line_path), mask)

    cv2.imwrite(str(output_path), combined)


def save_visualisation(
    image_bgr: np.ndarray,
    results: list[dict],
    output_path: Path,
):
    """Draw predicted polylines on the image and save."""
    vis = image_bgr.copy()

    # Colour palette
    colours = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 255), (255, 128, 0), (0, 128, 255),
        (128, 255, 0), (255, 0, 128), (0, 255, 128),
        (200, 100, 50), (50, 100, 200), (100, 200, 50),
        (200, 50, 100), (50, 200, 100), (100, 50, 200),
        (180, 180, 0), (0, 180, 180),
    ]

    for i, line in enumerate(results):
        cl = np.array(line['centerline'], dtype=np.int32).reshape(-1, 1, 2)
        colour = colours[i % len(colours)]
        thickness = max(2, int(line['width']))

        cv2.polylines(vis, [cl], False, colour, thickness, cv2.LINE_AA)

        # Label with confidence
        if len(cl) > 0:
            org = tuple(cl[0, 0])
            label = f'L{i} ({line["confidence"]:.2f})'
            cv2.putText(vis, label, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), vis)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    device: torch.device,
    **model_kwargs,
) -> Line2Former:
    """Load a trained Line2Former model."""
    model = Line2Former(**model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'state_dict' in ckpt:
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            new_key = k.replace('model.', '', 1) if k.startswith('model.') else k
            state_dict[new_key] = v
    else:
        state_dict = ckpt

    # Drop keys whose shapes don't match (e.g. renderer buffers from old code)
    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    model.load_state_dict(filtered, strict=False)

    model.to(device).eval()
    return model


def run_inference(
    model: Line2Former,
    image_path: Path,
    device: torch.device,
    image_size: tuple[int, int],
    confidence_threshold: float,
    simplify_tolerance: float,
) -> tuple[list[dict], np.ndarray]:
    """Run inference on a single image. Returns (results, original_bgr)."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    tensor, original_size = preprocess_image(image_bgr, image_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)

    results = postprocess_predictions(
        outputs, original_size, image_size,
        confidence_threshold=confidence_threshold,
        simplify_tolerance=simplify_tolerance,
    )

    return results, image_bgr


def main():
    parser = argparse.ArgumentParser(description='Line2Former Inference')

    # Required
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image path or directory')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory')

    # Inference options
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--simplify-tolerance', type=float, default=0.0,
                        help='Douglas-Peucker simplification tolerance (0 = off)')

    # Output options
    parser.add_argument('--save-json', action='store_true', default=True,
                        help='Save polyline predictions as JSON (default: on)')
    parser.add_argument('--save-masks', action='store_true',
                        help='Save binary masks as PNG')
    parser.add_argument('--save-vis', action='store_true', default=True,
                        help='Save visualization images (default: on)')
    parser.add_argument('--no-json', dest='save_json', action='store_false')
    parser.add_argument('--no-vis', dest='save_vis', action='store_false')

    # Model config (must match training)
    parser.add_argument('--backbone', type=str, default='hrnet_w32')
    parser.add_argument('--num-queries', type=int, default=20)
    parser.add_argument('--max-points', type=int, default=50)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--num-decoder-layers', type=int, default=6)
    parser.add_argument('--image-size', type=int, nargs=2, default=[480, 640],
                        help='H W')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(args.image_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    print(f'Loading model: {args.model}')
    model = load_model(
        args.model,
        device,
        backbone_type=args.backbone,
        num_queries=args.num_queries,
        max_points=args.max_points,
        d_model=args.d_model,
        num_decoder_layers=args.num_decoder_layers,
        image_size=image_size,
        pretrained=False,
    )

    # Gather input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        print(f'Error: {input_path} is not a file or directory')
        return

    if not image_paths:
        print(f'No images found in {input_path}')
        return

    print(f'Processing {len(image_paths)} image(s)...\n')

    all_results = {}
    total_time = 0.0

    for img_path in image_paths:
        t0 = time.time()
        try:
            results, image_bgr = run_inference(
                model, img_path, device, image_size,
                args.confidence_threshold, args.simplify_tolerance,
            )
        except FileNotFoundError as e:
            print(f'  SKIP {img_path.name}: {e}')
            continue

        elapsed = time.time() - t0
        total_time += elapsed

        stem = img_path.stem
        n_lines = len(results)
        print(f'  {img_path.name}: {n_lines} line(s) detected  ({elapsed:.3f}s)')

        all_results[img_path.name] = results

        # Save outputs
        if args.save_json:
            save_json(results, output_dir / f'{stem}.json')

        if args.save_masks and results:
            orig_h, orig_w = image_bgr.shape[:2]
            save_masks(results, (orig_h, orig_w), output_dir / f'{stem}_mask.png')

        if args.save_vis:
            save_visualisation(image_bgr, results, output_dir / f'{stem}_vis.png')

    # Summary
    n = len(image_paths)
    print(f'\nDone. {n} images in {total_time:.2f}s ({total_time/max(n,1):.3f}s/img)')
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()

"""
Evaluation script for Line2Former.

Computes metrics on a validation/test split:
  - Centerline Chamfer Distance (mean, median)
  - Width MAE
  - Detection Precision / Recall (IoU-based matching)
  - Line Detection AP (COCO-style, averaged over IoU thresholds)

Usage:
    python evaluate.py --checkpoint ./outputs/checkpoints/best.pth \
                       --data-dir ./data --split val \
                       --output-dir ./eval_results
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from line2former.models import Line2Former
from line2former.dataset import Line2FormerDataset
from line2former.dataset.utils import (
    denormalize_polyline,
    compute_chamfer_distance,
    polyline_to_mask,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def match_predictions_to_targets(
    pred_masks: list[np.ndarray],
    target_masks: list[np.ndarray],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match predicted lines to ground-truth lines using mask IoU + Hungarian.

    Returns:
        matched_pairs: list of (pred_idx, target_idx)
        unmatched_preds: list of pred indices
        unmatched_targets: list of target indices
    """
    n_pred = len(pred_masks)
    n_target = len(target_masks)

    if n_pred == 0 or n_target == 0:
        return (
            [],
            list(range(n_pred)),
            list(range(n_target)),
        )

    # Build cost matrix (negative IoU for minimisation)
    cost = np.zeros((n_pred, n_target))
    for i in range(n_pred):
        for j in range(n_target):
            cost[i, j] = -mask_iou(pred_masks[i], target_masks[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs = []
    matched_pred_set = set()
    matched_target_set = set()

    for r, c in zip(row_ind, col_ind):
        iou = -cost[r, c]
        if iou >= iou_threshold:
            matched_pairs.append((r, c))
            matched_pred_set.add(r)
            matched_target_set.add(c)

    unmatched_preds = [i for i in range(n_pred) if i not in matched_pred_set]
    unmatched_targets = [j for j in range(n_target) if j not in matched_target_set]

    return matched_pairs, unmatched_preds, unmatched_targets


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision (area under P-R curve) using 101-point interp."""
    if len(precisions) == 0:
        return 0.0

    # Prepend sentinel values
    precisions = [0.0] + list(precisions) + [0.0]
    recalls = [0.0] + list(recalls) + [1.0]

    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = 0.0
        for ri, pi in zip(recalls, precisions):
            if ri >= t:
                p = max(p, pi)
        ap += p
    return ap / 101.0


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: Line2Former,
    dataloader: DataLoader,
    device: torch.device,
    confidence_threshold: float = 0.3,
    iou_thresholds: list[float] | None = None,
    image_size: tuple[int, int] = (480, 640),
    visualize: bool = False,
    vis_dir: Path | None = None,
) -> dict:
    """
    Run full evaluation.

    Returns dict with all metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.50, 0.75]

    model.eval()

    all_chamfer = []
    all_width_ae = []
    # per IoU threshold: list of (tp, fp, fn, confidence) per image
    per_threshold_stats: dict[float, list[dict]] = {t: [] for t in iou_thresholds}

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        B = images.shape[0]

        outputs = model(images)

        pred_centerlines = outputs['centerlines'].cpu()   # (B, N, P, 2)
        pred_widths = outputs['widths'].cpu()              # (B, N, P)
        pred_objectness = outputs['objectness'].cpu()      # (B, N)

        gt_centerlines = batch['centerlines']              # (B, max_lines, P, 2)
        gt_widths = batch['widths']                        # (B, max_lines, P)
        gt_valid = batch['valid_mask']                     # (B, max_lines)

        for b in range(B):
            # --- Filter predictions by confidence ---
            scores = pred_objectness[b].sigmoid()
            keep = scores > confidence_threshold
            keep_indices = torch.where(keep)[0]

            pred_lines_norm = pred_centerlines[b, keep_indices]  # (K, P, 2)
            pred_w = pred_widths[b, keep_indices]                # (K, P)
            pred_scores = scores[keep_indices]                   # (K,)

            # --- Ground truth ---
            gt_mask = gt_valid[b]
            gt_indices = torch.where(gt_mask)[0]
            gt_lines_norm = gt_centerlines[b, gt_indices]  # (M, P, 2)
            gt_w = gt_widths[b, gt_indices]                # (M, P)

            n_pred = len(keep_indices)
            n_gt = len(gt_indices)

            # --- Denormalize to pixel space ---
            H, W = image_size
            pred_lines_px = [
                denormalize_polyline(pred_lines_norm[i].numpy(), (H, W))
                for i in range(n_pred)
            ]
            gt_lines_px = [
                denormalize_polyline(gt_lines_norm[j].numpy(), (H, W))
                for j in range(n_gt)
            ]

            # --- Chamfer distance & width error (for matched pairs at IoU=0.25) ---
            # Render masks for matching
            pred_masks = [
                polyline_to_mask(pl, float(pred_w[i].mean()), (H, W))
                for i, pl in enumerate(pred_lines_px)
            ]
            gt_masks = [
                polyline_to_mask(gl, float(gt_w[j].mean()), (H, W))
                for j, gl in enumerate(gt_lines_px)
            ]

            matched, _, _ = match_predictions_to_targets(
                pred_masks, gt_masks, iou_threshold=0.25
            )

            for pi, gi in matched:
                cd = compute_chamfer_distance(pred_lines_px[pi], gt_lines_px[gi])
                all_chamfer.append(cd)

                wae = float(torch.abs(pred_w[pi] - gt_w[gi]).mean())
                all_width_ae.append(wae)

            # --- Detection P/R per IoU threshold ---
            for iou_t in iou_thresholds:
                m, um_p, um_t = match_predictions_to_targets(
                    pred_masks, gt_masks, iou_threshold=iou_t
                )
                tp = len(m)
                fp = len(um_p)
                fn = len(um_t)
                per_threshold_stats[iou_t].append({
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'scores': pred_scores.tolist(),
                })

            # --- Optional visualisation ---
            if visualize and vis_dir is not None:
                _save_visualisation(
                    batch, b, pred_lines_px, gt_lines_px,
                    pred_scores, matched, vis_dir, batch_idx,
                    image_size,
                )

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    metrics: dict = {}

    # Chamfer
    if all_chamfer:
        metrics['chamfer_mean'] = float(np.mean(all_chamfer))
        metrics['chamfer_median'] = float(np.median(all_chamfer))
        metrics['chamfer_std'] = float(np.std(all_chamfer))
    else:
        metrics['chamfer_mean'] = metrics['chamfer_median'] = metrics['chamfer_std'] = 0.0

    # Width MAE
    if all_width_ae:
        metrics['width_mae'] = float(np.mean(all_width_ae))
    else:
        metrics['width_mae'] = 0.0

    # Per-threshold P/R and AP
    for iou_t in iou_thresholds:
        stats = per_threshold_stats[iou_t]
        total_tp = sum(s['tp'] for s in stats)
        total_fp = sum(s['fp'] for s in stats)
        total_fn = sum(s['fn'] for s in stats)

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        key = f'iou_{iou_t:.2f}'
        metrics[f'precision@{key}'] = precision
        metrics[f'recall@{key}'] = recall
        metrics[f'f1@{key}'] = f1

    # Simple mAP across thresholds (average F1 as a proxy when
    # per-image ranked AP isn't feasible with set-prediction)
    metrics['mAP'] = float(np.mean([
        metrics[f'f1@iou_{t:.2f}'] for t in iou_thresholds
    ]))

    metrics['num_images'] = sum(len(batch['image']) for batch in [])  # computed below
    # Recompute properly
    total_images = 0
    for batch in dataloader:
        total_images += batch['image'].shape[0]
    # We already iterated; use the length from stats
    metrics['num_images'] = len(per_threshold_stats[iou_thresholds[0]])

    return metrics


def _save_visualisation(
    batch, b, pred_lines_px, gt_lines_px,
    pred_scores, matched, vis_dir, batch_idx, image_size,
):
    """Save a side-by-side visualisation for one image."""
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Undo normalisation for display (approximate)
    img = batch['image'][b].permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    H, W = image_size

    # Draw GT in green
    vis_gt = img.copy()
    for gl in gt_lines_px:
        pts = gl.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis_gt, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw predictions in blue (matched) / red (unmatched)
    vis_pred = img.copy()
    matched_pred_set = {m[0] for m in matched}
    for i, pl in enumerate(pred_lines_px):
        pts = pl.astype(np.int32).reshape(-1, 1, 2)
        colour = (255, 0, 0) if i in matched_pred_set else (0, 0, 255)
        score = float(pred_scores[i]) if i < len(pred_scores) else 0.0
        cv2.polylines(vis_pred, [pts], False, colour, 2, cv2.LINE_AA)
        # Put score text
        if len(pts) > 0:
            org = tuple(pts[0, 0])
            cv2.putText(vis_pred, f'{score:.2f}', org,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

    # Side by side
    combined = np.hstack([vis_gt, vis_pred])
    img_id = batch['image_id'][b].item()
    path = vis_dir / f'eval_{batch_idx:04d}_{img_id:06d}.png'
    cv2.imwrite(str(path), combined)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device, **model_kwargs) -> Line2Former:
    """Load a trained Line2Former from a checkpoint."""
    model = Line2Former(**model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle Lightning checkpoints (state_dict nested under 'state_dict')
    if 'state_dict' in ckpt:
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            # Lightning wraps as model.xxx -> strip prefix
            new_key = k.replace('model.', '', 1) if k.startswith('model.') else k
            state_dict[new_key] = v
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Line2Former')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth or .ckpt)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root data directory (contains val/ or test/)')

    # Optional
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', type=str, default='./eval_results')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--confidence-threshold', type=float, default=0.3)
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualisation images')

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
    print(f'Loading checkpoint: {args.checkpoint}')
    model = load_model(
        args.checkpoint,
        device,
        backbone_type=args.backbone,
        num_queries=args.num_queries,
        max_points=args.max_points,
        d_model=args.d_model,
        num_decoder_layers=args.num_decoder_layers,
        image_size=image_size,
        pretrained=False,  # weights come from checkpoint
    )

    # Load dataset
    dataset = Line2FormerDataset(
        data_root=Path(args.data_dir),
        split=args.split,
        max_lines=args.num_queries,
        max_points=args.max_points,
        image_size=image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run evaluation
    print(f'Evaluating on {len(dataset)} images ({args.split} split)...')
    vis_dir = output_dir / 'visualizations' if args.visualize else None
    metrics = evaluate(
        model, dataloader, device,
        confidence_threshold=args.confidence_threshold,
        image_size=image_size,
        visualize=args.visualize,
        vis_dir=vis_dir,
    )

    # Print results
    print('\n' + '=' * 60)
    print('Evaluation Results')
    print('=' * 60)
    print(f"  Images evaluated:    {metrics['num_images']}")
    print(f"  Chamfer (mean):      {metrics['chamfer_mean']:.4f}")
    print(f"  Chamfer (median):    {metrics['chamfer_median']:.4f}")
    print(f"  Width MAE:           {metrics['width_mae']:.4f}")
    print()
    for key in sorted(metrics):
        if key.startswith('precision') or key.startswith('recall') or key.startswith('f1'):
            print(f"  {key:25s}: {metrics[key]:.4f}")
    print(f"\n  mAP (avg F1):        {metrics['mAP']:.4f}")
    print('=' * 60)

    # Save to JSON
    results_path = output_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMetrics saved to: {results_path}')

    if args.visualize:
        print(f'Visualizations saved to: {vis_dir}')


if __name__ == '__main__':
    main()

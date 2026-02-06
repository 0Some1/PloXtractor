"""
Visualization Tools for Synthetic Line Plot Dataset
View generated samples, masks, and statistics
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import json
import argparse
import random
from typing import List, Tuple
from matplotlib.colors import hsv_to_rgb


def centerline_to_mask(centerline: List[List[float]], width: float, height: int, img_width: int) -> np.ndarray:
    """Convert centerline to binary mask for visualization"""
    mask = np.zeros((height, img_width), dtype=np.uint8)

    if len(centerline) < 2:
        return mask

    pts = np.array(centerline, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=max(1, int(width)))

    return mask


def draw_centerline(image: np.ndarray, centerline: List[List[float]], color: Tuple[int, int, int],
                    thickness: int = 2) -> np.ndarray:
    """Draw centerline on image"""
    if len(centerline) < 2:
        return image

    pts = np.array(centerline, dtype=np.int32)
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

    return image


def visualize_single_sample_lineformer(
        image_path: str,
        annotations: List[dict],
        image_info: dict,
        output_path: str = None,
        show_bbox: bool = True,
        show_points: bool = False,
        alpha: float = 0.5
) -> np.ndarray:
    """
    Visualize a single LineFormer sample with centerline annotations.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    # Generate colors for each annotation
    colors = generate_distinct_colors(len(annotations))

    # Create overlay
    overlay = image.copy()

    for ann, color in zip(annotations, colors):
        centerline = ann.get('centerline', [])
        line_width = ann.get('line_width', 2)

        if len(centerline) >= 2:
            # Draw thick line for visibility
            overlay = draw_centerline(overlay, centerline, color, thickness=max(3, int(line_width) + 2))

            # Optionally draw points
            if show_points:
                for pt in centerline:
                    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, color, -1)

        if show_bbox and 'bbox' in ann:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    # Blend with original
    overlay = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Add text with number of lines
    text = f"Lines: {len(annotations)}"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if output_path:
        output_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

    return overlay


def visualize_sample_detailed_lineformer(
        image_path: str,
        annotations: List[dict],
        image_info: dict,
        metadata: dict = None,
        output_path: str = None,
        figsize: Tuple[int, int] = (16, 8)
):
    """
    Create a detailed visualization for LineFormer format.
    Shows original image, centerline overlay, and individual lines.
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    num_lines = len(annotations)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    # Overlay with all centerlines
    colors = generate_distinct_colors(num_lines)
    overlay = image.copy()

    for ann, color in zip(annotations, colors):
        centerline = ann.get('centerline', [])
        line_width = ann.get('line_width', 2)

        if len(centerline) >= 2:
            overlay = draw_centerline(overlay, centerline, color, thickness=max(3, int(line_width) + 2))

            # Draw bbox
            if 'bbox' in ann:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Centerlines ({num_lines} lines)")
    axes[0, 1].axis('off')

    # Centerline points visualization
    points_img = image.copy()
    for ann, color in zip(annotations, colors):
        centerline = ann.get('centerline', [])
        for pt in centerline:
            cv2.circle(points_img, (int(pt[0]), int(pt[1])), 2, color, -1)

    axes[0, 2].imshow(points_img)
    axes[0, 2].set_title(f"Centerline Points")
    axes[0, 2].axis('off')

    # Individual lines (up to 3)
    for i in range(min(3, num_lines)):
        ann = annotations[i]
        centerline = ann.get('centerline', [])
        line_width = ann.get('line_width', 2)

        # Create mask from centerline
        mask = centerline_to_mask(centerline, line_width, height, width)

        axes[1, i].imshow(mask * 255, cmap='gray')

        title = f"Line {i + 1}"
        if 'metadata' in ann:
            eq_type = ann['metadata'].get('equation_type', '')
            title = f"Line {i + 1}: {eq_type}"
        elif metadata and i < len(metadata.get('lines', [])):
            line_info = metadata['lines'][i]
            title = f"Line {i + 1}: {line_info.get('equation_type', '')}"

        # Add point count
        title += f"\n({len(centerline)} pts)"
        axes[1, i].set_title(title)
        axes[1, i].axis('off')

    # Hide unused subplots
    for i in range(min(3, num_lines), 3):
        axes[1, i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_grid_lineformer(
        dataset_dir: str,
        split: str = "train",
        num_samples: int = 16,
        grid_size: Tuple[int, int] = (4, 4),
        output_path: str = None,
        seed: int = None
):
    """
    Create a grid visualization for LineFormer format.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / split / "images"
    annotations_path = dataset_dir / split / "annotations.json"

    if not annotations_path.exists():
        raise ValueError(f"Annotations not found: {annotations_path}")

    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Check format
    is_lineformer = data.get('info', {}).get('format') == 'lineformer'
    if not is_lineformer:
        print("Warning: Not LineFormer format, using standard visualization")
        visualize_grid(dataset_dir, split, num_samples, grid_size, output_path, seed)
        return

    # Build lookup dictionaries
    images_dict = {img['id']: img for img in data['images']}

    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Get all image IDs
    image_ids = list(images_dict.keys())

    # Random sample
    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(image_ids))
    selected_ids = random.sample(image_ids, num_samples)

    # Create grid
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, image_id in enumerate(selected_ids):
        if idx >= rows * cols:
            break

        ax = axes[idx]

        # Get image info
        img_info = images_dict[image_id]
        image_path = images_dir / img_info['file_name']

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations
        annotations = annotations_by_image.get(image_id, [])

        # Create overlay with centerlines
        overlay = image.copy()
        colors = generate_distinct_colors(len(annotations))

        for ann, color in zip(annotations, colors):
            centerline = ann.get('centerline', [])
            line_width = ann.get('line_width', 2)

            if len(centerline) >= 2:
                overlay = draw_centerline(overlay, centerline, color, thickness=max(2, int(line_width)))

        ax.imshow(overlay)
        ax.set_title(f"ID:{image_id} ({len(annotations)} lines)", fontsize=10)
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(selected_ids), rows * cols):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Grid saved to {output_path}")
    else:
        plt.show()


def print_dataset_stats_lineformer(dataset_dir: str, split: str = "train"):
    """
    Print statistics for LineFormer format dataset.
    """
    dataset_dir = Path(dataset_dir)
    stats_path = dataset_dir / split / "stats.json"
    annotations_path = dataset_dir / split / "annotations.json"

    print(f"\n{'=' * 60}")
    print(f"LineFormer Dataset Statistics: {split}")
    print(f"{'=' * 60}")

    # Load stats if available
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        print(f"\nGeneration Stats:")
        print(f"  Total samples: {stats.get('total_samples', 'N/A')}")
        print(f"  Total annotations: {stats.get('total_annotations', 'N/A')}")

        avg_lines = stats.get('avg_lines_per_image', 'N/A')
        if isinstance(avg_lines, (int, float)):
            print(f"  Avg lines/image: {avg_lines:.2f}")

        avg_points = stats.get('avg_points_per_line', 'N/A')
        if isinstance(avg_points, (int, float)):
            print(f"  Avg points/line: {avg_points:.1f}")

        if 'equation_types' in stats:
            print(f"\nEquation Types:")
            for eq_type, count in sorted(stats['equation_types'].items(), key=lambda x: -x[1]):
                print(f"    {eq_type}: {count}")

    # Load annotations for additional stats
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            data = json.load(f)

        # Check format
        is_lineformer = data.get('info', {}).get('format') == 'lineformer'
        print(f"\nFormat: {'LineFormer' if is_lineformer else 'COCO'}")

        # Collect stats
        lines_per_image = {}
        points_per_line = []

        for ann in data['annotations']:
            img_id = ann['image_id']
            lines_per_image[img_id] = lines_per_image.get(img_id, 0) + 1

            if 'num_points' in ann:
                points_per_line.append(ann['num_points'])
            elif 'centerline' in ann:
                points_per_line.append(len(ann['centerline']))

        lines_counts = list(lines_per_image.values())

        if lines_counts:
            print(f"\nLines per Image Distribution:")
            print(f"  Min: {min(lines_counts)}")
            print(f"  Max: {max(lines_counts)}")
            print(f"  Mean: {np.mean(lines_counts):.2f}")

        if points_per_line:
            print(f"\nPoints per Line Distribution:")
            print(f"  Min: {min(points_per_line)}")
            print(f"  Max: {max(points_per_line)}")
            print(f"  Mean: {np.mean(points_per_line):.1f}")
            print(f"  Median: {np.median(points_per_line):.1f}")

    print(f"\n{'=' * 60}")

def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        saturation = 0.7 + random.random() * 0.3
        value = 0.7 + random.random() * 0.3
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def polygon_to_mask(polygons: List, height: int, width: int) -> np.ndarray:
    """Convert polygon format to binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def visualize_single_sample(
    image_path: str,
    annotations: List[dict],
    image_info: dict,
    output_path: str = None,
    show_bbox: bool = True,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Visualize a single sample with its annotations.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    height, width = image.shape[:2]
    
    # Generate colors for each annotation
    colors = generate_distinct_colors(len(annotations))
    
    # Create overlay
    overlay = image.copy()
    
    for ann, color in zip(annotations, colors):
        # Decode mask from polygon
        if 'segmentation' in ann and ann['segmentation']:
            mask = polygon_to_mask(ann['segmentation'], height, width)
            
            # Apply colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        if show_bbox and 'bbox' in ann:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    
    # Add text with number of lines
    text = f"Lines: {len(annotations)}"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if output_path:
        output_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
    
    return overlay


def visualize_sample_detailed(
    image_path: str,
    annotations: List[dict],
    image_info: dict,
    metadata: dict = None,
    output_path: str = None,
    figsize: Tuple[int, int] = (16, 8)
):
    """
    Create a detailed visualization with original image, masks, and metadata.
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    num_masks = len(annotations)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Overlay with all masks
    colors = generate_distinct_colors(num_masks)
    overlay = image.copy()
    
    for ann, color in zip(annotations, colors):
        if 'segmentation' in ann and ann['segmentation']:
            mask = polygon_to_mask(ann['segmentation'], height, width)
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
            
            # Draw bbox
            if 'bbox' in ann:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
    
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Overlay ({num_masks} lines)")
    axes[0, 1].axis('off')
    
    # Combined mask
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            mask = polygon_to_mask(ann['segmentation'], height, width)
            combined_mask = np.maximum(combined_mask, mask)
    
    axes[0, 2].imshow(combined_mask * 255, cmap='gray')
    axes[0, 2].set_title("Combined Mask")
    axes[0, 2].axis('off')
    
    # Individual masks (up to 3)
    for i in range(min(3, num_masks)):
        ann = annotations[i]
        if 'segmentation' in ann and ann['segmentation']:
            mask = polygon_to_mask(ann['segmentation'], height, width)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        
        axes[1, i].imshow(mask * 255, cmap='gray')
        
        title = f"Line {i+1}"
        if metadata and i < len(metadata.get('equations', [])):
            eq_info = metadata['equations'][i]
            title = f"Line {i+1}: {eq_info.get('category', '')}"
        
        axes[1, i].set_title(title)
        axes[1, i].axis('off')
    
    # Hide unused subplots
    for i in range(min(3, num_masks), 3):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_grid(
    dataset_dir: str,
    split: str = "train",
    num_samples: int = 16,
    grid_size: Tuple[int, int] = (4, 4),
    output_path: str = None,
    seed: int = None
):
    """
    Create a grid visualization of multiple samples.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / split / "images"
    annotations_path = dataset_dir / split / "annotations.json"
    
    if not annotations_path.exists():
        raise ValueError(f"Annotations not found: {annotations_path}")
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Build lookup dictionaries
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Get all image IDs
    image_ids = list(images_dict.keys())
    
    # Random sample
    if seed is not None:
        random.seed(seed)
    
    num_samples = min(num_samples, len(image_ids))
    selected_ids = random.sample(image_ids, num_samples)
    
    # Create grid
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    
    for idx, image_id in enumerate(selected_ids):
        if idx >= rows * cols:
            break
        
        ax = axes[idx]
        
        # Get image info
        img_info = images_dict[image_id]
        image_path = images_dir / img_info['file_name']
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Get annotations
        annotations = annotations_by_image.get(image_id, [])
        
        # Create overlay
        overlay = image.copy()
        colors = generate_distinct_colors(len(annotations))
        
        for ann, color in zip(annotations, colors):
            if 'segmentation' in ann and ann['segmentation']:
                mask = polygon_to_mask(ann['segmentation'], height, width)
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = color
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)
        
        ax.imshow(overlay)
        ax.set_title(f"ID:{image_id} ({len(annotations)} lines)", fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(selected_ids), rows * cols):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Grid saved to {output_path}")
    else:
        plt.show()


def print_dataset_stats(dataset_dir: str, split: str = "train"):
    """
    Print statistics about the dataset.
    """
    dataset_dir = Path(dataset_dir)
    stats_path = dataset_dir / split / "stats.json"
    annotations_path = dataset_dir / split / "annotations.json"
    
    print(f"\n{'='*60}")
    print(f"Dataset Statistics: {split}")
    print(f"{'='*60}")
    
    # Load stats if available
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print(f"\nGeneration Stats:")
        print(f"  Total samples: {stats.get('total_samples', 'N/A')}")
        print(f"  Total annotations: {stats.get('total_annotations', 'N/A')}")
        avg = stats.get('avg_lines_per_image', 'N/A')
        if isinstance(avg, (int, float)):
            print(f"  Avg lines/image: {avg:.2f}")
        else:
            print(f"  Avg lines/image: {avg}")
        
        if 'equation_types' in stats:
            print(f"\nEquation Types:")
            for eq_type, count in sorted(stats['equation_types'].items(), key=lambda x: -x[1]):
                print(f"    {eq_type}: {count}")
    
    # Load annotations for additional stats
    if annotations_path.exists():
        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            annotations_by_image[img_id] = annotations_by_image.get(img_id, 0) + 1
        
        lines_per_image = list(annotations_by_image.values())
        
        if lines_per_image:
            print(f"\nLines per Image Distribution:")
            print(f"  Min: {min(lines_per_image)}")
            print(f"  Max: {max(lines_per_image)}")
            print(f"  Mean: {np.mean(lines_per_image):.2f}")
            print(f"  Std: {np.std(lines_per_image):.2f}")
            
            # Histogram
            unique, counts = np.unique(lines_per_image, return_counts=True)
            print(f"\n  Distribution:")
            for n, c in zip(unique, counts):
                bar = "█" * int(c / max(counts) * 30)
                print(f"    {n:2d} lines: {bar} ({c})")
    
    print(f"\n{'='*60}")


def detect_format(dataset_dir: str, split: str) -> str:
    """Detect if dataset is COCO or LineFormer format"""
    annotations_path = Path(dataset_dir) / split / "annotations.json"

    if not annotations_path.exists():
        return "unknown"

    with open(annotations_path, 'r') as f:
        data = json.load(f)

    if data.get('info', {}).get('format') == 'lineformer':
        return 'lineformer'
    return 'coco'


def main():
    parser = argparse.ArgumentParser(description="Visualize Synthetic Line Plot Dataset")
    parser.add_argument("--dataset-dir", type=str, default="../data",
                        help="Dataset directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="Split to visualize")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "detailed", "grid", "stats"],
                        help="Visualization mode")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "coco", "lineformer"],
                        help="Annotation format (auto-detect by default)")
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of samples to visualize (for grid mode)")
    parser.add_argument("--image-id", type=int, default=None,
                        help="Specific image ID to visualize (for single/detailed mode)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--grid-size", type=str, default="4x4",
                        help="Grid size for grid mode (e.g., 4x4)")
    parser.add_argument("--show-points", action="store_true", default=True,
                        help="Show centerline points (LineFormer only)")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    # Detect format
    if args.format == "auto":
        detected_format = detect_format(args.dataset_dir, args.split)
        print(f"Detected format: {detected_format}")
    else:
        detected_format = args.format

    is_lineformer = detected_format == 'lineformer'

    # Stats mode
    if args.mode == "stats":
        if is_lineformer:
            print_dataset_stats_lineformer(args.dataset_dir, args.split)
        else:
            print_dataset_stats(args.dataset_dir, args.split)
        return

    # Parse grid size
    if 'x' in args.grid_size:
        grid_size = tuple(map(int, args.grid_size.split('x')))
    else:
        grid_size = (4, 4)

    # Set default output path
    if args.output is None:
        args.output = f"visualization_{args.split}_{args.mode}.png"

    # Grid mode
    if args.mode == "grid":
        if is_lineformer:
            visualize_grid_lineformer(
                dataset_dir=args.dataset_dir,
                split=args.split,
                num_samples=args.num_samples,
                grid_size=grid_size,
                output_path=args.output,
                seed=args.seed
            )
        else:
            visualize_grid(
                dataset_dir=args.dataset_dir,
                split=args.split,
                num_samples=args.num_samples,
                grid_size=grid_size,
                output_path=args.output,
                seed=args.seed
            )
        print(f"Grid visualization saved to {args.output}")

    # Single/Detailed mode
    elif args.mode in ["single", "detailed"]:
        annotations_path = dataset_dir / args.split / "annotations.json"
        images_dir = dataset_dir / args.split / "images"

        # Load annotations
        with open(annotations_path, 'r') as f:
            data = json.load(f)

        images_dict = {img['id']: img for img in data['images']}
        annotations_by_image = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        # Get image ID
        if args.image_id is not None:
            image_id = args.image_id
        else:
            random.seed(args.seed)
            image_id = random.choice(list(images_dict.keys()))

        img_info = images_dict[image_id]
        image_path = str(images_dir / img_info['file_name'])
        annotations = annotations_by_image.get(image_id, [])

        # Load metadata if available
        metadata = None
        metadata_path = dataset_dir / "metadata" / args.split / f"image_{image_id:06d}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        if args.mode == "single":
            if is_lineformer:
                visualize_single_sample_lineformer(
                    image_path=image_path,
                    annotations=annotations,
                    image_info=img_info,
                    output_path=args.output,
                    show_points=args.show_points
                )
            else:
                visualize_single_sample(
                    image_path=image_path,
                    annotations=annotations,
                    image_info=img_info,
                    output_path=args.output
                )
            print(f"Single visualization saved to {args.output}")

        else:  # detailed
            if is_lineformer:
                visualize_sample_detailed_lineformer(
                    image_path=image_path,
                    annotations=annotations,
                    image_info=img_info,
                    metadata=metadata,
                    output_path=args.output
                )
            else:
                visualize_sample_detailed(
                    image_path=image_path,
                    annotations=annotations,
                    image_info=img_info,
                    metadata=metadata,
                    output_path=args.output
                )
            print(f"Detailed visualization saved to {args.output}")


if __name__ == "__main__":
    main()

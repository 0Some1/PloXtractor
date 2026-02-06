"""
Utilities for LineFormer annotation format.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional


def build_lineformer_annotations(
        samples: List[Dict[str, Any]],
        output_path: str,
        split: str = "train"
) -> Dict[str, Any]:
    """
    Build LineFormer format annotations from samples.

    Args:
        samples: List of sample dictionaries from generate_sample()
        output_path: Path to save annotations JSON
        split: Dataset split name

    Returns:
        Annotations dictionary
    """
    annotations = {
        "info": {
            "description": f"Synthetic Line Plot Dataset - {split}",
            "version": "2.0",
            "format": "lineformer",
        },
        "categories": [
            {"id": 1, "name": "line", "supercategory": "chart_element"}
        ],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1

    for sample in samples:
        image_id = sample["sample_id"]

        # Add image entry
        annotations["images"].append({
            "id": image_id,
            "file_name": sample["image_filename"],
            "width": sample["image_size"]["width"],
            "height": sample["image_size"]["height"],
        })

        # Add line annotations
        for line in sample["lines"]:
            centerline = line["centerline"]

            # Compute bounding box from centerline
            if len(centerline) > 0:
                xs = [p[0] for p in centerline]
                ys = [p[1] for p in centerline]
                bbox = [
                    min(xs), min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys)
                ]
            else:
                bbox = [0, 0, 0, 0]

            annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "centerline": centerline,
                "num_points": line["num_points"],
                "line_width": line["line_width_pixels"],
                "bbox": bbox,
                "data_coordinates": {
                    "x": line["data_x"],
                    "y": line["data_y"],
                },
                "metadata": {
                    "equation_type": line["equation_type"],
                    "equation_name": line["equation_name"],
                    "color": line["color"],
                    "linestyle": line["linestyle"],
                }
            })

            annotation_id += 1

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Saved {len(annotations['annotations'])} annotations to {output_path}")

    return annotations


def normalize_centerline(
        centerline: List[List[float]],
        image_width: int,
        image_height: int
) -> List[List[float]]:
    """
    Normalize centerline coordinates to [0, 1] range.

    Args:
        centerline: List of [x, y] pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Normalized centerline
    """
    return [
        [p[0] / image_width, p[1] / image_height]
        for p in centerline
    ]


def resample_centerline(
        centerline: List[List[float]],
        num_points: int = 50
) -> np.ndarray:
    """
    Resample centerline to fixed number of evenly-spaced points.

    This is important for batching - all lines need same number of points.

    Args:
        centerline: List of [x, y] coordinates
        num_points: Target number of points

    Returns:
        Resampled centerline as (num_points, 2) array
    """
    if len(centerline) < 2:
        # Return zeros if line is too short
        return np.zeros((num_points, 2))

    centerline = np.array(centerline)

    # Compute cumulative arc length
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        return np.tile(centerline[0], (num_points, 1))

    # Target arc lengths for resampled points
    target_lengths = np.linspace(0, total_length, num_points)

    # Interpolate x and y separately
    resampled = np.zeros((num_points, 2))
    resampled[:, 0] = np.interp(target_lengths, cumulative_length, centerline[:, 0])
    resampled[:, 1] = np.interp(target_lengths, cumulative_length, centerline[:, 1])

    return resampled


def centerline_to_mask(
        centerline: List[List[float]],
        width: float,
        image_size: tuple,
        aa_scale: int = 2
) -> np.ndarray:
    """
    Render a centerline to a binary mask (for verification/comparison).

    Args:
        centerline: List of [x, y] pixel coordinates
        width: Line width in pixels
        image_size: (width, height)
        aa_scale: Anti-aliasing scale factor

    Returns:
        Binary mask (H, W)
    """
    import cv2

    img_w, img_h = image_size

    # Create larger image for anti-aliasing
    canvas = np.zeros((img_h * aa_scale, img_w * aa_scale), dtype=np.uint8)

    # Scale centerline
    scaled_centerline = np.array(centerline) * aa_scale
    scaled_width = int(width * aa_scale)

    # Draw polyline
    pts = scaled_centerline.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=scaled_width)

    # Downsample with anti-aliasing
    mask = cv2.resize(canvas, (img_w, img_h), interpolation=cv2.INTER_AREA)

    # Threshold
    mask = (mask > 127).astype(np.uint8) * 255

    return mask


def compute_line_statistics(annotations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute statistics about the line annotations.

    Args:
        annotations: LineFormer format annotations

    Returns:
        Statistics dictionary
    """
    num_points_list = []
    line_lengths_list = []
    widths_list = []

    for ann in annotations["annotations"]:
        centerline = np.array(ann["centerline"])

        num_points_list.append(ann["num_points"])
        widths_list.append(ann["line_width"])

        # Compute line length
        if len(centerline) > 1:
            diffs = np.diff(centerline, axis=0)
            length = np.sqrt((diffs ** 2).sum(axis=1)).sum()
            line_lengths_list.append(length)

    return {
        "num_annotations": len(annotations["annotations"]),
        "num_images": len(annotations["images"]),
        "points_per_line": {
            "min": min(num_points_list),
            "max": max(num_points_list),
            "mean": np.mean(num_points_list),
            "median": np.median(num_points_list),
        },
        "line_length_pixels": {
            "min": min(line_lengths_list),
            "max": max(line_lengths_list),
            "mean": np.mean(line_lengths_list),
            "median": np.median(line_lengths_list),
        },
        "line_width_pixels": {
            "min": min(widths_list),
            "max": max(widths_list),
            "mean": np.mean(widths_list),
        },
    }
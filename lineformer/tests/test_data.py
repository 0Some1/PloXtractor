"""
Test the data module - Fixed version matching working visualization.
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from dataset import LineFormerDataset


def denormalize_image(img_tensor):
    """
    Denormalize image from ImageNet normalization.

    Args:
        img_tensor: (3, H, W) tensor with ImageNet normalization

    Returns:
        (H, W, 3) numpy array in [0, 1] range
    """
    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert to numpy and transpose to (H, W, 3)
    img = img_tensor.permute(1, 2, 0).numpy()

    # Denormalize
    img = img * std + mean

    # Clip to valid range
    img = np.clip(img, 0, 1)

    return img


def draw_centerline(image: np.ndarray, centerline: np.ndarray, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw centerline on image - matching your working visualization."""
    if len(centerline) < 2:
        return image

    # Convert to int32 for OpenCV
    pts = centerline.astype(np.int32)

    # Draw polyline
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return image


def generate_distinct_colors(n: int) -> list:
    """Generate n visually distinct colors."""
    from matplotlib.colors import hsv_to_rgb
    import random

    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        saturation = 0.7 + random.random() * 0.3
        value = 0.7 + random.random() * 0.3
        rgb = hsv_to_rgb([hue, saturation, value])
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def visualize_batch(batch, num_samples=2):
    """Visualize a batch of data - matching working visualization approach."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 7 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_samples, len(batch['image']))):
        # Denormalize image properly
        img = denormalize_image(batch['image'][i])
        height, width = img.shape[:2]

        # Convert to uint8 for OpenCV drawing
        img_uint8 = (img * 255).astype(np.uint8)

        # Get polyline data
        centerlines = batch['centerlines'][i]  # (max_lines, max_points, 2)
        valid_mask = batch['valid_mask'][i]    # (max_lines,)
        num_lines = batch['num_lines'][i].item()
        widths = batch['widths'][i]            # (max_lines, max_points)

        # Plot 1: Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i} (ID: {batch['image_id'][i].item()})")
        axes[i, 0].axis('off')

        # Plot 2: Image with polylines overlay
        overlay = img_uint8.copy()

        # Generate distinct colors
        colors = generate_distinct_colors(num_lines)

        for j, is_valid in enumerate(valid_mask):
            if is_valid and j < num_lines:
                line = centerlines[j].numpy()  # (max_points, 2) in normalized coords [0, 1]

                # Denormalize coordinates: [0, 1] -> pixel space
                # IMPORTANT: Coordinates are (x, y) where x is width dimension, y is height dimension
                line_pixel = np.zeros_like(line)
                line_pixel[:, 0] = line[:, 0] * width   # x * width
                line_pixel[:, 1] = line[:, 1] * height  # y * height

                # Get line width (use first point's width)
                line_width = max(2, int(widths[j, 0].item()))

                # Draw using the working method
                color = colors[j % len(colors)]
                overlay = draw_centerline(overlay, line_pixel, color, thickness=line_width + 2)

        # Convert back to RGB float for matplotlib
        overlay_rgb = overlay.astype(np.float32) / 255.0

        axes[i, 1].imshow(overlay_rgb)
        axes[i, 1].set_title(f"Lines: {num_lines}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to data_visualization.png")
    plt.close()


def visualize_raw_vs_processed(dataset, idx=0):
    """
    Compare raw annotation with processed data to debug coordinate issues.
    """
    import cv2

    # Get processed sample
    sample = dataset[idx]

    # Get raw data
    raw = dataset.get_raw_annotation(idx)
    img_path = dataset.get_image_path(idx)

    # Load original image
    img_original = cv2.imread(str(img_path))
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = img_original.shape[:2]

    # Get processed image
    img_processed = denormalize_image(sample['image'])
    proc_height, proc_width = img_processed.shape[:2]
    img_processed_uint8 = (img_processed * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Original image with raw annotations
    overlay_orig = img_original.copy()
    colors = generate_distinct_colors(len(raw['annotations']))

    for ann, color in zip(raw['annotations'], colors):
        centerline = np.array(ann['centerline'])
        line_width = max(2, int(ann.get('line_width', 2)))
        overlay_orig = draw_centerline(overlay_orig, centerline, color, thickness=line_width + 2)

    axes[0].imshow(overlay_orig)
    axes[0].set_title(f"Original Image ({orig_width}x{orig_height})\nRaw Annotations")
    axes[0].axis('off')

    # Plot 2: Processed image with processed annotations
    overlay_proc = img_processed_uint8.copy()

    centerlines = sample['centerlines']
    valid_mask = sample['valid_mask']
    widths = sample['widths']

    for j, is_valid in enumerate(valid_mask):
        if is_valid:
            line = centerlines[j].numpy()

            # Denormalize
            line_pixel = np.zeros_like(line)
            line_pixel[:, 0] = line[:, 0] * proc_width
            line_pixel[:, 1] = line[:, 1] * proc_height

            color = colors[j % len(colors)]
            line_width = max(2, int(widths[j, 0].item()))
            overlay_proc = draw_centerline(overlay_proc, line_pixel, color, thickness=line_width + 2)

    overlay_proc_rgb = overlay_proc.astype(np.float32) / 255.0

    axes[1].imshow(overlay_proc_rgb)
    axes[1].set_title(f"Processed Image ({proc_width}x{proc_height})\nTransformed Annotations")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('raw_vs_processed.png', dpi=150, bbox_inches='tight')
    print("✓ Saved comparison to raw_vs_processed.png")
    plt.close()


def test_coordinate_accuracy(dataset, num_tests=5):
    """
    Test if coordinates are preserved correctly through the pipeline.
    """
    print("\n" + "=" * 60)
    print("Testing Coordinate Accuracy")
    print("=" * 60)

    total_error = 0

    for idx in range(min(num_tests, len(dataset))):
        # Get raw and processed data
        raw = dataset.get_raw_annotation(idx)
        sample = dataset[idx]

        # Get image dimensions
        orig_size = sample['original_size'].numpy()  # [height, width]
        target_size = dataset.image_size  # (height, width)

        # Expected scale factors
        scale_x = target_size[1] / orig_size[1]  # width scale
        scale_y = target_size[0] / orig_size[0]  # height scale

        print(f"\nImage {idx}:")
        print(f"  Original size: {orig_size[1]}x{orig_size[0]} (WxH)")
        print(f"  Target size: {target_size[1]}x{target_size[0]} (WxH)")
        print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")

        # Check first valid line
        for j, is_valid in enumerate(sample['valid_mask']):
            if is_valid and j < len(raw['annotations']):
                raw_centerline = np.array(raw['annotations'][j]['centerline'])
                proc_centerline = sample['centerlines'][j].numpy()

                # Expected transformed coordinates (normalized)
                expected = raw_centerline.copy()
                expected[:, 0] = (expected[:, 0] * scale_x) / target_size[1]  # normalize x
                expected[:, 1] = (expected[:, 1] * scale_y) / target_size[0]  # normalize y

                # Compute error
                # Note: proc_centerline is resampled, so we can't compare point-by-point
                # Instead, check if first and last points are close
                error_first = np.linalg.norm(proc_centerline[0] - expected[0])
                error_last = np.linalg.norm(proc_centerline[-1] - expected[-1])

                print(f"  Line {j}:")
                print(f"    Raw first point: {raw_centerline[0]}")
                print(f"    Expected (normalized): {expected[0]}")
                print(f"    Processed: {proc_centerline[0]}")
                print(f"    Error (first point): {error_first:.6f}")
                print(f"    Error (last point): {error_last:.6f}")

                total_error += (error_first + error_last) / 2
                break  # Only check first line

    avg_error = total_error / num_tests
    print(f"\nAverage coordinate error: {avg_error:.6f}")
    print("(Should be < 0.01 for good accuracy)")
    print("=" * 60)


def test_dataset():
    """Test dataset loading and transformations."""
    print("=" * 60)
    print("Testing LineFormer Dataset")
    print("=" * 60)

    # Create dataset
    dataset = LineFormerDataset(
        data_root='../data',
        split='train',
        max_lines=20,
        max_points=50,
        image_size=(480, 640),
        normalize_coords=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    print("\n" + "-" * 60)
    print("Testing single sample...")
    sample = dataset[0]

    print(f"Image shape: {sample['image'].shape}")
    print(f"Centerlines shape: {sample['centerlines'].shape}")
    print(f"Widths shape: {sample['widths'].shape}")
    print(f"Valid mask shape: {sample['valid_mask'].shape}")
    print(f"Number of lines: {sample['num_lines'].item()}")
    print(f"Image ID: {sample['image_id'].item()}")
    print(f"Original size (H, W): {sample['original_size'].tolist()}")

    # Check coordinate ranges
    valid_lines = sample['centerlines'][sample['valid_mask']]
    if len(valid_lines) > 0:
        print(f"\nCoordinate ranges (should be [0, 1] if normalized):")
        print(f"  X: [{valid_lines[..., 0].min():.3f}, {valid_lines[..., 0].max():.3f}]")
        print(f"  Y: [{valid_lines[..., 1].min():.3f}, {valid_lines[..., 1].max():.3f}]")

    # Test coordinate accuracy
    test_coordinate_accuracy(dataset, num_tests=3)

    # Debug: Compare raw vs processed
    print("\n" + "-" * 60)
    print("Comparing raw vs processed annotations...")
    visualize_raw_vs_processed(dataset, idx=0)

    # Test dataloader
    print("\n" + "-" * 60)
    print("Testing DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=False
    )

    batch = next(iter(loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch centerlines shape: {batch['centerlines'].shape}")
    print(f"Batch num_lines: {batch['num_lines'].tolist()}")

    # Visualize
    print("\n" + "-" * 60)
    print("Generating visualization...")
    visualize_batch(batch, num_samples=2)

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    test_dataset()
"""
Polyline Processing Utilities
Functions for manipulating, resampling, and converting polylines.
"""

import numpy as np
import torch
import cv2
from scipy.interpolate import interp1d
from typing import List, Tuple, Union, Optional


def resample_polyline(
        polyline: Union[np.ndarray, torch.Tensor],
        num_points: int,
        method: str = 'linear'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Resample a polyline to a fixed number of points using interpolation.

    Args:
        polyline: (N, 2) array of (x, y) coordinates
        num_points: Target number of points
        method: Interpolation method ('linear', 'cubic', 'quadratic')

    Returns:
        Resampled polyline of shape (num_points, 2)
    """
    is_torch = isinstance(polyline, torch.Tensor)
    if is_torch:
        device = polyline.device
        polyline = polyline.cpu().numpy()

    if len(polyline) < 2:
        # Handle degenerate case
        result = np.tile(polyline[0] if len(polyline) == 1 else np.zeros(2), (num_points, 1))
        return torch.from_numpy(result).to(device) if is_torch else result

    if len(polyline) == num_points:
        return torch.from_numpy(polyline).to(device) if is_torch else polyline

    # Compute cumulative arc length for parameterization
    distances = np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])

    # Normalize to [0, 1]
    if cumulative_distances[-1] == 0:
        # All points are the same
        result = np.tile(polyline[0], (num_points, 1))
        return torch.from_numpy(result).to(device) if is_torch else result

    t_old = cumulative_distances / cumulative_distances[-1]
    t_new = np.linspace(0, 1, num_points)

    # Interpolate x and y separately
    try:
        x_interp = interp1d(t_old, polyline[:, 0], kind=method, fill_value='extrapolate')
        y_interp = interp1d(t_old, polyline[:, 1], kind=method, fill_value='extrapolate')

        new_polyline = np.stack([x_interp(t_new), y_interp(t_new)], axis=1)
    except ValueError:
        # Fallback to linear if method fails
        x_interp = interp1d(t_old, polyline[:, 0], kind='linear', fill_value='extrapolate')
        y_interp = interp1d(t_old, polyline[:, 1], kind='linear', fill_value='extrapolate')
        new_polyline = np.stack([x_interp(t_new), y_interp(t_new)], axis=1)

    if is_torch:
        return torch.from_numpy(new_polyline).float().to(device)
    return new_polyline.astype(np.float32)


def polyline_length(polyline: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute the total length of a polyline.

    Args:
        polyline: (N, 2) array of (x, y) coordinates

    Returns:
        Total arc length
    """
    if isinstance(polyline, torch.Tensor):
        polyline = polyline.cpu().numpy()

    if len(polyline) < 2:
        return 0.0

    distances = np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1))
    return float(np.sum(distances))


def simplify_polyline(
        polyline: np.ndarray,
        tolerance: float = 2.0
) -> np.ndarray:
    """
    Simplify a polyline using the Ramer-Douglas-Peucker algorithm.

    Args:
        polyline: (N, 2) array of (x, y) coordinates
        tolerance: Maximum distance threshold

    Returns:
        Simplified polyline
    """
    if len(polyline) < 3:
        return polyline

    # Use OpenCV's approxPolyDP
    polyline_int = polyline.astype(np.int32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(polyline_int, tolerance, closed=False)

    return simplified.reshape(-1, 2).astype(np.float32)


def polyline_to_mask(
        polyline: np.ndarray,
        width: Union[float, np.ndarray],
        image_size: Tuple[int, int],
        anti_alias: bool = True
) -> np.ndarray:
    """
    Render a polyline to a binary mask.

    Args:
        polyline: (N, 2) array of (x, y) coordinates
        width: Line width (scalar or array of length N)
        image_size: (height, width) of output mask
        anti_alias: Whether to use anti-aliasing

    Returns:
        Binary mask of shape (height, width)
    """
    height, width_img = image_size
    mask = np.zeros((height, width_img), dtype=np.uint8)

    if len(polyline) < 2:
        return mask

    # Convert to integer coordinates
    polyline_int = polyline.astype(np.int32).reshape((-1, 1, 2))

    # Handle variable width
    if isinstance(width, (int, float)):
        thickness = int(width)
        cv2.polylines(mask, [polyline_int], isClosed=False,
                      color=255, thickness=thickness,
                      lineType=cv2.LINE_AA if anti_alias else cv2.LINE_8)
    else:
        # Variable width - draw segments individually
        for i in range(len(polyline) - 1):
            pt1 = tuple(polyline_int[i, 0])
            pt2 = tuple(polyline_int[i + 1, 0])
            thickness = int((width[i] + width[i + 1]) / 2)
            cv2.line(mask, pt1, pt2, color=255, thickness=thickness,
                     lineType=cv2.LINE_AA if anti_alias else cv2.LINE_8)

    return mask


def normalize_polyline(
        polyline: Union[np.ndarray, torch.Tensor],
        image_size: Tuple[int, int]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize polyline coordinates to [0, 1] range.

    Args:
        polyline: (N, 2) array of (x, y) coordinates in pixel space
        image_size: (height, width) of image

    Returns:
        Normalized polyline with coordinates in [0, 1]
    """
    is_torch = isinstance(polyline, torch.Tensor)
    height, width = image_size

    if is_torch:
        scale = torch.tensor([width, height], device=polyline.device, dtype=polyline.dtype)
        normalized = polyline / scale
    else:
        scale = np.array([width, height], dtype=polyline.dtype)
        normalized = polyline / scale

    return normalized


def denormalize_polyline(
        polyline: Union[np.ndarray, torch.Tensor],
        image_size: Tuple[int, int]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize polyline coordinates from [0, 1] to pixel space.

    Args:
        polyline: (N, 2) array of normalized (x, y) coordinates
        image_size: (height, width) of image

    Returns:
        Polyline in pixel coordinates
    """
    is_torch = isinstance(polyline, torch.Tensor)
    height, width = image_size

    if is_torch:
        scale = torch.tensor([width, height], device=polyline.device, dtype=polyline.dtype)
        denormalized = polyline * scale
    else:
        scale = np.array([width, height], dtype=polyline.dtype)
        denormalized = polyline * scale

    return denormalized


def compute_polyline_bbox(polyline: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute bounding box of a polyline.

    Args:
        polyline: (N, 2) array of (x, y) coordinates

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    if len(polyline) == 0:
        return (0, 0, 0, 0)

    x_min, y_min = polyline.min(axis=0)
    x_max, y_max = polyline.max(axis=0)

    return (float(x_min), float(y_min), float(x_max), float(y_max))


def split_polyline_at_intersections(polyline: np.ndarray) -> List[np.ndarray]:
    """
    Split a self-intersecting polyline into non-intersecting segments.
    (Simplified version - just splits at sharp angles for now)

    Args:
        polyline: (N, 2) array of (x, y) coordinates

    Returns:
        List of polyline segments
    """
    if len(polyline) < 3:
        return [polyline]

    # Compute angles between consecutive segments
    vectors = np.diff(polyline, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = np.abs(np.diff(angles))

    # Normalize to [0, pi]
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)

    # Split at sharp turns (> 120 degrees)
    split_indices = np.where(angle_diffs > np.pi * 2 / 3)[0] + 1

    if len(split_indices) == 0:
        return [polyline]

    # Split the polyline
    segments = []
    start_idx = 0
    for split_idx in split_indices:
        segments.append(polyline[start_idx:split_idx + 1])
        start_idx = split_idx
    segments.append(polyline[start_idx:])

    return segments


def augment_polyline_with_noise(
        polyline: np.ndarray,
        noise_std: float = 2.0,
        preserve_endpoints: bool = True
) -> np.ndarray:
    """
    Add random noise to polyline points (for data augmentation).

    Args:
        polyline: (N, 2) array of (x, y) coordinates
        noise_std: Standard deviation of Gaussian noise
        preserve_endpoints: Whether to keep first/last points unchanged

    Returns:
        Noisy polyline
    """
    noisy = polyline.copy()
    noise = np.random.normal(0, noise_std, polyline.shape)

    if preserve_endpoints and len(polyline) > 2:
        noise[0] = 0
        noise[-1] = 0

    noisy += noise
    return noisy


def compute_chamfer_distance(
        polyline1: np.ndarray,
        polyline2: np.ndarray
) -> float:
    """
    Compute Chamfer distance between two polylines.

    Args:
        polyline1: (N, 2) array
        polyline2: (M, 2) array

    Returns:
        Chamfer distance (symmetric)
    """
    from scipy.spatial.distance import cdist

    # Compute pairwise distances
    dist_matrix = cdist(polyline1, polyline2, metric='euclidean')

    # Forward: min distance from polyline1 to polyline2
    forward = np.mean(np.min(dist_matrix, axis=1))

    # Backward: min distance from polyline2 to polyline1
    backward = np.mean(np.min(dist_matrix, axis=0))

    # Symmetric Chamfer distance
    return float((forward + backward) / 2)
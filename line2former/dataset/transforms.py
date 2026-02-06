"""
Data Augmentation for Line2Former
Simplified version with manual transform handling.
"""

import numpy as np
import cv2
import albumentations as A
from typing import Dict, List, Tuple


class Line2FormerTransform:
    """
    Manual transform pipeline for images and polylines.
    More reliable than trying to integrate with albumentations DualTransform.
    """

    def __init__(self, image_size: Tuple[int, int], is_train: bool = True):
        """
        Args:
            image_size: (height, width) target size
            is_train: Whether this is for training (enables augmentations)
        """
        self.image_size = image_size
        self.is_train = is_train
        self.target_h, self.target_w = image_size

        # Image-only transforms (applied after geometric transforms)
        if is_train:
            self.color_transform = A.Compose([
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(2, 4), p=0.5),
                    A.MedianBlur(blur_limit=2, p=0.5),
                ], p=0.3),
                A.GaussNoise(std_range=(0.1, 0.6), p=0.3),
            ])
        else:
            self.color_transform = None

        # Normalization (always applied)
        self.normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image: np.ndarray, polylines: List[np.ndarray],
                 widths: List[float]) -> Dict:
        """
        Apply transforms to image and polylines.

        Args:
            image: (H, W, 3) numpy array, uint8, RGB
            polylines: List of (N, 2) numpy arrays in pixel coordinates
            widths: List of line widths

        Returns:
            Dictionary with transformed data
        """
        orig_h, orig_w = image.shape[:2]

        # 1. RESIZE (always applied)
        image_resized, polylines_resized = self._resize(image, polylines, orig_h, orig_w)


        # 2. ROTATION (training only)
        if self.is_train and np.random.rand() < 0.5:
            angle = np.random.uniform(-2, 2)
            image_resized, polylines_resized = self._rotate(image_resized, polylines_resized, angle)

        # 3. COLOR AUGMENTATIONS (training only, image only)
        if self.is_train and self.color_transform is not None:
            image_resized = self.color_transform(image=image_resized)['image']

        # 4. NORMALIZE (always applied)
        image_final = self.normalize(image=image_resized)['image']

        return {
            'image': image_final,
            'polylines': polylines_resized,
            'widths': widths
        }

    def _resize(self, image: np.ndarray, polylines: List[np.ndarray],
                orig_h: int, orig_w: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Resize image and scale polylines."""
        # Resize image
        image_resized = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        # Scale factors
        scale_x = self.target_w / orig_w
        scale_y = self.target_h / orig_h

        # Scale polylines
        polylines_resized = []
        for poly in polylines:
            if len(poly) == 0:
                polylines_resized.append(poly)
                continue

            poly_scaled = poly.copy()
            poly_scaled[:, 0] *= scale_x  # x coordinates
            poly_scaled[:, 1] *= scale_y  # y coordinates

            # Clip to image bounds
            poly_scaled[:, 0] = np.clip(poly_scaled[:, 0], 0, self.target_w - 1)
            poly_scaled[:, 1] = np.clip(poly_scaled[:, 1], 0, self.target_h - 1)

            polylines_resized.append(poly_scaled)

        return image_resized, polylines_resized

    def _horizontal_flip(self, image: np.ndarray, polylines: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Flip image and polylines horizontally."""
        # Flip image
        image_flipped = cv2.flip(image, 1)

        # Flip polylines
        polylines_flipped = []
        for poly in polylines:
            if len(poly) == 0:
                polylines_flipped.append(poly)
                continue

            poly_flipped = poly.copy()
            poly_flipped[:, 0] = self.target_w - 1 - poly_flipped[:, 0]
            polylines_flipped.append(poly_flipped)

        return image_flipped, polylines_flipped

    def _rotate(self, image: np.ndarray, polylines: List[np.ndarray],
                angle: float) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Rotate image and polylines."""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate image
        image_rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # Rotate polylines
        polylines_rotated = []
        for poly in polylines:
            if len(poly) == 0:
                polylines_rotated.append(poly)
                continue

            # Apply rotation matrix to each point
            # M is 2x3, we need to add homogeneous coordinate
            ones = np.ones((len(poly), 1))
            poly_homogeneous = np.hstack([poly, ones])  # (N, 3)
            poly_rotated = (M @ poly_homogeneous.T).T  # (N, 2)

            # Clip to image bounds
            poly_rotated[:, 0] = np.clip(poly_rotated[:, 0], 0, w - 1)
            poly_rotated[:, 1] = np.clip(poly_rotated[:, 1], 0, h - 1)

            polylines_rotated.append(poly_rotated)

        return image_rotated, polylines_rotated


def get_train_transforms(image_size: [int, int] = (480, 640)) -> Line2FormerTransform:
    """Get training transforms."""
    return Line2FormerTransform(image_size, is_train=True)


def get_val_transforms(image_size: [int, int] = (480, 640)) -> Line2FormerTransform:
    """Get validation transforms."""
    return Line2FormerTransform(image_size, is_train=False)


def get_transforms(train: bool = True, image_size: tuple = (480, 640)):
    """
    Get data transforms for training or validation.

    Args:
        train: If True, return training transforms with augmentation
        image_size: Target image size (H, W)

    Returns:
        transform: Transform function
    """
    if train:
        get_train_transforms(image_size)
    else:
        get_val_transforms(image_size)
"""
LineFormer Dataset
Loads synthetic line plot data with COCO-style annotations.
"""

import torch
from torch.utils.data import Dataset
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .utils import resample_polyline, normalize_polyline
from .transforms import get_train_transforms, get_val_transforms


class LineFormerDataset(Dataset):
    """
    Dataset for line extraction from chart images.

    Annotation format (COCO-style):
    {
      "images": [{"id": 0, "file_name": "image_000000.png", "width": 640, "height": 480}],
      "annotations": [
        {
          "id": 1,
          "image_id": 0,
          "category_id": 1,
          "centerline": [[x1, y1], [x2, y2], ...],
          "line_width": 2.5,
          "bbox": [x, y, w, h]
        }
      ]
    }
    """

    def __init__(
            self,
            data_root: Path,
            split: str = 'train',
            max_lines: int = 20,
            max_points: int = 50,
            image_size: Tuple[int, int] = (480, 640),
            normalize_coords: bool = True,
            transform: Optional[object] = None,
            min_line_points: int = 2  # ← FIXED: Changed from 5 to 2
    ):
        """
        Args:
            data_root: Path to data directory (contains train/ and val/)
            split: 'train' or 'val'
            max_lines: Maximum number of lines per image (for padding)
            max_points: Maximum points per polyline (for resampling)
            image_size: (height, width) for resizing
            normalize_coords: Whether to normalize polyline coords to [0, 1]
            transform: Custom transform (if None, uses default)
            min_line_points: Minimum points required for a valid line
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_lines = max_lines
        self.max_points = max_points
        self.image_size = image_size
        self.normalize_coords = normalize_coords
        self.min_line_points = min_line_points

        # Load annotations
        ann_path = self.data_root / split / 'annotations.json'
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations not found: {ann_path}")

        with open(ann_path, 'r') as f:
            self.coco = json.load(f)

        # Build image and annotation mappings
        self.images = {img['id']: img for img in self.coco['images']}
        self.image_ids = sorted(self.images.keys())

        # Group annotations by image_id
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id in self.annotations:
                self.annotations[img_id].append(ann)

        # Set up transforms
        if transform is None:
            if split == 'train':
                self.transform = get_train_transforms(image_size)
            else:
                self.transform = get_val_transforms(image_size)
        else:
            self.transform = transform

        print(f"Loaded {split} dataset: {len(self.image_ids)} images, "
              f"{len(self.coco['annotations'])} lines")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - image: (3, H, W) tensor
                - centerlines: (max_lines, max_points, 2) tensor
                - widths: (max_lines, max_points) tensor
                - valid_mask: (max_lines,) boolean tensor
                - num_lines: scalar (number of actual lines)
                - image_id: scalar
                - original_size: (H, W) of the original image
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image at ORIGINAL size
        img_path = self.data_root / self.split / 'images' / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store ACTUAL original size (not from JSON, in case of mismatch)
        original_height, original_width = image.shape[:2]

        # Load annotations
        anns = self.annotations[img_id]

        # Extract polylines and widths (in ORIGINAL image pixel coordinates)
        polylines = []
        widths = []

        for ann in anns:
            centerline = np.array(ann['centerline'], dtype=np.float32)

            # Filter out degenerate lines
            if len(centerline) < self.min_line_points:
                continue

            polylines.append(centerline)
            widths.append(ann.get('line_width', 2.0))



        # Apply transformations (resize + augmentations)
        if self.transform is not None:
            transformed = self.transform(image, polylines, widths)
            image = transformed['image']
            polylines = transformed['polylines']
            widths = transformed['widths']



        # Convert to tensors with padding
        centerlines_tensor, widths_tensor, valid_mask = self._process_annotations(
            polylines, widths, self.image_size
        )

        # Convert image to tensor (C, H, W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return {
            'image': image,
            'centerlines': centerlines_tensor,
            'widths': widths_tensor,
            'valid_mask': valid_mask,
            'num_lines': torch.tensor(len(polylines), dtype=torch.long),
            'image_id': torch.tensor(img_id, dtype=torch.long),
            'original_size': torch.tensor([original_height, original_width], dtype=torch.long)
        }

    def _process_annotations(
            self,
            polylines: List[np.ndarray],
            widths: List[float],
            image_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert list of polylines to padded tensors.

        Args:
            polylines: List of (N, 2) arrays in pixel coordinates (after transform)
            widths: List of line widths
            image_size: (height, width) of the transformed image

        Returns:
            centerlines: (max_lines, max_points, 2)
            widths: (max_lines, max_points)
            valid_mask: (max_lines,)
        """
        centerlines = torch.zeros(self.max_lines, self.max_points, 2, dtype=torch.float32)
        widths_tensor = torch.zeros(self.max_lines, self.max_points, dtype=torch.float32)
        valid_mask = torch.zeros(self.max_lines, dtype=torch.bool)

        num_lines = min(len(polylines), self.max_lines)

        for i in range(num_lines):
            polyline = polylines[i]
            width = widths[i]

            # Resample to fixed number of points
            polyline_resampled = resample_polyline(polyline, self.max_points)

            # Normalize coordinates if requested
            if self.normalize_coords:
                polyline_resampled = normalize_polyline(polyline_resampled, image_size)

            # Clip to valid range
            if self.normalize_coords:
                polyline_resampled = np.clip(polyline_resampled, 0, 1)
            else:
                height, width_img = image_size
                polyline_resampled[:, 0] = np.clip(polyline_resampled[:, 0], 0, width_img - 1)
                polyline_resampled[:, 1] = np.clip(polyline_resampled[:, 1], 0, height - 1)

            centerlines[i] = torch.from_numpy(polyline_resampled)
            widths_tensor[i] = width  # Constant width for all points
            valid_mask[i] = True

        return centerlines, widths_tensor, valid_mask

    def get_image_path(self, idx: int) -> Path:
        """Get the file path for an image."""
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        return self.data_root / self.split / 'images' / img_info['file_name']

    def get_raw_annotation(self, idx: int) -> Dict:
        """Get raw annotation without processing."""
        img_id = self.image_ids[idx]
        return {
            'image_info': self.images[img_id],
            'annotations': self.annotations[img_id]
        }
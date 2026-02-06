"""
COCO Format Utilities
Convert generated masks to COCO instance segmentation format
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path


def mask_to_polygon(mask: np.ndarray, tolerance: float = 1.0) -> List[List[float]]:
    """
    Convert binary mask to polygon format.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 255
        tolerance: Simplification tolerance for polygon approximation
        
    Returns:
        List of polygons, each polygon is [x1, y1, x2, y2, ...]
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Simplify contour
        epsilon = tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Need at least 3 points for a valid polygon
        if len(approx) >= 3:
            # Flatten to [x1, y1, x2, y2, ...]
            polygon = approx.flatten().tolist()
            # Convert to float for JSON serialization
            polygon = [float(p) for p in polygon]
            polygons.append(polygon)
    
    return polygons


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary mask to COCO RLE format.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        
    Returns:
        RLE dictionary with 'counts' and 'size'
    """
    # Flatten in Fortran order (column-major)
    pixels = mask.flatten(order='F')
    
    # Convert to binary
    pixels = (pixels > 0).astype(np.uint8)
    
    # Compute run-length encoding
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    return {
        'counts': runs.tolist(),
        'size': list(mask.shape)
    }


def compute_bbox(mask: np.ndarray) -> List[float]:
    """
    Compute bounding box from mask.
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        [x, y, width, height] in COCO format
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def compute_area(mask: np.ndarray) -> int:
    """Compute area (number of pixels) of mask"""
    return int(np.sum(mask > 0))


class COCOAnnotationBuilder:
    """
    Builder class for creating COCO format annotation files.
    """
    
    def __init__(
        self,
        description: str = "Synthetic Line Plot Dataset",
        version: str = "1.0",
        categories: List[Dict] = None
    ):
        """
        Initialize the COCO annotation builder.
        
        Args:
            description: Dataset description
            version: Dataset version
            categories: List of category dicts. Default is single "line" category.
        """
        self.description = description
        self.version = version
        
        if categories is None:
            categories = [
                {"id": 1, "name": "line", "supercategory": "chart_element"}
            ]
        self.categories = categories
        
        self.images = []
        self.annotations = []
        
        self._image_id_counter = 0
        self._annotation_id_counter = 0
    
    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        **kwargs
    ) -> int:
        """
        Add an image entry.
        
        Args:
            file_name: Image filename (relative to images directory)
            width: Image width in pixels
            height: Image height in pixels
            **kwargs: Additional metadata
            
        Returns:
            Image ID
        """
        self._image_id_counter += 1
        
        image_entry = {
            "id": self._image_id_counter,
            "file_name": file_name,
            "width": width,
            "height": height,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        image_entry.update(kwargs)
        
        self.images.append(image_entry)
        return self._image_id_counter
    
    def add_annotation(
        self,
        image_id: int,
        mask: np.ndarray,
        category_id: int = 1,
        use_rle: bool = False,
        **kwargs
    ) -> Optional[int]:
        """
        Add an annotation entry from a binary mask.
        
        Args:
            image_id: ID of the image this annotation belongs to
            mask: Binary mask (H, W)
            category_id: Category ID (default 1 for "line")
            use_rle: If True, use RLE encoding; else use polygon
            **kwargs: Additional metadata
            
        Returns:
            Annotation ID, or None if mask is empty
        """
        # Compute area
        area = compute_area(mask)
        if area == 0:
            return None
        
        # Compute bounding box
        bbox = compute_bbox(mask)
        
        # Convert to segmentation format
        if use_rle:
            segmentation = mask_to_rle(mask)
        else:
            segmentation = mask_to_polygon(mask)
            if not segmentation:  # No valid polygons
                return None
        
        self._annotation_id_counter += 1
        
        annotation_entry = {
            "id": self._annotation_id_counter,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
        }
        annotation_entry.update(kwargs)
        
        self.annotations.append(annotation_entry)
        return self._annotation_id_counter
    
    def add_annotations_from_masks(
        self,
        image_id: int,
        masks: List[np.ndarray],
        category_id: int = 1,
        use_rle: bool = False
    ) -> List[int]:
        """
        Add multiple annotations from a list of masks.
        
        Args:
            image_id: Image ID
            masks: List of binary masks
            category_id: Category ID
            use_rle: Use RLE encoding
            
        Returns:
            List of annotation IDs (excluding None for empty masks)
        """
        annotation_ids = []
        for mask in masks:
            ann_id = self.add_annotation(
                image_id=image_id,
                mask=mask,
                category_id=category_id,
                use_rle=use_rle
            )
            if ann_id is not None:
                annotation_ids.append(ann_id)
        return annotation_ids
    
    def build(self) -> Dict[str, Any]:
        """
        Build the complete COCO annotation dictionary.
        
        Returns:
            COCO format dictionary
        """
        return {
            "info": {
                "description": self.description,
                "version": self.version,
                "year": datetime.now().year,
                "contributor": "Synthetic Line Plot Generator",
                "date_created": datetime.now().strftime("%Y-%m-%d"),
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Synthetic Data License",
                    "url": ""
                }
            ],
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }
    
    def save(self, output_path: str):
        """
        Save annotations to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        coco_dict = self.build()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(coco_dict, f, indent=2)
        
        print(f"Saved COCO annotations to {output_path}")
        print(f"  Images: {len(self.images)}")
        print(f"  Annotations: {len(self.annotations)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the annotations"""
        if not self.annotations:
            return {"images": 0, "annotations": 0}
        
        # Annotations per image
        ann_per_image = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            ann_per_image[img_id] = ann_per_image.get(img_id, 0) + 1
        
        counts = list(ann_per_image.values())
        
        return {
            "images": len(self.images),
            "annotations": len(self.annotations),
            "avg_annotations_per_image": np.mean(counts) if counts else 0,
            "min_annotations_per_image": min(counts) if counts else 0,
            "max_annotations_per_image": max(counts) if counts else 0,
            "annotations_distribution": {
                i: counts.count(i) for i in range(1, max(counts) + 1) if counts.count(i) > 0
            } if counts else {}
        }


def verify_coco_format(annotation_path: str) -> bool:
    """
    Verify that a COCO annotation file is valid.
    
    Args:
        annotation_path: Path to annotation JSON file
        
    Returns:
        True if valid, raises exception otherwise
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # Check required keys
    required_keys = ['info', 'images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    # Check images
    image_ids = set()
    for img in data['images']:
        if 'id' not in img or 'file_name' not in img:
            raise ValueError(f"Invalid image entry: {img}")
        if img['id'] in image_ids:
            raise ValueError(f"Duplicate image ID: {img['id']}")
        image_ids.add(img['id'])
    
    # Check annotations
    annotation_ids = set()
    for ann in data['annotations']:
        if 'id' not in ann or 'image_id' not in ann or 'segmentation' not in ann:
            raise ValueError(f"Invalid annotation entry: {ann}")
        if ann['id'] in annotation_ids:
            raise ValueError(f"Duplicate annotation ID: {ann['id']}")
        if ann['image_id'] not in image_ids:
            raise ValueError(f"Annotation references non-existent image: {ann['image_id']}")
        annotation_ids.add(ann['id'])
    
    # Check categories
    category_ids = set()
    for cat in data['categories']:
        if 'id' not in cat or 'name' not in cat:
            raise ValueError(f"Invalid category entry: {cat}")
        category_ids.add(cat['id'])
    
    # Verify all annotations reference valid categories
    for ann in data['annotations']:
        if ann.get('category_id') not in category_ids:
            raise ValueError(f"Annotation references non-existent category: {ann['category_id']}")
    
    print(f"COCO format verification passed!")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Categories: {len(data['categories'])}")
    
    return True


# Test
if __name__ == "__main__":
    print("Testing COCO utilities")
    print("=" * 50)
    
    # Create a test mask
    mask = np.zeros((100, 200), dtype=np.uint8)
    cv2.line(mask, (10, 50), (190, 50), 255, 2)
    
    # Test polygon conversion
    polygons = mask_to_polygon(mask)
    print(f"\nPolygon points: {len(polygons[0]) if polygons else 0}")
    
    # Test RLE conversion
    rle = mask_to_rle(mask)
    print(f"RLE counts length: {len(rle['counts'])}")
    
    # Test bbox
    bbox = compute_bbox(mask)
    print(f"Bounding box: {bbox}")
    
    # Test area
    area = compute_area(mask)
    print(f"Area: {area} pixels")
    
    # Test builder
    print("\n" + "=" * 50)
    print("Testing COCOAnnotationBuilder")
    
    builder = COCOAnnotationBuilder(description="Test Dataset")
    
    # Add image
    img_id = builder.add_image("test_image.png", width=200, height=100)
    print(f"Added image with ID: {img_id}")
    
    # Add annotation
    ann_id = builder.add_annotation(img_id, mask)
    print(f"Added annotation with ID: {ann_id}")
    
    # Get stats
    stats = builder.get_stats()
    print(f"Stats: {stats}")
    
    # Build and display
    coco_dict = builder.build()
    print(f"\nCOCO dict keys: {list(coco_dict.keys())}")

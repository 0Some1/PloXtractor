# utils.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LinePlotDataset(Dataset):
    def __init__(self, data_root, split, processor):
        self.root = data_root
        self.split = split
        self.processor = processor
        self.img_dir = os.path.join(data_root, split, "images")

        # Load COCO annotations
        json_path = os.path.join(data_root, split, "annotations.json")
        with open(json_path) as f:
            coco = json.load(f)

        self.images = {img['id']: img for img in coco['images']}
        self.annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # 1. Load Image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Process Image ONLY using the processor
        # We assume single image input, so we take the 0-th element
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)  # (3, H_new, W_new)

        # Get the new size after processor resizing
        new_h, new_w = pixel_values.shape[1:]

        # 3. Process Masks Manually (to handle overlaps correctly)
        anns = self.annotations.get(img_id, [])
        instance_masks = []
        class_labels = []

        h_orig, w_orig = image.shape[:2]

        for ann in anns:
            # Draw polygon to mask (Original Size)
            mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)

            # Resize mask to match the processor's output size
            # Use NEAREST neighbor to keep values 0 or 1
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            instance_masks.append(mask_resized)
            class_labels.append(1)  # Class ID 1 for "line"

        # Handle images with no lines (empty)
        if len(instance_masks) == 0:
            # Create a dummy background mask
            instance_masks = [np.zeros((new_h, new_w), dtype=np.uint8)]
            class_labels = [0]  # Background class (0)

        # Convert to Tensor
        mask_tensor = torch.tensor(np.array(instance_masks), dtype=torch.float32)
        class_tensor = torch.tensor(class_labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_tensor,
            "class_labels": class_tensor,
            "pixel_mask": inputs.get("pixel_mask", None)  # Pass pixel_mask if it exists (for padding)
        }


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])

    # Check if pixel_mask exists (it might be None for some processors)
    if batch[0]["pixel_mask"] is not None:
        pixel_mask = torch.stack([x["pixel_mask"].squeeze(0) for x in batch])
    else:
        pixel_mask = None

    mask_labels = [x["mask_labels"] for x in batch]
    class_labels = [x["class_labels"] for x in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }
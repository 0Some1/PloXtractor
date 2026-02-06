import argparse
import torch
import cv2
import numpy as np
import os
import json
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    # Use raw strings (r"...") for Windows paths
    parser.add_argument("--model-path", default=r"D:\Project\PloXtractor\hf_mask2former\output_hf\checkpoint-8")
    parser.add_argument("--image",
                        default=r"D:\Project\PloXtractor\SyntheticLinePlotDataset\test\images\image_000003.png")
    parser.add_argument("--output", default="result.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path}...")

    # 1. Load Processor
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(args.model_path)
    except OSError:
        processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")

    # Force the processor to know about our custom classes
    processor.id2label = {0: "background", 1: "line"}
    processor.label2id = {"background": 0, "line": 1}

    # 2. Load Model
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_path).to(device)
    model.eval()

    # 3. Preprocess
    print(f"Processing {args.image}...")
    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 4. Predict
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Post-process (Universal Handler)
    # We try instance segmentation first. If the model config forces panoptic, we adapt.
    target_sizes = [image.size[::-1]]  # (H, W)

    # Note: We explicitly ask for panoptic if that's what the model is producing,
    # but let's stick to the method that produced your screenshot.
    # If the previous run used `post_process_instance_segmentation` and gave that result,
    # it means the model config is overriding the method behavior.

    # Let's use the generic post-processor which handles the format automatically
    results = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=target_sizes,
        threshold=0.5
    )
    result = results[0]

    # 6. Visualization (Robust to Format)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # CHECK 1: Is it the Panoptic format (from your screenshot)?
    if 'segments_info' in result:
        print("Detected Panoptic/Segments Info format.")
        panoptic_map = result['segmentation'].cpu().numpy()  # The 2D ID map
        segments_info = result['segments_info']

        found_counts = 0
        for info in segments_info:
            score = info['score']
            if score < 0.5: continue

            inst_id = info['id']
            label_id = info['label_id']

            # Create binary mask for this specific instance ID
            mask = (panoptic_map == inst_id).astype(np.uint8)

            # Visualize
            color = np.random.randint(0, 255, 3).tolist()
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_cv, contours, -1, color, 2)
            found_counts += 1

        print(f"Visualized {found_counts} instances from segments_info.")

    # CHECK 2: Is it the standard Instance format?
    elif 'masks' in result or 'segmentation' in result:
        # Standard format usually has 'segmentation' (N,H,W) or 'masks'
        masks = result.get('masks', result.get('segmentation'))
        scores = result['scores']
        labels = result['labels']

        print(f"Detected Standard Instance format with {len(masks)} masks.")

        for mask, label, score in zip(masks, labels, scores):
            if score < 0.5: continue
            mask_np = mask.cpu().numpy().astype(np.uint8)
            color = np.random.randint(0, 255, 3).tolist()
            contours, _ = cv2.findContours(mask_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_cv, contours, -1, color, 2)

    else:
        print("Unknown result format. Keys found:", result.keys())

    cv2.imwrite(args.output, img_cv)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
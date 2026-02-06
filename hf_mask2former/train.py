import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerConfig
)
from utils import LinePlotDataset, collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="D:/Project/PloXtractor/SyntheticLinePlotDataset")
    parser.add_argument("--output-dir", default="output_hf")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Processor
    # Just load with default config.
    model_checkpoint = "facebook/mask2former-swin-tiny-coco-instance"
    processor = Mask2FormerImageProcessor.from_pretrained(
        model_checkpoint,
    )

    # 2. Setup Dataset
    train_dataset = LinePlotDataset(args.data_dir, "train", processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 3. Setup Model
    # We ignore mismatched sizes because we are fine-tuning on 1 class (line) vs COCO's 80
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_checkpoint,
        id2label={0: "background", 1: "line"},
        label2id={"background": 0, "line": 1},
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress:
            # Move to device
            pixel_values = batch["pixel_values"].to(device)

            # Handle pixel_mask (it might be None)
            pixel_mask = batch["pixel_mask"]
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)

            mask_labels = [x.to(device) for x in batch["mask_labels"]]
            class_labels = [x.to(device) for x in batch["class_labels"]]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,  # Pass None if it is None, model handles it
                mask_labels=mask_labels,
                class_labels=class_labels
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch + 1} Avg Loss: {epoch_loss / len(train_loader)}")

        # Save checkpoint
        model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))
        processor.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch}"))


if __name__ == "__main__":
    main()
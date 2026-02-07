"""
Training script for Line2Former
Uses PyTorch Lightning for clean, modular training.
"""

import os
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from line2former.models import Line2Former
from line2former.losses import Line2FormerLoss
from line2former.dataset import Line2FormerDataset
from torch.utils.data import DataLoader


class Line2FormerLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Line2Former.
    """

    def __init__(
            self,
            # Model config
            backbone_name: str = 'hrnet_w32',
            num_queries: int = 20,
            max_points: int = 50,
            d_model: int = 256,
            nhead: int = 8,
            num_decoder_layers: int = 6,

            # Loss config
            weight_centerline: float = 5.0,
            weight_width: float = 1.0,
            weight_objectness: float = 2.0,
            weight_mask: float = 2.0,

            # Optimizer config
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-4,
            lr_scheduler: str = 'cosine',
            warmup_epochs: int = 5,
            max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Build model
        self.model = Line2Former(
            backbone_type=backbone_name,
            num_queries=num_queries,
            max_points=max_points,
            d_model=d_model,
            num_heads=nhead,
            num_decoder_layers=num_decoder_layers,
        )

        # Build loss
        self.criterion = Line2FormerLoss(
            weight_centerline=weight_centerline,
            weight_width=weight_width,
            weight_objectness=weight_objectness,
            weight_mask=weight_mask,
            num_queries=num_queries,
        )

        # Metrics storage
        self.validation_step_outputs = []

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        """Training step."""
        images = batch['image']
        targets = {
            'centerlines': batch['centerlines'],
            'widths': batch['widths'],
            'valid_mask': batch['valid_mask'],
        }

        # Forward pass
        outputs = self(images)

        # Compute loss
        loss_dict = self.criterion(outputs, targets)

        # Log losses
        self.log('train/loss', loss_dict['loss_total'], prog_bar=True)
        self.log('train/loss_centerline', loss_dict['loss_centerline'])
        self.log('train/loss_width', loss_dict['loss_width'])
        self.log('train/loss_objectness', loss_dict['loss_objectness'])
        self.log('train/loss_mask', loss_dict['loss_mask'])

        return loss_dict['loss_total']

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images = batch['image']
        targets = {
            'centerlines': batch['centerlines'],
            'widths': batch['widths'],
            'valid_mask': batch['valid_mask'],
        }

        # Forward pass
        outputs = self(images)

        # Compute loss
        loss_dict = self.criterion(outputs, targets)

        # Log losses
        self.log('val/loss', loss_dict['loss_total'], prog_bar=True)
        self.log('val/loss_centerline', loss_dict['loss_centerline'])
        self.log('val/loss_width', loss_dict['loss_width'])
        self.log('val/loss_objectness', loss_dict['loss_objectness'])
        self.log('val/loss_mask', loss_dict['loss_mask'])

        # Store for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': loss_dict['loss_total'].detach(),
            'centerlines': outputs['centerlines'].detach(),
            'objectness': outputs['objectness'].detach(),
        })

        return loss_dict['loss_total']

    def on_validation_epoch_end(self):
        """Compute epoch-level metrics."""
        if len(self.validation_step_outputs) == 0:
            return

        # Average loss
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log('val/epoch_loss', avg_loss)

        # Clear outputs
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate learning rates for backbone and rest
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.hparams.learning_rate * 0.1},
            {'params': other_params, 'lr': self.hparams.learning_rate},
        ], weight_decay=self.hparams.weight_decay)

        # Learning rate scheduler
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6,
            )
        elif self.hparams.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


class LinePlotDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for line plot dataset.
    """

    def __init__(
            self,
            data_root: str,
            train_annotation: str,
            val_annotation: str,
            batch_size: int = 8,
            num_workers: int = 4,
            image_size: tuple = (480, 640),
            max_lines: int = 10,
            max_points: int = 50,
            min_line_points: int = 10,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.train_annotation = train_annotation
        self.val_annotation = val_annotation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_lines = max_lines
        self.max_points = max_points
        self.min_line_points = min_line_points

    def setup(self, stage=None):
        """Load datasets."""
        if stage == 'fit' or stage is None:
            # Training dataset
            self.train_dataset = Line2FormerDataset(
                data_root = self.data_root,
                split='train',
                max_lines=self.max_lines,
                max_points=self.max_points,
                min_line_points=self.min_line_points,
            )

            # Validation dataset
            self.val_dataset = Line2FormerDataset(
                data_root=self.data_root,
                split='val',
                max_lines=self.max_lines,
                max_points=self.max_points,
                min_line_points=self.min_line_points,
            )

            print(f"Train dataset: {len(self.train_dataset)} samples")
            print(f"Val dataset: {len(self.val_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Line2Former')

    # Data arguments
    parser.add_argument('--data_root', type=str, default=r"../data",
                        help='Root directory of dataset')
    parser.add_argument('--train_annotation', type=str, default='annotations_train.json',
                        help='Training annotation file')
    parser.add_argument('--val_annotation', type=str, default='annotations_val.json',
                        help='Validation annotation file')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='hrnet_w32',
                        choices=['hrnet_w18', 'hrnet_w32', 'hrnet_w48',
                                 'dilated_resnet50', 'hrnet', 'lightweight'])
    parser.add_argument('--num_queries', type=int, default=20)
    parser.add_argument('--max_points', type=int, default=50)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_decoder_layers', type=int, default=6)

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau'])

    # Loss weights
    parser.add_argument('--weight_centerline', type=float, default=5.0)
    parser.add_argument('--weight_width', type=float, default=1.0)
    parser.add_argument('--weight_objectness', type=float, default=2.0)
    parser.add_argument('--weight_mask', type=float, default=0.0)

    # System arguments
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32',
                        choices=['16', '32', 'bf16'])
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup data module
    data_module = LinePlotDataModule(
        data_root=args.data_root,
        train_annotation=args.train_annotation,
        val_annotation=args.val_annotation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Setup model
    model = Line2FormerLightning(
        backbone_name=args.backbone,
        num_queries=args.num_queries,
        max_points=args.max_points,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        weight_centerline=args.weight_centerline,
        weight_width=args.weight_width,
        weight_objectness=args.weight_objectness,
        weight_mask=args.weight_mask,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        max_epochs=args.max_epochs,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='line2former-{epoch:02d}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
        ),
    ]

    # Loggers
    loggers = [
        TensorBoardLogger(
            save_dir=output_dir / 'logs',
            name='line2former',
        ),
        CSVLogger(
            save_dir=output_dir / 'logs',
            name='line2former',
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        gradient_clip_val=0.1,
        accumulate_grad_batches=1,
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from,
    )

    print("Training complete!")
    print(f"Best model saved at: {output_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
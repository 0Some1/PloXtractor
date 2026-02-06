"""
LineFormer: Main Model
Combines all components for end-to-end line extraction.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .backbone import HighResBackbone, LightweightBackbone
from .line_aware_conv import LineAwareFeatureExtractor
from .decoder import LineQueryDecoder, PositionalEncoding2D, LinePredictionHead
from .renderer import DifferentiableLineRenderer


class LineFormer(nn.Module):
    """
    LineFormer: Transformer-based model for line extraction from images.

    Architecture:
        1. High-resolution backbone (preserves spatial details)
        2. Line-aware feature extraction (oriented convolutions + ridge detection)
        3. Transformer decoder with line queries
        4. Dual prediction heads (centerline + width)
        5. Differentiable line renderer (for training)
    """

    def __init__(
            self,
            # Model config
            num_queries: int = 20,
            max_points: int = 50,
            d_model: int = 256,

            # Backbone config
            backbone_type: str = 'lightweight',  # 'lightweight' or 'hrnet'
            backbone_channels: int = 256,

            # Decoder config
            num_decoder_layers: int = 6,
            num_heads: int = 8,
            dim_feedforward: int = 1024,
            dropout: float = 0.1,

            # Line-aware config
            use_line_aware: bool = True,
            num_orientations: int = 8,

            # Renderer config
            image_size: Tuple[int, int] = (480, 640),
            renderer_sigma: float = 1.0,
    ):
        """
        Args:
            num_queries: Maximum number of lines per image
            max_points: Maximum points per polyline
            d_model: Hidden dimension
            backbone_type: 'lightweight' or 'hrnet'
            backbone_channels: Output channels from backbone
            num_decoder_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            use_line_aware: Whether to use line-aware convolutions
            num_orientations: Number of orientations for line-aware conv
            image_size: (H, W) input image size
            renderer_sigma: Smoothing parameter for renderer
        """
        super().__init__()

        self.num_queries = num_queries
        self.max_points = max_points
        self.d_model = d_model
        self.image_size = image_size

        # 1. Backbone
        if backbone_type == 'lightweight':
            self.backbone = LightweightBackbone(
                in_channels=3,
                output_channels=backbone_channels,
            )
        elif backbone_type == 'hrnet':
            self.backbone = HighResBackbone(
                in_channels=3,
                base_channels=64,
                output_channels=backbone_channels,
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")

        # 2. Line-aware feature extraction
        if use_line_aware:
            self.line_aware_conv = LineAwareFeatureExtractor(
                in_channels=backbone_channels,
                out_channels=d_model,
                num_orientations=num_orientations,
            )
        else:
            self.line_aware_conv = nn.Sequential(
                nn.Conv2d(backbone_channels, d_model, kernel_size=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
            )

        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding2D(
            d_model=d_model,
            max_h=image_size[0] // 4,
            max_w=image_size[1] // 4,
        )

        # 4. Transformer decoder
        self.decoder = LineQueryDecoder(
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_points=max_points,
        )

        # 5. Prediction heads
        self.prediction_head = LinePredictionHead(
            d_model=d_model,
            max_points=max_points,
        )

        # 6. Differentiable renderer
        self.renderer = DifferentiableLineRenderer(
            image_size=image_size,
            sigma=renderer_sigma,
        )

    def forward(
            self,
            images: torch.Tensor,
            return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: (B, 3, H, W) input images
            return_features: Whether to return intermediate features

        Returns:
            outputs: Dictionary containing:
                - centerlines: (B, num_queries, max_points, 2) predicted polylines
                - widths: (B, num_queries, max_points) predicted widths
                - objectness: (B, num_queries) objectness scores
                - masks: (B, num_queries, H, W) rendered masks (if return_features=True)
        """
        B = images.shape[0]

        # 1. Extract features
        backbone_features = self.backbone(images)  # (B, C, H/4, W/4)

        # 2. Line-aware features
        line_features = self.line_aware_conv(backbone_features)  # (B, d_model, H/4, W/4)

        # 3. Add positional encoding
        pos_embed = self.pos_encoder(line_features)  # (B, d_model, H/4, W/4)

        # 4. Decode line queries
        queries = self.decoder(line_features, pos_embed)  # (B, num_queries, d_model)

        # 5. Predict centerlines and widths
        centerlines, widths, objectness = self.prediction_head(queries)

        outputs = {
            'centerlines': centerlines,
            'widths': widths,
            'objectness': objectness,
        }

        # Optional: Render masks
        if return_features:
            # Create valid mask from objectness
            valid_mask = objectness > 0.5

            # Render masks
            masks = self.renderer(centerlines, widths, valid_mask)
            outputs['masks'] = masks
            outputs['features'] = line_features

        return outputs

    def render_predictions(
            self,
            centerlines: torch.Tensor,
            widths: torch.Tensor,
            objectness: torch.Tensor,
            threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Render predicted polylines as masks.

        Args:
            centerlines: (B, num_queries, max_points, 2)
            widths: (B, num_queries, max_points)
            objectness: (B, num_queries)
            threshold: Objectness threshold

        Returns:
            masks: (B, num_queries, H, W)
        """
        valid_mask = objectness > threshold
        masks = self.renderer(centerlines, widths, valid_mask)
        return masks

    @torch.no_grad()
    def predict(
            self,
            images: torch.Tensor,
            objectness_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode: predict lines and filter by objectness.

        Args:
            images: (B, 3, H, W)
            objectness_threshold: Threshold for filtering predictions

        Returns:
            predictions: Dictionary with filtered predictions
        """
        outputs = self.forward(images, return_features=False)

        centerlines = outputs['centerlines']
        widths = outputs['widths']
        objectness = outputs['objectness']

        # Filter by objectness
        valid_mask = objectness > objectness_threshold

        # Apply mask
        predictions = {
            'centerlines': centerlines,
            'widths': widths,
            'objectness': objectness,
            'valid_mask': valid_mask,
        }

        return predictions


class LineFormerLite(nn.Module):
    """
    Lightweight version of LineFormer for faster training/testing.
    Simplified architecture with fewer parameters.
    """

    def __init__(
            self,
            num_queries: int = 20,
            max_points: int = 50,
            d_model: int = 128,
            num_decoder_layers: int = 3,
            image_size: Tuple[int, int] = (480, 640),
    ):
        super().__init__()

        self.num_queries = num_queries
        self.max_points = max_points

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # After: Add assertion
        assert d_model % 4 == 0, "d_model must be divisible by 4 for positional encoding"
        self.pos_encoder = PositionalEncoding2D(
            d_model=d_model,
            max_h=image_size[0] // 4 + 16,  # Add padding for safety
            max_w=image_size[1] // 4 + 16,
        )

        # Decoder
        self.decoder = LineQueryDecoder(
            d_model=d_model,
            num_queries=num_queries,
            num_heads=4,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=512,
            max_points=max_points,
        )

        # Prediction head
        self.prediction_head = LinePredictionHead(d_model, max_points)

        # Renderer
        self.renderer = DifferentiableLineRenderer(image_size)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        pos_embed = self.pos_encoder(features)
        queries = self.decoder(features, pos_embed)
        centerlines, widths, objectness = self.prediction_head(queries)

        return {
            'centerlines': centerlines,
            'widths': widths,
            'objectness': objectness,
        }
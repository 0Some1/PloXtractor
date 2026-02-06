"""
High-Resolution Backbone for LineFormer
Preserves spatial resolution for thin line detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class HighResBackbone(nn.Module):
    """
    High-resolution feature extractor based on HRNet architecture.
    Maintains 1/4 resolution instead of 1/32 to preserve thin line details.
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            output_channels: int = 256,
            num_stages: int = 3,
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            base_channels: Base number of channels
            output_channels: Final output channels
            num_stages: Number of parallel resolution stages
        """
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = output_channels

        # Stem: Reduce to 1/4 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Multi-resolution parallel branches
        self.stage1 = self._make_stage(base_channels, base_channels, num_blocks=2)

        # Stage 2: Add transition layers
        self.transition1 = self._make_transition_layer([base_channels], [base_channels, base_channels * 2])
        self.stage2 = self._make_parallel_stages([base_channels, base_channels * 2], num_blocks=2)

        # Stage 3: Add transition layers
        self.transition2 = self._make_transition_layer(
            [base_channels, base_channels * 2],
            [base_channels, base_channels * 2, base_channels * 4]
        )
        self.stage3 = self._make_parallel_stages(
            [base_channels, base_channels * 2, base_channels * 4],
            num_blocks=3
        )

        # Fusion layer: Merge all resolutions to highest resolution
        self.fusion = self._make_fusion_layer(
            [base_channels, base_channels * 2, base_channels * 4],
            output_channels
        )

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Create a single-resolution stage."""
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels
            ))
        return nn.Sequential(*layers)

    def _make_parallel_stages(self, channels_list: List[int], num_blocks: int) -> nn.ModuleList:
        """Create parallel stages at different resolutions."""
        stages = nn.ModuleList()
        for channels in channels_list:
            stages.append(self._make_stage(channels, channels, num_blocks))
        return stages

    def _make_transition_layer(self, in_channels_list: List[int], out_channels_list: List[int]) -> nn.ModuleList:
        """
        Create transition layers to change number of branches and channels.

        Args:
            in_channels_list: List of input channels for each branch
            out_channels_list: List of output channels for each branch
        """
        num_in = len(in_channels_list)
        num_out = len(out_channels_list)

        transition_layers = nn.ModuleList()

        for i in range(num_out):
            if i < num_in:
                # Branch already exists, just adjust channels if needed
                if in_channels_list[i] != out_channels_list[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[i], out_channels_list[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(nn.Identity())
            else:
                # New branch, downsample from previous branch
                downsample_layers = []
                for j in range(i - num_in + 1):
                    in_ch = in_channels_list[-1] if j == 0 else out_channels_list[i]
                    out_ch = out_channels_list[i]
                    downsample_layers.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*downsample_layers))

        return transition_layers

    def _make_fusion_layer(self, in_channels_list: List[int], out_channels: int) -> nn.Module:
        """Fuse multi-resolution features to single high-res output."""
        return FusionModule(in_channels_list, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            features: (B, output_channels, H/4, W/4) high-res features
        """
        # Stem: (B, 3, H, W) -> (B, C, H/4, W/4)
        x = self.stem(x)

        # Stage 1: Single resolution
        x1 = self.stage1(x)

        # Transition 1: Create multi-resolution branches
        x_list = []
        for i, transition in enumerate(self.transition1):
            if i == 0:
                x_list.append(transition(x1))
            else:
                x_list.append(transition(x1))

        # Stage 2: Two parallel resolutions
        stage2_out = []
        for i, stage in enumerate(self.stage2):
            stage2_out.append(stage(x_list[i]))

        # Transition 2: Expand to three branches
        x_list = []
        for i, transition in enumerate(self.transition2):
            if i < len(stage2_out):
                x_list.append(transition(stage2_out[i]))
            else:
                x_list.append(transition(stage2_out[-1]))

        # Stage 3: Three parallel resolutions
        stage3_out = []
        for i, stage in enumerate(self.stage3):
            stage3_out.append(stage(x_list[i]))

        # Fusion: Merge all resolutions to highest resolution
        features = self.fusion(stage3_out)

        return features


class ResidualBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out += residual
        out = F.relu(out, inplace=True)

        return out


class FusionModule(nn.Module):
    """Fuse multi-resolution features to single output."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()

        # Upsample and project each resolution to the highest resolution
        self.upsample_layers = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            layers = []

            # Project channels
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))

            # Upsample if not highest resolution
            if i > 0:
                layers.append(nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=True))

            self.upsample_layers.append(nn.Sequential(*layers))

        # Final fusion conv
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: List of features at different resolutions
                [x1: (B, C1, H/4, W/4), x2: (B, C2, H/8, W/8), x3: (B, C3, H/16, W/16)]

        Returns:
            fused: (B, out_channels, H/4, W/4)
        """
        # Upsample and sum all features
        fused = 0
        for i, features in enumerate(features_list):
            fused = fused + self.upsample_layers[i](features)

        # Final fusion
        fused = self.fusion_conv(fused)

        return fused


# Lightweight version for faster training/testing
class LightweightBackbone(nn.Module):
    """Simplified backbone for faster prototyping."""

    def __init__(
            self,
            in_channels: int = 3,
            output_channels: int = 256,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            # 1/2 resolution
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 1/4 resolution
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Residual blocks at 1/4 resolution
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),

            # Final projection
            nn.Conv2d(256, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            features: (B, output_channels, H/4, W/4)
        """
        return self.backbone(x)
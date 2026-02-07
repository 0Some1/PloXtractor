"""
High-Resolution Backbone for Line2Former

Supports:
  - 'hrnet_w18', 'hrnet_w32', 'hrnet_w48': Pretrained HRNet from timm
  - 'dilated_resnet50': Pretrained ResNet-50 with dilated convolutions (torchvision)
  - 'lightweight': Simple CNN for fast prototyping (no pretrained weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PretrainedHRNetBackbone(nn.Module):
    """
    Pretrained HRNet backbone from timm.

    HRNet maintains parallel multi-resolution branches throughout the network,
    making it ideal for thin-structure detection. The highest-resolution branch
    stays at 1/4 input resolution.

    Requires: pip install timm
    """

    def __init__(
            self,
            variant: str = 'hrnet_w32',
            output_channels: int = 256,
            pretrained: bool = True,
    ):
        """
        Args:
            variant: HRNet variant ('hrnet_w18', 'hrnet_w32', 'hrnet_w48')
            output_channels: Desired output channel dimension
            pretrained: Whether to load ImageNet-pretrained weights
        """
        super().__init__()

        import timm

        # Create HRNet as a feature extractor (no classification head)
        self.hrnet = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
        )

        # timm HRNet with features_only returns features at strides [2, 4, 8, 16, 32].
        # We target stride 4 (1/4 resolution) as our output. Find which index that is,
        # and only fuse features from stride >= 4 (skip the stride-2 stem feature).
        feature_channels = self.hrnet.feature_info.channels()
        feature_strides = self.hrnet.feature_info.reduction()

        # Find features at stride >= 4 to fuse (skip stride-2 stem)
        self._fuse_indices = [i for i, s in enumerate(feature_strides) if s >= 4]
        assert len(self._fuse_indices) > 0, "No features at stride >= 4 found"

        # The target resolution is stride 4 (the smallest stride we keep)
        self._target_idx = self._fuse_indices[0]
        self._target_stride = feature_strides[self._target_idx]

        # Fusion: project each kept branch and upsample to stride-4 resolution
        self.upsample_layers = nn.ModuleDict()
        for i in self._fuse_indices:
            ch = feature_channels[i]
            layers = [
                nn.Conv2d(ch, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
            ]
            self.upsample_layers[str(i)] = nn.Sequential(*layers)

        # Final 3x3 conv after fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            features: (B, output_channels, H/4, W/4)
        """
        # Extract multi-scale features
        feature_maps = self.hrnet(x)  # list of tensors at different scales

        # Target spatial size: stride-4 feature map (H/4, W/4)
        target_h, target_w = feature_maps[self._target_idx].shape[2:]

        # Project each kept scale and upsample to stride-4 resolution, then sum
        fused = None
        for i in self._fuse_indices:
            projected = self.upsample_layers[str(i)](feature_maps[i])
            if projected.shape[2:] != (target_h, target_w):
                projected = F.interpolate(
                    projected,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                )
            if fused is None:
                fused = projected
            else:
                fused = fused + projected

        return self.fusion_conv(fused)


class DilatedResNetBackbone(nn.Module):
    """
    Pretrained ResNet-50 with dilated (atrous) convolutions.

    Replaces stride-2 in layer3 and layer4 with dilated convolutions so the
    output stays at 1/4 resolution (instead of the usual 1/32). This is the
    same approach used in DeepLab and DETR.

    Uses torchvision only (no extra dependencies).
    """

    def __init__(
            self,
            output_channels: int = 256,
            pretrained: bool = True,
    ):
        super().__init__()

        import torchvision.models as tv_models

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = tv_models.resnet50(weights=weights)

        # Keep layers up to 1/4 resolution (stem + layer1 = stride 4)
        self.stem = nn.Sequential(
            resnet.conv1,   # stride 2 -> 1/2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # stride 2 -> 1/4
        )
        self.layer1 = resnet.layer1  # 1/4, 256 ch

        # layer2: keep stride 2 -> 1/8
        self.layer2 = resnet.layer2  # 1/8, 512 ch

        # layer3: replace stride with dilation=2 -> stays 1/8
        self.layer3 = resnet.layer3  # 1/8 with dilation, 1024 ch
        self._replace_stride_with_dilation(self.layer3, dilation=2)

        # layer4: replace stride with dilation=4 -> stays 1/8
        self.layer4 = resnet.layer4  # 1/8 with dilation, 2048 ch
        self._replace_stride_with_dilation(self.layer4, dilation=4)

        # Project layer1 (1/4, 256ch) directly
        self.proj_layer1 = nn.Sequential(
            nn.Conv2d(256, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        # Project layer4 (1/8, 2048ch) and upsample to 1/4
        self.proj_layer4 = nn.Sequential(
            nn.Conv2d(2048, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _replace_stride_with_dilation(layer: nn.Module, dilation: int):
        """Replace stride-2 convolutions with dilation in a ResNet layer."""
        for name, module in layer.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                    module.dilation = (dilation, dilation)
                    module.padding = (dilation * (module.kernel_size[0] - 1) // 2,
                                      dilation * (module.kernel_size[1] - 1) // 2)
            # Also fix the downsample layer's stride
            if isinstance(module, nn.Sequential) and name == '0.downsample':
                for sub in module.modules():
                    if isinstance(sub, nn.Conv2d) and sub.stride == (2, 2):
                        sub.stride = (1, 1)

        # Fix the downsample in the first block directly
        if hasattr(layer[0], 'downsample') and layer[0].downsample is not None:
            for sub in layer[0].downsample.modules():
                if isinstance(sub, nn.Conv2d) and sub.stride == (2, 2):
                    sub.stride = (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            features: (B, output_channels, H/4, W/4)
        """
        x = self.stem(x)          # (B, 64, H/4, W/4)
        c1 = self.layer1(x)       # (B, 256, H/4, W/4)
        c2 = self.layer2(c1)      # (B, 512, H/8, W/8)
        c3 = self.layer3(c2)      # (B, 1024, H/8, W/8)  dilated
        c4 = self.layer4(c3)      # (B, 2048, H/8, W/8)  dilated

        # Fuse high-res (layer1) with deep features (layer4)
        feat_high = self.proj_layer1(c1)  # (B, out_ch, H/4, W/4)
        feat_deep = self.proj_layer4(c4)  # (B, out_ch, H/8, W/8)
        feat_deep = F.interpolate(
            feat_deep,
            size=feat_high.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        return self.fusion_conv(feat_high + feat_deep)


# ---------- Original custom backbones (kept for backward compatibility) ----------

class HighResBackbone(nn.Module):
    """
    Custom high-resolution feature extractor (HRNet-style, no pretrained weights).
    Maintains 1/4 resolution instead of 1/32 to preserve thin line details.
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 64,
            output_channels: int = 256,
            num_stages: int = 3,
    ):
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

        self.transition1 = self._make_transition_layer([base_channels], [base_channels, base_channels * 2])
        self.stage2 = self._make_parallel_stages([base_channels, base_channels * 2], num_blocks=2)

        self.transition2 = self._make_transition_layer(
            [base_channels, base_channels * 2],
            [base_channels, base_channels * 2, base_channels * 4]
        )
        self.stage3 = self._make_parallel_stages(
            [base_channels, base_channels * 2, base_channels * 4],
            num_blocks=3
        )

        self.fusion = FusionModule(
            [base_channels, base_channels * 2, base_channels * 4],
            output_channels
        )

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels
            ))
        return nn.Sequential(*layers)

    def _make_parallel_stages(self, channels_list: List[int], num_blocks: int) -> nn.ModuleList:
        stages = nn.ModuleList()
        for channels in channels_list:
            stages.append(self._make_stage(channels, channels, num_blocks))
        return stages

    def _make_transition_layer(self, in_channels_list: List[int], out_channels_list: List[int]) -> nn.ModuleList:
        num_in = len(in_channels_list)
        num_out = len(out_channels_list)

        transition_layers = nn.ModuleList()

        for i in range(num_out):
            if i < num_in:
                if in_channels_list[i] != out_channels_list[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(in_channels_list[i], out_channels_list[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(out_channels_list[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(nn.Identity())
            else:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x1 = self.stage1(x)

        x_list = []
        for i, transition in enumerate(self.transition1):
            x_list.append(transition(x1))

        stage2_out = []
        for i, stage in enumerate(self.stage2):
            stage2_out.append(stage(x_list[i]))

        x_list = []
        for i, transition in enumerate(self.transition2):
            if i < len(stage2_out):
                x_list.append(transition(stage2_out[i]))
            else:
                x_list.append(transition(stage2_out[-1]))

        stage3_out = []
        for i, stage in enumerate(self.stage3):
            stage3_out.append(stage(x_list[i]))

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

        self.upsample_layers = nn.ModuleList()
        for i, in_channels in enumerate(in_channels_list):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            if i > 0:
                layers.append(nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=True))
            self.upsample_layers.append(nn.Sequential(*layers))

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        fused = 0
        for i, features in enumerate(features_list):
            fused = fused + self.upsample_layers[i](features)
        fused = self.fusion_conv(fused)
        return fused


class LightweightBackbone(nn.Module):
    """Simplified backbone for faster prototyping."""

    def __init__(
            self,
            in_channels: int = 3,
            output_channels: int = 256,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),

            nn.Conv2d(256, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

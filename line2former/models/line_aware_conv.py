"""
Line-Aware Convolutions
Specialized convolutions for detecting thin line structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LineAwareConv(nn.Module):
    """
    Convolution layer with oriented filters for line detection.
    Uses multiple oriented kernels to detect lines at different angles.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_orientations: int = 8,
            kernel_size: int = 7,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels per orientation
            num_orientations: Number of orientation angles (e.g., 8 for 0°, 45°, 90°, ...)
            kernel_size: Size of the convolutional kernel
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size

        # Create oriented convolutions
        self.oriented_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            for _ in range(num_orientations)
        ])

        # Initialize with oriented line filters
        self._initialize_oriented_kernels()

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * num_orientations, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_oriented_kernels(self):
        """Initialize kernels with oriented line patterns."""
        for i, conv in enumerate(self.oriented_convs):
            angle = i * (180.0 / self.num_orientations)  # 0°, 22.5°, 45°, ...
            kernel = self._create_line_kernel(angle)

            # Initialize the convolutional weights with the line kernel
            with torch.no_grad():
                for out_ch in range(self.out_channels):
                    for in_ch in range(self.in_channels):
                        conv.weight[out_ch, in_ch, :, :] = kernel

    def _create_line_kernel(self, angle_degrees: float) -> torch.Tensor:
        """
        Create a line detection kernel at a specific angle.

        Args:
            angle_degrees: Angle in degrees (0 = horizontal, 90 = vertical)

        Returns:
            kernel: (kernel_size, kernel_size) tensor
        """
        k = self.kernel_size
        kernel = torch.zeros(k, k)

        angle_rad = math.radians(angle_degrees)
        center = k // 2

        # Create a line through the center at the specified angle
        for i in range(k):
            for j in range(k):
                # Distance from the line passing through center at given angle
                x = j - center
                y = i - center

                # Distance to line: |ax + by| / sqrt(a^2 + b^2)
                # Line equation: y*cos(θ) - x*sin(θ) = 0
                distance = abs(y * math.cos(angle_rad) - x * math.sin(angle_rad))

                # Gaussian profile perpendicular to the line
                sigma = 0.8
                kernel[i, j] = math.exp(-(distance ** 2) / (2 * sigma ** 2))

        # Normalize
        kernel = kernel / kernel.sum()

        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            out: (B, out_channels, H, W)
        """
        # Apply all oriented convolutions
        oriented_features = []
        for conv in self.oriented_convs:
            oriented_features.append(conv(x))

        # Concatenate and fuse
        concat_features = torch.cat(oriented_features, dim=1)  # (B, out_channels * num_orientations, H, W)
        out = self.fusion(concat_features)

        return out


class RidgeDetector(nn.Module):
    """
    Ridge detection module for thin line extraction.
    Uses second derivatives (Hessian) to detect ridge-like structures.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Sobel filters for gradients
        self.register_buffer('sobel_x', self._get_sobel_kernel_x())
        self.register_buffer('sobel_y', self._get_sobel_kernel_y())

        # Process gradient information
        self.process = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),  # Ix, Iy, magnitude
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _get_sobel_kernel_x(self) -> torch.Tensor:
        """Sobel kernel for horizontal gradients."""
        kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        return kernel

    def _get_sobel_kernel_y(self) -> torch.Tensor:
        """Sobel kernel for vertical gradients."""
        kernel = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            ridges: (B, out_channels, H, W)
        """
        B, C, H, W = x.shape

        # Compute gradients for each channel
        grad_x_list = []
        grad_y_list = []

        for i in range(C):
            channel = x[:, i:i + 1, :, :]  # (B, 1, H, W)

            # Compute gradients using Sobel
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)

            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)

        grad_x = torch.cat(grad_x_list, dim=1)  # (B, C, H, W)
        grad_y = torch.cat(grad_y_list, dim=1)  # (B, C, H, W)

        # Gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Concatenate gradient information
        features = torch.cat([grad_x, grad_y, grad_magnitude], dim=1)  # (B, C*3, H, W)

        # Process
        ridges = self.process(features)

        return ridges


class LineAwareFeatureExtractor(nn.Module):
    """
    Combines oriented convolutions and ridge detection for line-aware features.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_orientations: int = 8,
    ):
        super().__init__()

        self.oriented_conv = LineAwareConv(in_channels, out_channels // 2, num_orientations)
        self.ridge_detector = RidgeDetector(in_channels, out_channels // 2)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            features: (B, out_channels, H, W) line-aware features
        """
        oriented_features = self.oriented_conv(x)  # (B, out_channels/2, H, W)
        ridge_features = self.ridge_detector(x)  # (B, out_channels/2, H, W)

        # Concatenate and fuse
        combined = torch.cat([oriented_features, ridge_features], dim=1)
        features = self.fusion(combined)

        return features
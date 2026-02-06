"""
Differentiable Line Renderer
Converts predicted polylines + widths to binary masks for supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableLineRenderer(nn.Module):
    """
    Renders polylines as binary masks using differentiable operations.
    Uses soft rasterization for gradient flow.
    """

    def __init__(
            self,
            image_size: tuple = (480, 640),
            sigma: float = 1.0,
    ):
        """
        Args:
            image_size: (H, W) output size
            sigma: Smoothing parameter for soft rasterization
        """
        super().__init__()

        self.image_size = image_size
        self.sigma = sigma

        # Create coordinate grids
        H, W = image_size
        y_coords = torch.linspace(0, 1, H).view(-1, 1).repeat(1, W)
        x_coords = torch.linspace(0, 1, W).view(1, -1).repeat(H, 1)

        # (H, W, 2)
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1)
        self.register_buffer('pixel_coords', pixel_coords)

    def forward(
            self,
            centerlines: torch.Tensor,
            widths: torch.Tensor,
            valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render polylines as soft binary masks.

        Args:
            centerlines: (B, N, P, 2) normalized polyline coordinates
            widths: (B, N, P) line widths in pixels
            valid_mask: (B, N) which lines are valid

        Returns:
            masks: (B, N, H, W) soft binary masks
        """
        B, N, P, _ = centerlines.shape
        H, W = self.image_size

        # Expand pixel coordinates: (1, 1, H, W, 2)
        pixel_coords = self.pixel_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)
        pixel_coords = pixel_coords.repeat(B, N, 1, 1, 1)  # (B, N, H, W, 2)

        # Initialize mask
        masks = torch.zeros(B, N, H, W, device=centerlines.device)

        # For each line segment
        for i in range(P - 1):
            p1 = centerlines[:, :, i, :]  # (B, N, 2)
            p2 = centerlines[:, :, i + 1, :]  # (B, N, 2)
            w = widths[:, :, i]  # (B, N)

            # Compute distance from each pixel to line segment
            dist = self._point_to_segment_distance(
                pixel_coords.view(B, N, H * W, 2),
                p1,
                p2
            )  # (B, N, H*W)

            dist = dist.view(B, N, H, W)

            # Convert width to normalized coordinates
            w_normalized = w / W  # Approximate normalization
            w_normalized = w_normalized.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)

            # Soft rasterization using Gaussian
            segment_mask = torch.exp(-(dist ** 2) / (2 * (w_normalized * self.sigma) ** 2))

            # Accumulate (max to avoid double-counting)
            masks = torch.max(masks, segment_mask)

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        masks = masks * valid_mask.float()

        return masks

    def _point_to_segment_distance(
            self,
            points: torch.Tensor,
            p1: torch.Tensor,
            p2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance from points to line segment.

        Args:
            points: (B, N, M, 2) query points
            p1: (B, N, 2) segment start
            p2: (B, N, 2) segment end

        Returns:
            distances: (B, N, M) distances
        """
        # Expand dimensions
        p1 = p1.unsqueeze(2)  # (B, N, 1, 2)
        p2 = p2.unsqueeze(2)  # (B, N, 1, 2)

        # Vector from p1 to p2
        v = p2 - p1  # (B, N, 1, 2)

        # Vector from p1 to points
        w = points - p1  # (B, N, M, 2)

        # Project w onto v
        c1 = (w * v).sum(dim=-1)  # (B, N, M)
        c2 = (v * v).sum(dim=-1)  # (B, N, 1)

        # Clamp to segment bounds
        t = torch.clamp(c1 / (c2 + 1e-8), 0, 1)  # (B, N, M)

        # Closest point on segment
        projection = p1 + t.unsqueeze(-1) * v  # (B, N, M, 2)

        # Distance to closest point
        dist = torch.norm(points - projection, dim=-1)  # (B, N, M)

        return dist

    def render_hard(
            self,
            centerlines: torch.Tensor,
            widths: torch.Tensor,
            valid_mask: torch.Tensor,
            threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Render hard binary masks (non-differentiable, for visualization).

        Args:
            centerlines: (B, N, P, 2)
            widths: (B, N, P)
            valid_mask: (B, N)
            threshold: Threshold for binarization

        Returns:
            masks: (B, N, H, W) binary masks
        """
        soft_masks = self.forward(centerlines, widths, valid_mask)
        hard_masks = (soft_masks > threshold).float()
        return hard_masks
"""
Differentiable Line Renderer
Converts predicted polylines + widths to binary masks for supervision.

Memory-efficient: processes one line at a time instead of materializing the
full (B, N, H, W) coordinate grid simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableLineRenderer(nn.Module):
    """
    Renders polylines as binary masks using differentiable operations.
    Uses soft rasterization for gradient flow.

    Memory strategy: iterates over lines (N dimension) and segments (P dimension)
    one at a time, keeping only a (B, H*W) distance buffer per line.  This reduces
    peak memory from O(B*N*H*W) to O(B*H*W).
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

        # Create coordinate grids once: (H*W, 2)
        H, W = image_size
        y_coords = torch.linspace(0, 1, H)
        x_coords = torch.linspace(0, 1, W)
        # grid_y, grid_x each (H, W)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # (H*W, 2) with [x, y] ordering to match centerline format
        pixel_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
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
        M = H * W  # number of pixels

        # pixel_coords: (M, 2) — shared across batch & lines
        pixel_coords = self.pixel_coords  # (M, 2)

        masks_list = []

        for n in range(N):
            # Extract this line across the batch: (B, P, 2) and (B, P)
            line_pts = centerlines[:, n]   # (B, P, 2)
            line_w = widths[:, n]          # (B, P)
            line_valid = valid_mask[:, n]  # (B,)

            # Accumulate the maximum Gaussian response across all segments
            # Start with zeros: (B, M)
            line_mask = torch.zeros(B, M, device=centerlines.device)

            for i in range(P - 1):
                p1 = line_pts[:, i, :]      # (B, 2)
                p2 = line_pts[:, i + 1, :]  # (B, 2)
                w = line_w[:, i]            # (B,)

                # Compute distance from every pixel to this segment
                dist = self._point_to_segment_distance_batched(
                    pixel_coords, p1, p2
                )  # (B, M)

                # Normalized width (convert pixel width to [0,1] coordinate space)
                # Use geometric mean of H and W for non-square images
                scale = (H * W) ** 0.5
                w_normalized = w / scale  # (B,)
                w_normalized = w_normalized.unsqueeze(-1)  # (B, 1)

                # Soft rasterization via Gaussian
                segment_mask = torch.exp(
                    -(dist ** 2) / (2.0 * (w_normalized * self.sigma) ** 2 + 1e-8)
                )  # (B, M)

                line_mask = torch.max(line_mask, segment_mask)

            # Mask out invalid lines: (B,) -> (B, 1)
            line_mask = line_mask * line_valid.float().unsqueeze(-1)

            # Reshape to (B, H, W) and collect
            masks_list.append(line_mask.view(B, H, W))

        # Stack along line dimension: (B, N, H, W)
        masks = torch.stack(masks_list, dim=1)
        return masks

    @staticmethod
    def _point_to_segment_distance_batched(
            pixels: torch.Tensor,
            p1: torch.Tensor,
            p2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance from every pixel to a line segment, batched over B.

        Args:
            pixels: (M, 2) pixel coordinates (shared across batch)
            p1: (B, 2) segment start
            p2: (B, 2) segment end

        Returns:
            distances: (B, M)
        """
        # Expand: p1 (B, 1, 2), p2 (B, 1, 2), pixels (1, M, 2)
        p1 = p1.unsqueeze(1)       # (B, 1, 2)
        p2 = p2.unsqueeze(1)       # (B, 1, 2)
        px = pixels.unsqueeze(0)   # (1, M, 2)

        # Segment direction
        v = p2 - p1                # (B, 1, 2)

        # Vector from p1 to each pixel
        w = px - p1                # (B, M, 2)

        # Project onto segment: t = dot(w, v) / dot(v, v), clamped to [0, 1]
        c1 = (w * v).sum(dim=-1)           # (B, M)
        c2 = (v * v).sum(dim=-1)           # (B, 1)
        t = torch.clamp(c1 / (c2 + 1e-8), 0.0, 1.0)  # (B, M)

        # Closest point on segment
        projection = p1 + t.unsqueeze(-1) * v  # (B, M, 2)

        # Euclidean distance
        dist = torch.norm(px - projection, dim=-1)  # (B, M)
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

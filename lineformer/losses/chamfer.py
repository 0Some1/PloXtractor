"""
Chamfer Distance for Polyline Matching
Measures the symmetric distance between two sets of points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChamferDistance(nn.Module):
    """
    Chamfer distance between two point sets.

    For two point sets P and Q:
    CD(P, Q) = (1/|P|) * Σ min ||p - q||² + (1/|Q|) * Σ min ||q - p||²
    """

    def __init__(self, reduction: str = 'mean', bidirectional: bool = True):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
            bidirectional: If True, compute symmetric chamfer distance
        """
        super().__init__()
        self.reduction = reduction
        self.bidirectional = bidirectional

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            pred_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Chamfer distance.

        Args:
            pred: (B, N, 2) or (N, 2) predicted points
            target: (B, M, 2) or (M, 2) target points
            pred_mask: (B, N) or (N,) optional mask for valid pred points
            target_mask: (B, M) or (M,) optional mask for valid target points

        Returns:
            distance: scalar or (B,) chamfer distances
        """
        # Handle 2D input
        squeeze_output = False
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            squeeze_output = True
            if pred_mask is not None:
                pred_mask = pred_mask.unsqueeze(0)
            if target_mask is not None:
                target_mask = target_mask.unsqueeze(0)

        B, N, _ = pred.shape
        M = target.shape[1]
        device = pred.device

        # Compute pairwise distances: (B, N, M)
        dist_matrix = self._pairwise_distance(pred, target)

        # Forward direction: pred -> target
        # For each predicted point, find nearest target point
        if target_mask is not None:
            # Set masked distances to infinity
            dist_matrix_forward = dist_matrix.clone()
            dist_matrix_forward = dist_matrix_forward.masked_fill(
                ~target_mask.unsqueeze(1), float('inf')
            )
            dist_pred_to_target = dist_matrix_forward.min(dim=2)[0]  # (B, N)
        else:
            dist_pred_to_target = dist_matrix.min(dim=2)[0]  # (B, N)

        # Apply pred mask and compute mean
        if pred_mask is not None:
            dist_pred_to_target = dist_pred_to_target.masked_fill(~pred_mask, 0.0)
            num_pred = pred_mask.sum(dim=1).clamp(min=1).float()
            forward_dist = dist_pred_to_target.sum(dim=1) / num_pred
        else:
            forward_dist = dist_pred_to_target.mean(dim=1)

        if not self.bidirectional:
            return self._apply_reduction(forward_dist)

        # Backward direction: target -> pred
        if pred_mask is not None:
            dist_matrix_backward = dist_matrix.clone()
            dist_matrix_backward = dist_matrix_backward.masked_fill(
                ~pred_mask.unsqueeze(2), float('inf')
            )
            dist_target_to_pred = dist_matrix_backward.min(dim=1)[0]  # (B, M)
        else:
            dist_target_to_pred = dist_matrix.min(dim=1)[0]  # (B, M)

        # Apply target mask and compute mean
        if target_mask is not None:
            dist_target_to_pred = dist_target_to_pred.masked_fill(~target_mask, 0.0)
            num_target = target_mask.sum(dim=1).clamp(min=1).float()
            backward_dist = dist_target_to_pred.sum(dim=1) / num_target
        else:
            backward_dist = dist_target_to_pred.mean(dim=1)

        # Symmetric chamfer distance
        chamfer_dist = forward_dist + backward_dist

        return self._apply_reduction(chamfer_dist)

    def _apply_reduction(self, dist: torch.Tensor) -> torch.Tensor:
        """Apply reduction operation."""
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        else:
            return dist

    def _pairwise_distance(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise Euclidean distance.

        Args:
            x: (B, N, D)
            y: (B, M, D)

        Returns:
            dist: (B, N, M)
        """
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(2)  # (B, N, 1, D)
        y_expanded = y.unsqueeze(1)  # (B, 1, M, D)

        # Compute squared Euclidean distance
        dist = torch.sum((x_expanded - y_expanded) ** 2, dim=-1)  # (B, N, M)

        return dist

def chamfer_distance(
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        bidirectional: bool = True,
) -> torch.Tensor:
    """
    Functional interface for Chamfer distance.

    Args:
        pred: (B, N, 2) predicted points
        target: (B, M, 2) target points
        pred_mask: (B, N) mask for valid pred points
        target_mask: (B, M) mask for valid target points
        reduction: 'mean', 'sum', or 'none'
        bidirectional: If True, compute symmetric distance

    Returns:
        distance: Chamfer distance
    """
    cd = ChamferDistance(reduction=reduction, bidirectional=bidirectional)
    return cd(pred, target, pred_mask, target_mask)


class PolylineChamferLoss(nn.Module):
    """
    Chamfer distance loss specifically for polylines.
    Handles batched polylines with variable lengths.
    """

    def __init__(
            self,
            reduction: str = 'mean',
            normalize: bool = True,
    ):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
            normalize: Whether to normalize by image size
        """
        super().__init__()
        self.reduction = reduction
        self.normalize = normalize
        self.chamfer = ChamferDistance(reduction='none', bidirectional=True)

    def forward(
            self,
            pred_lines: torch.Tensor,
            target_lines: torch.Tensor,
            pred_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute chamfer loss for batched polylines.

        Args:
            pred_lines: (B, N_pred, P, 2) predicted polylines
            target_lines: (B, N_target, P, 2) target polylines
            pred_mask: (B, N_pred) mask for valid predicted lines
            target_mask: (B, N_target) mask for valid target lines

        Returns:
            loss: Chamfer distance loss
        """
        B, N_pred, P, _ = pred_lines.shape
        N_target = target_lines.shape[1]

        # Compute pairwise chamfer distances between all pred-target pairs
        # (B, N_pred, N_target)
        pairwise_chamfer = torch.zeros(B, N_pred, N_target, device=pred_lines.device)

        for i in range(N_pred):
            for j in range(N_target):
                # Extract single polylines: (B, P, 2)
                pred_poly = pred_lines[:, i, :, :]
                target_poly = target_lines[:, j, :, :]

                # Compute chamfer distance: (B,)
                dist = self.chamfer(pred_poly, target_poly)
                pairwise_chamfer[:, i, j] = dist

        # Apply masks
        if pred_mask is not None:
            # Set invalid predictions to large value
            pairwise_chamfer = pairwise_chamfer.masked_fill(
                ~pred_mask.unsqueeze(2), float('inf')
            )

        if target_mask is not None:
            # Set invalid targets to large value
            pairwise_chamfer = pairwise_chamfer.masked_fill(
                ~target_mask.unsqueeze(1), float('inf')
            )

        # For each prediction, find best matching target (minimum chamfer)
        min_chamfer, _ = pairwise_chamfer.min(dim=2)  # (B, N_pred)

        # Filter out invalid predictions
        if pred_mask is not None:
            min_chamfer = min_chamfer * pred_mask.float()
            num_valid = pred_mask.sum(dim=1).clamp(min=1)
            loss = min_chamfer.sum(dim=1) / num_valid
        else:
            loss = min_chamfer.mean(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HausdorffDistance(nn.Module):
    """
    Hausdorff distance between two point sets.
    More sensitive to outliers than Chamfer distance.
    """

    def __init__(self, reduction: str = 'mean', bidirectional: bool = True):
        super().__init__()
        self.reduction = reduction
        self.bidirectional = bidirectional

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            pred_mask: Optional[torch.Tensor] = None,
            target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Hausdorff distance.

        Args:
            pred: (B, N, 2) predicted points
            target: (B, M, 2) target points
            pred_mask: (B, N) mask for valid pred points
            target_mask: (B, M) mask for valid target points

        Returns:
            distance: Hausdorff distance
        """
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            if pred_mask is not None:
                pred_mask = pred_mask.unsqueeze(0)
            if target_mask is not None:
                target_mask = target_mask.unsqueeze(0)

        # Compute pairwise distances
        dist_matrix = self._pairwise_distance(pred, target)

        # Forward: max of min distances from pred to target
        if target_mask is not None:
            dist_matrix_forward = dist_matrix.clone()
            dist_matrix_forward = dist_matrix_forward.masked_fill(
                ~target_mask.unsqueeze(1), float('inf')
            )
            min_dist_pred_to_target = dist_matrix_forward.min(dim=2)[0]
        else:
            min_dist_pred_to_target = dist_matrix.min(dim=2)[0]

        if pred_mask is not None:
            min_dist_pred_to_target = min_dist_pred_to_target.masked_fill(
                ~pred_mask, float('-inf')
            )

        forward_hausdorff = min_dist_pred_to_target.max(dim=1)[0]

        if not self.bidirectional:
            return self._apply_reduction(forward_hausdorff)

        # Backward: max of min distances from target to pred
        if pred_mask is not None:
            dist_matrix_backward = dist_matrix.clone()
            dist_matrix_backward = dist_matrix_backward.masked_fill(
                ~pred_mask.unsqueeze(2), float('inf')
            )
            min_dist_target_to_pred = dist_matrix_backward.min(dim=1)[0]
        else:
            min_dist_target_to_pred = dist_matrix.min(dim=1)[0]

        if target_mask is not None:
            min_dist_target_to_pred = min_dist_target_to_pred.masked_fill(
                ~target_mask, float('-inf')
            )

        backward_hausdorff = min_dist_target_to_pred.max(dim=1)[0]

        # Symmetric Hausdorff
        hausdorff = torch.max(forward_hausdorff, backward_hausdorff)

        return self._apply_reduction(hausdorff)

    def _apply_reduction(self, dist: torch.Tensor) -> torch.Tensor:
        """Apply reduction operation."""
        if self.reduction == 'mean':
            return dist.mean()
        elif self.reduction == 'sum':
            return dist.sum()
        else:
            return dist

    def _pairwise_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distance."""
        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)
        return torch.sqrt(torch.sum((x_expanded - y_expanded) ** 2, dim=-1) + 1e-8)
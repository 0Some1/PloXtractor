"""
Hungarian Matcher for Optimal Line Assignment
Based on DETR's matcher implementation.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List, Optional
import numpy as np

from .chamfer import chamfer_distance


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for optimal bipartite matching between predictions and targets.

    Computes assignment cost based on:
    - Chamfer distance between centerlines
    - Width difference
    - Objectness score
    """

    def __init__(
            self,
            cost_centerline: float = 5.0,
            cost_width: float = 1.0,
            cost_objectness: float = 1.0,
    ):
        """
        Args:
            cost_centerline: Weight for centerline chamfer distance
            cost_width: Weight for width difference
            cost_objectness: Weight for objectness classification
        """
        super().__init__()
        self.cost_centerline = cost_centerline
        self.cost_width = cost_width
        self.cost_objectness = cost_objectness

    @torch.no_grad()
    def forward(
            self,
            pred_centerlines: torch.Tensor,
            pred_widths: torch.Tensor,
            pred_objectness: torch.Tensor,
            target_centerlines: torch.Tensor,
            target_widths: torch.Tensor,
            target_valid_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute optimal matching between predictions and targets.

        Args:
            pred_centerlines: (B, N_pred, P, 2) predicted polylines
            pred_widths: (B, N_pred, P) predicted widths
            pred_objectness: (B, N_pred) objectness scores (logits)
            target_centerlines: (B, N_target, P, 2) target polylines
            target_widths: (B, N_target, P) target widths
            target_valid_mask: (B, N_target) which targets are valid

        Returns:
            indices: List of (pred_indices, target_indices) for each batch
                     where pred_indices and target_indices are LongTensors
        """
        B, N_pred = pred_centerlines.shape[:2]
        N_target = target_centerlines.shape[1]

        # Compute objectness probability
        pred_prob = pred_objectness.sigmoid()  # (B, N_pred)

        # Compute cost matrices for each sample in batch
        indices = []

        for b in range(B):
            # Get valid targets for this sample
            valid_targets = target_valid_mask[b]  # (N_target,)
            num_valid = valid_targets.sum().item()

            if num_valid == 0:
                # No valid targets, return empty matching
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_centerlines.device),
                    torch.tensor([], dtype=torch.long, device=pred_centerlines.device)
                ))
                continue

            # Extract valid targets
            valid_target_centerlines = target_centerlines[b][valid_targets]  # (num_valid, P, 2)
            valid_target_widths = target_widths[b][valid_targets]  # (num_valid, P)

            # Compute cost matrix: (N_pred, num_valid)
            cost_matrix = torch.zeros(N_pred, num_valid, device=pred_centerlines.device)

            # 1. Centerline cost (Chamfer distance)
            for i in range(N_pred):
                for j in range(num_valid):
                    chamfer = chamfer_distance(
                        pred_centerlines[b, i],  # (P, 2)
                        valid_target_centerlines[j],  # (P, 2)
                        reduction='mean',
                        bidirectional=True,
                    )
                    cost_matrix[i, j] += self.cost_centerline * chamfer

            # 2. Width cost (L1 distance)
            if self.cost_width > 0:
                pred_widths_b = pred_widths[b].unsqueeze(1)  # (N_pred, 1, P)
                target_widths_b = valid_target_widths.unsqueeze(0)  # (1, num_valid, P)
                width_cost = torch.abs(pred_widths_b - target_widths_b).mean(dim=2)  # (N_pred, num_valid)
                cost_matrix += self.cost_width * width_cost

            # 3. Objectness cost (binary cross entropy)
            if self.cost_objectness > 0:
                # For matched pairs, we want objectness to be high (close to 1)
                # Cost is -log(prob) for positive class
                objectness_cost = -torch.log(pred_prob[b] + 1e-8).unsqueeze(1)  # (N_pred, 1)
                objectness_cost = objectness_cost.repeat(1, num_valid)  # (N_pred, num_valid)
                cost_matrix += self.cost_objectness * objectness_cost

            # Solve assignment problem using Hungarian algorithm
            cost_matrix_np = cost_matrix.cpu().numpy()
            pred_indices, target_indices = linear_sum_assignment(cost_matrix_np)

            # Convert back to torch tensors
            pred_indices = torch.as_tensor(pred_indices, dtype=torch.long, device=pred_centerlines.device)

            # Map back to original target indices (accounting for valid mask)
            valid_indices = torch.where(valid_targets)[0]
            target_indices = valid_indices[
                torch.as_tensor(target_indices, dtype=torch.long, device=pred_centerlines.device)]

            indices.append((pred_indices, target_indices))

        return indices


class GreedyMatcher(nn.Module):
    """
    Greedy matcher (faster alternative to Hungarian for prototyping).
    """

    def __init__(self, cost_centerline: float = 5.0):
        super().__init__()
        self.cost_centerline = cost_centerline

    @torch.no_grad()
    def forward(
            self,
            pred_centerlines: torch.Tensor,
            target_centerlines: torch.Tensor,
            target_valid_mask: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Greedy matching: for each target, find closest prediction.
        """
        B, N_pred = pred_centerlines.shape[:2]
        N_target = target_centerlines.shape[1]

        indices = []

        for b in range(B):
            valid_targets = target_valid_mask[b]
            num_valid = valid_targets.sum().item()

            if num_valid == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_centerlines.device),
                    torch.tensor([], dtype=torch.long, device=pred_centerlines.device)
                ))
                continue

            # Compute distances for all pairs
            valid_target_centerlines = target_centerlines[b][valid_targets]
            distances = torch.zeros(N_pred, num_valid, device=pred_centerlines.device)

            for i in range(N_pred):
                for j in range(num_valid):
                    distances[i, j] = chamfer_distance(
                        pred_centerlines[b, i],
                        valid_target_centerlines[j],
                        reduction='mean',
                    )

            # Greedy assignment
            pred_indices = []
            target_indices_local = []
            used_preds = set()

            for j in range(num_valid):
                # Find best unused prediction for this target
                available_dists = distances[:, j].clone()
                available_dists[list(used_preds)] = float('inf')

                best_pred = available_dists.argmin().item()
                if available_dists[best_pred] < float('inf'):
                    pred_indices.append(best_pred)
                    target_indices_local.append(j)
                    used_preds.add(best_pred)

            # Convert to tensors and map back
            if len(pred_indices) > 0:
                pred_indices = torch.tensor(pred_indices, dtype=torch.long, device=pred_centerlines.device)
                valid_indices = torch.where(valid_targets)[0]
                target_indices_global = valid_indices[
                    torch.tensor(target_indices_local, dtype=torch.long, device=pred_centerlines.device)]
                indices.append((pred_indices, target_indices_global))
            else:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_centerlines.device),
                    torch.tensor([], dtype=torch.long, device=pred_centerlines.device)
                ))

        return indices
"""
Composite Loss for Line2Former
Combines multiple loss components with Hungarian matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .chamfer import chamfer_distance, ChamferDistance
from .matcher import HungarianMatcher


class Line2FormerLoss(nn.Module):
    """
    Complete loss function for Line2Former training.

    Components:
    1. Centerline loss (Chamfer distance)
    2. Width loss (L1 distance)
    3. Objectness loss (BCE)
    4. Optional: Mask loss (Dice + BCE)
    """

    def __init__(
        self,
        # Loss weights
        weight_centerline: float = 5.0,
        weight_width: float = 1.0,
        weight_objectness: float = 2.0,
        weight_mask: float = 2.0,

        # Matcher config
        matcher_cost_centerline: float = 5.0,
        matcher_cost_width: float = 1.0,
        matcher_cost_objectness: float = 1.0,

        # Other settings
        num_queries: int = 20,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            weight_centerline: Weight for centerline Chamfer loss
            weight_width: Weight for width L1 loss
            weight_objectness: Weight for objectness classification loss
            weight_mask: Weight for mask rendering loss
            matcher_cost_*: Costs for Hungarian matching
            num_queries: Number of line queries
            use_focal_loss: Use focal loss for objectness instead of BCE
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()

        self.weight_centerline = weight_centerline
        self.weight_width = weight_width
        self.weight_objectness = weight_objectness
        self.weight_mask = weight_mask

        self.num_queries = num_queries
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Matcher for optimal assignment
        self.matcher = HungarianMatcher(
            cost_centerline=matcher_cost_centerline,
            cost_width=matcher_cost_width,
            cost_objectness=matcher_cost_objectness,
        )

        # Loss components
        self.chamfer = ChamferDistance(reduction='none', bidirectional=True)

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        """
        pred_centerlines = outputs['centerlines']
        pred_widths = outputs['widths']
        pred_objectness = outputs['objectness']

        target_centerlines = targets['centerlines']
        target_widths = targets['widths']
        target_valid_mask = targets['valid_mask']

        B = pred_centerlines.shape[0]

        # 1. Compute optimal matching
        indices = self.matcher(
            pred_centerlines,
            pred_widths,
            pred_objectness,
            target_centerlines,
            target_widths,
            target_valid_mask,
        )

        # 2. Compute losses using matched pairs
        loss_centerline = self._centerline_loss(
            pred_centerlines, target_centerlines, indices
        )

        loss_width = self._width_loss(
            pred_widths, target_widths, indices
        )

        loss_objectness = self._objectness_loss(
            pred_objectness, target_valid_mask, indices
        )

        # 3. Optional: Mask loss
        if 'masks' in outputs and 'masks' in targets:
            loss_mask = self._mask_loss(
                outputs['masks'], targets['masks'], indices
            )
        else:
            # Create zero loss connected to computation graph
            loss_mask = pred_centerlines.sum() * 0.0

        # 4. Total weighted loss
        loss_total = (
                self.weight_centerline * loss_centerline +
                self.weight_width * loss_width +
                self.weight_objectness * loss_objectness +
                self.weight_mask * loss_mask
        )

        # Return dictionary (keep all gradients attached)
        loss_dict = {
            'loss_total': loss_total,
            'loss_centerline': loss_centerline,
            'loss_width': loss_width,
            'loss_objectness': loss_objectness,
            'loss_mask': loss_mask,
        }

        return loss_dict

    def _centerline_loss(
            self,
            pred_centerlines: torch.Tensor,
            target_centerlines: torch.Tensor,
            indices: list,
    ) -> torch.Tensor:
        """
        Compute Chamfer distance loss for matched centerlines.

        Args:
            pred_centerlines: (B, N_pred, P, 2)
            target_centerlines: (B, N_target, P, 2)
            indices: List of (pred_idx, target_idx) tuples

        Returns:
            loss: Scalar loss
        """
        losses = []

        for b, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            # Extract matched pairs
            matched_pred = pred_centerlines[b, pred_idx]  # (num_matched, P, 2)
            matched_target = target_centerlines[b, target_idx]  # (num_matched, P, 2)

            # Compute Chamfer distance for each pair
            for i in range(len(pred_idx)):
                chamfer = chamfer_distance(
                    matched_pred[i],  # (P, 2)
                    matched_target[i],  # (P, 2)
                    reduction='mean',
                    bidirectional=True,
                )
                losses.append(chamfer)

        if len(losses) == 0:
            # Return zero loss that's part of computation graph
            return pred_centerlines.sum() * 0.0

        return torch.stack(losses).mean()

    def _width_loss(
            self,
            pred_widths: torch.Tensor,
            target_widths: torch.Tensor,
            indices: list,
    ) -> torch.Tensor:
        """
        Compute L1 loss for matched line widths.

        Args:
            pred_widths: (B, N_pred, P)
            target_widths: (B, N_target, P)
            indices: List of (pred_idx, target_idx) tuples

        Returns:
            loss: Scalar loss
        """
        losses = []

        for b, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            matched_pred = pred_widths[b, pred_idx]  # (num_matched, P)
            matched_target = target_widths[b, target_idx]  # (num_matched, P)

            # L1 loss
            loss = F.l1_loss(matched_pred, matched_target, reduction='mean')
            losses.append(loss)

        if len(losses) == 0:
            # Return zero loss that's part of computation graph
            return pred_widths.sum() * 0.0

        return torch.stack(losses).mean()

    def _objectness_loss(
            self,
            pred_objectness: torch.Tensor,
            target_valid_mask: torch.Tensor,
            indices: list,
    ) -> torch.Tensor:
        """
        Compute objectness classification loss.

        Matched predictions should have objectness=1,
        Unmatched predictions should have objectness=0.

        Args:
            pred_objectness: (B, N_pred) logits
            target_valid_mask: (B, N_target) which targets exist
            indices: List of (pred_idx, target_idx) tuples

        Returns:
            loss: Scalar loss
        """
        B, N_pred = pred_objectness.shape

        # Create target labels: 1 for matched, 0 for unmatched
        target_labels = torch.zeros_like(pred_objectness)  # (B, N_pred)

        for b, (pred_idx, target_idx) in enumerate(indices):
            target_labels[b, pred_idx] = 1.0

        if self.use_focal_loss:
            # Focal loss for handling class imbalance
            loss = self._focal_loss(pred_objectness, target_labels)
        else:
            # Standard binary cross entropy
            loss = F.binary_cross_entropy_with_logits(
                pred_objectness,
                target_labels,
                reduction='mean',
            )

        return loss

    def _focal_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        probs = torch.sigmoid(logits)

        # Compute focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.focal_gamma

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Apply focal weights
        focal_loss = focal_weight * bce_loss

        return focal_loss.mean()

    def _mask_loss(
            self,
            pred_masks: torch.Tensor,
            target_masks: torch.Tensor,
            indices: list,
    ) -> torch.Tensor:
        """
        Compute mask rendering loss (Dice + BCE).

        Args:
            pred_masks: (B, N_pred, H, W)
            target_masks: (B, N_target, H, W)
            indices: List of (pred_idx, target_idx) tuples

        Returns:
            loss: Scalar loss
        """
        dice_losses = []
        bce_losses = []

        for b, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue

            matched_pred = pred_masks[b, pred_idx]  # (num_matched, H, W)
            matched_target = target_masks[b, target_idx]  # (num_matched, H, W)

            # Dice loss
            dice = self._dice_loss(matched_pred, matched_target)
            dice_losses.append(dice)

            # BCE loss
            bce = F.binary_cross_entropy(
                matched_pred,
                matched_target,
                reduction='mean',
            )
            bce_losses.append(bce)

        if len(dice_losses) == 0:
            # Return zero loss that's part of computation graph
            return pred_masks.sum() * 0.0

        dice_loss = torch.stack(dice_losses).mean()
        bce_loss = torch.stack(bce_losses).mean()

        return dice_loss + bce_loss

    def _dice_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            smooth: float = 1.0,
    ) -> torch.Tensor:
        """
        Dice loss for segmentation.

        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        Loss = 1 - Dice
        """
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice.mean()


class SimplifiedLine2FormerLoss(nn.Module):
    """
    Simplified loss for quick prototyping.
    Uses greedy matching instead of Hungarian.
    """

    def __init__(
            self,
            weight_centerline: float = 5.0,
            weight_objectness: float = 2.0,
    ):
        super().__init__()

        self.weight_centerline = weight_centerline
        self.weight_objectness = weight_objectness

        self.chamfer = ChamferDistance(reduction='mean', bidirectional=True)

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Simplified loss computation.

        For each target, find the closest prediction and compute loss.
        """
        pred_centerlines = outputs['centerlines']
        pred_objectness = outputs['objectness']

        target_centerlines = targets['centerlines']
        target_valid_mask = targets['valid_mask']

        B = pred_centerlines.shape[0]

        # Centerline loss
        centerline_losses = []
        for b in range(B):
            valid_targets = target_valid_mask[b]
            if valid_targets.sum() == 0:
                continue

            valid_target_centerlines = target_centerlines[b][valid_targets]

            # For each target, find closest prediction
            for target_line in valid_target_centerlines:
                min_dist = None
                for pred_line in pred_centerlines[b]:
                    dist = chamfer_distance(
                        pred_line,
                        target_line,
                        reduction='mean',
                    )
                    if min_dist is None:
                        min_dist = dist
                    else:
                        min_dist = torch.min(min_dist, dist)

                if min_dist is not None:
                    centerline_losses.append(min_dist)

        if len(centerline_losses) > 0:
            loss_centerline = torch.stack(centerline_losses).mean()
        else:
            # Return zero loss connected to computation graph
            loss_centerline = pred_centerlines.sum() * 0.0

        # Objectness loss (simple: encourage at least one high-confidence prediction per valid target)
        num_valid_targets = target_valid_mask.sum(dim=1).float()  # (B,)

        # Encourage top-k predictions to have high objectness
        top_k_objectness, _ = torch.topk(
            pred_objectness,
            k=min(5, pred_objectness.shape[1]),
            dim=1,
        )

        loss_objectness = F.binary_cross_entropy_with_logits(
            top_k_objectness,
            torch.ones_like(top_k_objectness),
            reduction='mean',
        )

        # Total loss
        loss_total = (
                self.weight_centerline * loss_centerline +
                self.weight_objectness * loss_objectness
        )

        return {
            'loss_total': loss_total,
            'loss_centerline': loss_centerline,
            'loss_objectness': loss_objectness,
        }


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for intermediate decoder layers (like in DETR).
    """

    def __init__(self, main_loss: nn.Module, weight: float = 1.0):
        super().__init__()
        self.main_loss = main_loss
        self.weight = weight

    def forward(
            self,
            outputs_list: list,
            targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for each decoder layer output.

        Args:
            outputs_list: List of output dicts from each decoder layer
            targets: Ground truth targets

        Returns:
            loss_dict: Combined losses from all layers
        """
        total_losses = {}

        # Compute loss for each layer
        for i, outputs in enumerate(outputs_list):
            layer_losses = self.main_loss(outputs, targets)

            # Add layer index to keys
            for key, value in layer_losses.items():
                total_losses[f'{key}_layer{i}'] = value * self.weight

        # Sum all losses
        loss_total = sum(
            loss for key, loss in total_losses.items()
            if key.startswith('loss_total')
        )

        total_losses['loss_total'] = loss_total

        return total_losses
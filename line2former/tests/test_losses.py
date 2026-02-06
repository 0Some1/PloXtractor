"""
Test Loss Functions
"""

import torch
from line2former.losses import ChamferDistance, HungarianMatcher, Line2FormerLoss, SimplifiedLine2FormerLoss


def test_chamfer_distance():
    """Test Chamfer distance computation."""
    print("=" * 60)
    print("Testing Chamfer Distance")
    print("=" * 60)

    # Create synthetic data
    pred = torch.randn(2, 50, 2)  # (B, N, 2)
    target = torch.randn(2, 50, 2)

    # Test symmetric chamfer
    chamfer = ChamferDistance(reduction='mean', bidirectional=True)
    loss = chamfer(pred, target)

    print(f"Chamfer distance: {loss.item():.4f}")
    assert loss.item() >= 0, "Chamfer distance should be non-negative"

    # Test with masks
    pred_mask = torch.ones(2, 50, dtype=torch.bool)
    pred_mask[:, 40:] = False  # Mask last 10 points

    loss_masked = chamfer(pred, target, pred_mask=pred_mask)
    print(f"Masked Chamfer distance: {loss_masked.item():.4f}")

    print("✓ Chamfer distance test passed!\n")


def test_hungarian_matcher():
    """Test Hungarian matcher."""
    print("=" * 60)
    print("Testing Hungarian Matcher")
    print("=" * 60)

    B, N_pred, N_target, P = 2, 20, 5, 50

    # Create synthetic predictions and targets
    pred_centerlines = torch.randn(B, N_pred, P, 2) * 0.5 + 0.5  # [0, 1] range
    pred_widths = torch.rand(B, N_pred, P) * 5 + 1
    pred_objectness = torch.randn(B, N_pred)

    target_centerlines = torch.randn(B, N_target, P, 2) * 0.5 + 0.5
    target_widths = torch.rand(B, N_target, P) * 5 + 1
    target_valid_mask = torch.ones(B, N_target, dtype=torch.bool)
    target_valid_mask[:, 3:] = False  # Only first 3 targets are valid

    # Test matcher
    matcher = HungarianMatcher(
        cost_centerline=5.0,
        cost_width=1.0,
        cost_objectness=1.0,
    )

    indices = matcher(
        pred_centerlines,
        pred_widths,
        pred_objectness,
        target_centerlines,
        target_widths,
        target_valid_mask,
    )

    print(f"Number of batches: {len(indices)}")
    for b, (pred_idx, target_idx) in enumerate(indices):
        print(f"Batch {b}: {len(pred_idx)} matches")
        print(f"  Pred indices: {pred_idx.tolist()}")
        print(f"  Target indices: {target_idx.tolist()}")

    print("✓ Hungarian matcher test passed!\n")


def test_line2former_loss():
    """Test complete Line2Former loss."""
    print("=" * 60)
    print("Testing Line2Former Loss")
    print("=" * 60)

    B, N_queries, N_targets, P = 2, 20, 5, 50
    H, W = 480, 640

    # Create synthetic outputs - ADD requires_grad=True
    outputs = {
        'centerlines': torch.randn(B, N_queries, P, 2, requires_grad=True) * 0.5 + 0.5,
        'widths': torch.rand(B, N_queries, P, requires_grad=True) * 5 + 1,
        'objectness': torch.randn(B, N_queries, requires_grad=True),
        'masks': torch.rand(B, N_queries, H, W, requires_grad=True),
    }

    # Create synthetic targets (no grad needed)
    targets = {
        'centerlines': torch.randn(B, N_targets, P, 2) * 0.5 + 0.5,
        'widths': torch.rand(B, N_targets, P) * 5 + 1,
        'valid_mask': torch.ones(B, N_targets, dtype=torch.bool),
        'masks': torch.rand(B, N_targets, H, W),
    }
    targets['valid_mask'][:, 3:] = False  # Only first 3 targets valid

    # Test full loss
    loss_fn = Line2FormerLoss(
        weight_centerline=5.0,
        weight_width=1.0,
        weight_objectness=2.0,
        weight_mask=2.0,
    )

    loss_dict = loss_fn(outputs, targets)

    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    # Verify total loss is reasonable
    assert loss_dict['loss_total'].item() > 0, "Total loss should be positive"
    assert loss_dict['loss_total'].requires_grad, "Total loss should have gradients"

    # Test backward pass
    loss_dict['loss_total'].backward()

    print("✓ Backward pass successful")
    print("✓ Line2Former loss test passed!\n")


def test_simplified_loss():
    """Test simplified loss."""
    print("=" * 60)
    print("Testing Simplified Loss")
    print("=" * 60)

    B, N_queries, N_targets, P = 2, 20, 5, 50

    # ADD requires_grad=True
    outputs = {
        'centerlines': torch.randn(B, N_queries, P, 2, requires_grad=True) * 0.5 + 0.5,
        'objectness': torch.randn(B, N_queries, requires_grad=True),
    }

    targets = {
        'centerlines': torch.randn(B, N_targets, P, 2) * 0.5 + 0.5,
        'valid_mask': torch.ones(B, N_targets, dtype=torch.bool),
    }
    targets['valid_mask'][:, 3:] = False

    loss_fn = SimplifiedLine2FormerLoss()
    loss_dict = loss_fn(outputs, targets)

    print("Simplified loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    loss_dict['loss_total'].backward()

    print("✓ Simplified loss test passed!\n")


if __name__ == '__main__':
    test_chamfer_distance()
    test_hungarian_matcher()
    test_line2former_loss()
    test_simplified_loss()

    print("=" * 60)
    print("All loss tests passed! ✓")
    print("=" * 60)
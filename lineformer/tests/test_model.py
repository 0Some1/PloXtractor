"""
Test LineFormer Model
"""

import torch
from models import LineFormer, LineFormerLite


def test_lineformer():
    """Test full LineFormer model."""
    print("=" * 60)
    print("Testing LineFormer")
    print("=" * 60)

    # Create model
    model = LineFormer(
        num_queries=20,
        max_points=50,
        d_model=256,
        backbone_type='lightweight',
        num_decoder_layers=6,
        image_size=(480, 640),
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 480, 640)

    print("Input shape:", images.shape)
    print()

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images, return_features=True)

    # Check outputs
    print("Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    print()

    # Test prediction mode
    predictions = model.predict(images, objectness_threshold=0.5)
    print("Predictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\n✓ LineFormer test passed!")


def test_lineformer_lite():
    """Test lightweight LineFormer model."""
    print("\n" + "=" * 60)
    print("Testing LineFormerLite")
    print("=" * 60)

    # Create model
    model = LineFormerLite(
        num_queries=20,
        max_points=50,
        d_model=128,
        num_decoder_layers=3,
        image_size=(480, 640),
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 480, 640)

    print("Input shape:", images.shape)
    print()

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images)

    # Check outputs
    print("Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print("\n✓ LineFormerLite test passed!")


if __name__ == '__main__':
    test_lineformer()
    test_lineformer_lite()

    print("\n" + "=" * 60)
    print("All model tests passed! ✓")
    print("=" * 60)
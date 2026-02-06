"""
Test Line2Former Model
"""

import torch
from line2former.models import Line2Former, Line2FormerLite


def test_line2former():
    """Test full Line2Former model."""
    print("=" * 60)
    print("Testing Line2Former")
    print("=" * 60)

    # Create model
    model = Line2Former(
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

    print("\n✓ Line2Former test passed!")


def test_line2former_lite():
    """Test lightweight Line2Former model."""
    print("\n" + "=" * 60)
    print("Testing Line2FormerLite")
    print("=" * 60)

    # Create model
    model = Line2FormerLite(
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

    print("\n✓ Line2FormerLite test passed!")


if __name__ == '__main__':
    test_line2former()
    test_line2former_lite()

    print("\n" + "=" * 60)
    print("All model tests passed! ✓")
    print("=" * 60)
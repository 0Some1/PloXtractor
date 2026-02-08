"""
Test Line2Former Model
"""

import torch
from line2former.models import Line2Former, Line2FormerLite


def test_line2former(backbone_type='lightweight', pretrained=False):
    """Test full Line2Former model with a given backbone."""
    print("=" * 60)
    print(f"Testing Line2Former (backbone={backbone_type}, pretrained={pretrained})")
    print("=" * 60)

    # Create model
    model = Line2Former(
        num_queries=20,
        max_points=50,
        d_model=256,
        backbone_type=backbone_type,
        pretrained=pretrained,
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

    # Verify shapes
    assert outputs['centerlines'].shape == (batch_size, 20, 50, 2)
    assert outputs['widths'].shape == (batch_size, 20, 50)
    assert outputs['objectness'].shape == (batch_size, 20)
    assert outputs['masks'].shape == (batch_size, 20, 480, 640)

    # Test prediction mode
    predictions = model.predict(images, objectness_threshold=0.5)
    print("Predictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

    print(f"\n  Line2Former ({backbone_type}) test passed!")


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

    print("\n  Line2FormerLite test passed!")


if __name__ == '__main__':
    # Always test with lightweight (no dependencies needed)
    test_line2former(backbone_type='lightweight', pretrained=False)

    # Test with custom HRNet (no pretrained weights needed)
    test_line2former(backbone_type='hrnet', pretrained=False)

    # Test pretrained backbones if available
    try:
        import timm
        print("\n[timm available] Testing pretrained HRNet backbone...")
        test_line2former(backbone_type='hrnet_w32', pretrained=True)
    except ImportError:
        print("\n[timm not installed] Skipping pretrained HRNet test.")
        print("  Install with: pip install timm")

    try:
        import torchvision
        print("\n[torchvision available] Testing dilated ResNet-50 backbone...")
        test_line2former(backbone_type='dilated_resnet50', pretrained=True)
    except ImportError:
        print("\n[torchvision not installed] Skipping dilated ResNet-50 test.")

    test_line2former_lite()

    print("\n" + "=" * 60)
    print("All model tests passed!")
    print("=" * 60)

#!/usr/bin/env python3
"""
Quick Test Script
Generate a small dataset and visualize results to verify everything works
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("Synthetic Line Plot Dataset - Quick Test")
    print("=" * 60)
    
    # Test 1: Equation Bank
    print("\n[1/4] Testing Equation Bank...")
    try:
        from equation_bank import EquationBank
        import numpy as np
        
        bank = EquationBank(seed=42)
        x = np.linspace(-5, 5, 100)
        
        equations = bank.sample_multiple_equations(5, (-5, 5))
        print(f"  ✓ Sampled {len(equations)} equations:")
        for eq in equations:
            y = eq.func(x)
            print(f"    - {eq.category}: {eq.latex[:40]}...")
        
        print("  ✓ Equation Bank OK")
    except Exception as e:
        print(f"  ✗ Equation Bank FAILED: {e}")
        return False
    
    # Test 2: Style Bank
    print("\n[2/4] Testing Style Bank...")
    try:
        from style_bank import StyleBank
        
        style_bank = StyleBank(seed=42)
        plot_style, line_styles = style_bank.sample_complete_style(num_lines=3)
        
        print(f"  ✓ Plot style: {plot_style.figsize}, grid={plot_style.grid}")
        print(f"  ✓ Line styles: {len(line_styles)} styles")
        for ls in line_styles:
            print(f"    - color={ls.color}, style={ls.linestyle}, label={ls.label}")
        
        print("  ✓ Style Bank OK")
    except Exception as e:
        print(f"  ✗ Style Bank FAILED: {e}")
        return False
    
    # Test 3: Generate Small Dataset
    print("\n[3/4] Testing Dataset Generation...")
    try:
        from dataset_generator import LinePlotGenerator
        
        output_dir = Path("./test_output")
        
        generator = LinePlotGenerator(
            seed=42,
            mask_line_thickness=3,
            mask_threshold=0.5
        )
        
        # Generate small test set
        annotations_path = generator.generate_dataset(
            output_dir=str(output_dir),
            num_samples=10,  # Just 10 samples for quick test
            split="test",
            save_debug_masks=True,
            save_metadata=True,
            use_rle=False
        )
        
        print(f"  ✓ Generated 10 test samples")
        print(f"  ✓ Annotations saved to: {annotations_path}")
        print("  ✓ Dataset Generation OK")
    except Exception as e:
        print(f"  ✗ Dataset Generation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Visualization
    print("\n[4/4] Testing Visualization...")
    try:
        from visualize import visualize_grid, print_dataset_stats
        
        # Generate grid visualization
        visualize_grid(
            dataset_dir=str(output_dir),
            split="test",
            num_samples=9,
            grid_size=(3, 3),
            output_path=str(output_dir / "test_grid.png"),
            seed=42
        )
        
        print(f"  ✓ Grid visualization saved to: {output_dir / 'test_grid.png'}")
        
        # Print stats
        print_dataset_stats(str(output_dir), "test")
        
        print("  ✓ Visualization OK")
    except Exception as e:
        print(f"  ✗ Visualization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    print(f"\nTest output directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.relative_to(output_dir)} ({size_str})")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review test_grid.png to see generated samples")
    print("  2. Check debug_masks/ folder to see individual masks")
    print("  3. Run full generation with desired sample counts:")
    print()
    print("     python dataset_generator.py \\")
    print("         --output-dir ./SyntheticDataset \\")
    print("         --num-train 10000 \\")
    print("         --num-val 1000 \\")
    print("         --num-test 1000 \\")
    print("         --save-metadata")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

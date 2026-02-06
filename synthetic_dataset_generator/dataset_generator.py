"""
Synthetic Line Plot Dataset Generator
Generates line plots with perfect ground truth masks for instance segmentation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2
from pathlib import Path
import json
import random
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import argparse

from equation_bank import EquationBank, Equation
from style_bank import StyleBank, PlotStyle, LineStyle
from SyntheticDatasetGenerator.coco_utils import COCOAnnotationBuilder, compute_area


@dataclass
class GeneratedSample:
    """Represents a single generated sample"""
    image_path: str
    masks: List[np.ndarray]
    equations: List[Equation]
    line_styles: List[LineStyle]
    plot_style: PlotStyle
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    image_size: Tuple[int, int]  # (width, height)


class LinePlotGenerator:
    """
    Generator for synthetic line plot images with ground truth masks.
    """
    
    def __init__(
        self,
        seed: int = None,
        mask_line_thickness: int = 3,
        mask_threshold: float = 0.5,  # 50% for color bleeding
    ):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
            mask_line_thickness: Thickness of lines in mask (pixels)
            mask_threshold: Threshold for binarizing anti-aliased lines (0.5 = 50%)
        """
        self.seed = seed
        self.mask_line_thickness = mask_line_thickness
        self.mask_threshold = mask_threshold
        
        # Initialize banks
        self.equation_bank = EquationBank(seed=seed)
        self.style_bank = StyleBank(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _sample_x_range(self) -> Tuple[float, float]:
        """Sample a random x range for the plot"""
        # Various x range patterns
        patterns = [
            (-10, 10),
            (-5, 5),
            (0, 10),
            (-2 * np.pi, 2 * np.pi),
            (0, 2 * np.pi),
            (-1, 1),
            (0, 100),
            (-100, 100),
        ]
        
        base = random.choice(patterns)
        # Add slight variation
        variation = random.uniform(0.8, 1.2)
        return (base[0] * variation, base[1] * variation)
    
    def _compute_y_range(
        self,
        equations: List[Equation],
        x: np.ndarray,
        padding: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute appropriate y range that includes all lines.
        
        Args:
            equations: List of equations
            x: x values array
            padding: Fractional padding to add
            
        Returns:
            (y_min, y_max) tuple
        """
        all_y = []
        for eq in equations:
            try:
                y = eq.func(x)
                # Filter out inf and nan
                y_valid = y[np.isfinite(y)]
                if len(y_valid) > 0:
                    all_y.extend(y_valid)
            except Exception:
                continue
        
        if not all_y:
            return (-10, 10)
        
        y_min, y_max = min(all_y), max(all_y)
        
        # Add padding
        y_range = y_max - y_min
        if y_range < 1e-6:
            y_range = 1.0
        
        y_min -= padding * y_range
        y_max += padding * y_range
        
        return (y_min, y_max)
    
    def _render_plot_image(
        self,
        x: np.ndarray,
        equations: List[Equation],
        line_styles: List[LineStyle],
        plot_style: PlotStyle,
        y_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Render the complete plot image.
        
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        # Create figure
        fig, ax = plt.subplots(figsize=plot_style.figsize, dpi=plot_style.dpi)
        
        # Set background
        fig.patch.set_facecolor(plot_style.background_color)
        ax.set_facecolor(plot_style.background_color)
        
        # Plot each line
        for eq, ls in zip(equations, line_styles):
            try:
                y = eq.func(x)
                
                # Handle invalid values
                y = np.where(np.isfinite(y), y, np.nan)
                
                ax.plot(
                    x, y,
                    color=ls.color,
                    linestyle=ls.linestyle,
                    linewidth=ls.linewidth,
                    marker=ls.marker,
                    markersize=ls.markersize if ls.marker else 0,
                    alpha=ls.alpha,
                    label=ls.label
                )
            except Exception as e:
                print(f"Warning: Failed to plot {eq.name}: {e}")
        
        # Set axis limits
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_range)
        
        # Apply styling
        if plot_style.grid:
            ax.grid(
                True,
                alpha=plot_style.grid_alpha,
                linestyle=plot_style.grid_linestyle
            )
        
        if plot_style.title:
            ax.set_title(plot_style.title, fontsize=plot_style.title_fontsize)
        
        if plot_style.xlabel:
            ax.set_xlabel(plot_style.xlabel, fontsize=plot_style.axis_label_fontsize)
        if plot_style.ylabel:
            ax.set_ylabel(plot_style.ylabel, fontsize=plot_style.axis_label_fontsize)
        
        ax.tick_params(labelsize=plot_style.tick_label_fontsize)
        
        # Spine visibility
        for spine, visible in plot_style.spine_visible.items():
            ax.spines[spine].set_visible(visible)
        
        # Legend
        if plot_style.show_legend and any(ls.label for ls in line_styles):
            ax.legend(loc=plot_style.legend_loc)
        
        if plot_style.tight_layout:
            plt.tight_layout()
        
        # Render to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get RGB array
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        plt.close(fig)
        
        return image
    
    def _render_line_mask(
        self,
        x: np.ndarray,
        equation: Equation,
        plot_style: PlotStyle,
        line_style: LineStyle,
        y_range: Tuple[float, float],
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Render a binary mask for a single line.
        
        The key is to render the mask with EXACTLY the same figure setup
        as the main image, just with a black background and white line.
        
        Args:
            x: x values
            equation: The equation to render
            plot_style: Plot style (must match the main image exactly)
            line_style: Line style (for consistent line width)
            y_range: y axis limits
            image_size: (width, height) in pixels
            
        Returns:
            Binary mask (H, W) with values 0 or 255
        """
        width, height = image_size
        
        # Create figure with EXACTLY the same settings as the main image
        fig, ax = plt.subplots(figsize=plot_style.figsize, dpi=plot_style.dpi)
        
        # Black background everywhere
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Compute y values
        try:
            y = equation.func(x)
            y = np.where(np.isfinite(y), y, np.nan)
        except Exception:
            plt.close(fig)
            return np.zeros((height, width), dtype=np.uint8)
        
        # Plot line in white - use same linewidth as original for accurate mask
        ax.plot(
            x, y,
            color='white',
            linestyle='-',  # Always solid for mask
            linewidth=max(self.mask_line_thickness, line_style.linewidth),
            marker=None  # No markers in mask (as per user request)
        )
        
        # Set EXACT same axis limits as the main image
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_range)
        
        # IMPORTANT: Apply the SAME styling as main image to get same layout
        # This ensures the axes/plot area are in the exact same position
        
        # Keep tick labels but make them invisible (they affect layout!)
        ax.tick_params(labelsize=plot_style.tick_label_fontsize, colors='black')
        
        # Add invisible axis labels if the main image has them (affects layout!)
        if plot_style.xlabel:
            ax.set_xlabel(plot_style.xlabel, fontsize=plot_style.axis_label_fontsize, color='black')
        if plot_style.ylabel:
            ax.set_ylabel(plot_style.ylabel, fontsize=plot_style.axis_label_fontsize, color='black')
        
        # Add invisible title if the main image has one (affects layout!)
        if plot_style.title:
            ax.set_title(plot_style.title, fontsize=plot_style.title_fontsize, color='black')
        
        # Spine visibility - same as main but colored black
        for spine_name, visible in plot_style.spine_visible.items():
            ax.spines[spine_name].set_visible(visible)
            ax.spines[spine_name].set_color('black')
        
        # Grid - same setting but invisible
        if plot_style.grid:
            ax.grid(True, alpha=0, linestyle=plot_style.grid_linestyle)
        
        # MUST use same tight_layout setting
        if plot_style.tight_layout:
            plt.tight_layout()
        
        # Render to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        buf = canvas.buffer_rgba()
        mask_rgba = np.asarray(buf)
        
        plt.close(fig)
        
        # Convert to grayscale
        mask_gray = cv2.cvtColor(mask_rgba, cv2.COLOR_RGBA2GRAY)
        
        # Resize to match image size if needed (should be same, but just in case)
        if mask_gray.shape[:2] != (height, width):
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Apply threshold (50% for color bleeding)
        threshold_value = int(255 * self.mask_threshold)
        _, mask_binary = cv2.threshold(mask_gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        return mask_binary
    
    def generate_sample(
        self,
        num_lines: int = None,
        x_range: Tuple[float, float] = None,
        include_markers: bool = False,
        **style_kwargs
    ) -> GeneratedSample:
        """
        Generate a single sample (image + masks).
        
        Args:
            num_lines: Number of lines (random 1-10 if None)
            x_range: X axis range (random if None)
            include_markers: Whether to include markers on lines
            **style_kwargs: Passed to style sampling
            
        Returns:
            GeneratedSample object
        """
        # Determine number of lines
        if num_lines is None:
            num_lines = random.randint(1, 10)
        
        # Sample x range
        if x_range is None:
            x_range = self._sample_x_range()
        
        # Generate x values
        x = np.linspace(x_range[0], x_range[1], 500)
        
        # Sample equations
        equations = self.equation_bank.sample_multiple_equations(
            n=num_lines,
            x_range=x_range,
            ensure_diversity=True
        )
        
        # Compute y range
        y_range = self._compute_y_range(equations, x)
        
        # Sample styles
        plot_style, line_styles = self.style_bank.sample_complete_style(
            num_lines=num_lines,
            include_markers=include_markers,
            **style_kwargs
        )
        
        # Render main image
        image = self._render_plot_image(x, equations, line_styles, plot_style, y_range)
        image_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Render individual masks
        masks = []
        for eq, ls in zip(equations, line_styles):
            mask = self._render_line_mask(x, eq, plot_style, ls, y_range, image_size)
            masks.append(mask)
        
        return GeneratedSample(
            image_path="",  # Will be set when saving
            masks=masks,
            equations=equations,
            line_styles=line_styles,
            plot_style=plot_style,
            x_range=x_range,
            y_range=y_range,
            image_size=image_size
        ), image
    
    def generate_dataset(
        self,
        output_dir: str,
        num_samples: int,
        split: str = "train",
        save_debug_masks: bool = False,
        save_metadata: bool = True,
        use_rle: bool = False
    ) -> str:
        """
        Generate a complete dataset.
        
        Args:
            output_dir: Output directory
            num_samples: Number of samples to generate
            split: Split name (train, val, test)
            save_debug_masks: Whether to save individual mask images
            save_metadata: Whether to save generation metadata
            use_rle: Use RLE encoding for annotations
            
        Returns:
            Path to annotations file
        """
        output_dir = Path(output_dir)
        
        # Create directories
        images_dir = output_dir / split / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        if save_debug_masks:
            debug_masks_dir = output_dir / "debug_masks" / split
            debug_masks_dir.mkdir(parents=True, exist_ok=True)
        
        if save_metadata:
            metadata_dir = output_dir / "metadata" / split
            metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO builder
        coco_builder = COCOAnnotationBuilder(
            description=f"Synthetic Line Plot Dataset - {split}"
        )
        
        # Statistics
        stats = {
            "total_samples": num_samples,
            "total_annotations": 0,
            "lines_per_image": [],
            "equation_types": {},
        }
        
        print(f"\nGenerating {num_samples} samples for {split} split...")
        
        for i in tqdm(range(num_samples), desc=f"Generating {split}"):
            # Set seed for this sample (reproducible)
            if self.seed is not None:
                sample_seed = self.seed + i
                random.seed(sample_seed)
                np.random.seed(sample_seed)
            
            # Generate sample
            sample, image = self.generate_sample()
            
            # Save image
            image_filename = f"image_{i:06d}.png"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            sample.image_path = image_filename
            
            # Add to COCO
            image_id = coco_builder.add_image(
                file_name=image_filename,
                width=sample.image_size[0],
                height=sample.image_size[1]
            )
            
            # Add annotations
            valid_masks = 0
            for j, mask in enumerate(sample.masks):
                if compute_area(mask) > 0:
                    coco_builder.add_annotation(
                        image_id=image_id,
                        mask=mask,
                        use_rle=use_rle
                    )
                    valid_masks += 1
                    
                    # Save debug mask
                    if save_debug_masks:
                        mask_filename = f"image_{i:06d}_mask_{j}.png"
                        cv2.imwrite(str(debug_masks_dir / mask_filename), mask)
            
            # Update statistics
            stats["total_annotations"] += valid_masks
            stats["lines_per_image"].append(valid_masks)
            
            for eq in sample.equations:
                cat = eq.category
                stats["equation_types"][cat] = stats["equation_types"].get(cat, 0) + 1
            
            # Save metadata
            if save_metadata:
                metadata = {
                    "image_filename": image_filename,
                    "num_lines": len(sample.equations),
                    "x_range": sample.x_range,
                    "y_range": sample.y_range,
                    "image_size": sample.image_size,
                    "equations": [
                        {
                            "name": eq.name,
                            "category": eq.category,
                            "latex": eq.latex,
                            "params": eq.params
                        }
                        for eq in sample.equations
                    ],
                    "line_styles": [
                        {
                            "color": ls.color,
                            "linestyle": ls.linestyle,
                            "linewidth": ls.linewidth,
                            "label": ls.label
                        }
                        for ls in sample.line_styles
                    ]
                }
                
                metadata_path = metadata_dir / f"image_{i:06d}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Save annotations
        annotations_path = output_dir / split / "annotations.json"
        coco_builder.save(str(annotations_path))
        
        # Save statistics
        stats["avg_lines_per_image"] = np.mean(stats["lines_per_image"])
        stats_path = output_dir / split / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nGeneration complete for {split}:")
        print(f"  Images: {num_samples}")
        print(f"  Annotations: {stats['total_annotations']}")
        print(f"  Avg lines/image: {stats['avg_lines_per_image']:.2f}")
        
        return str(annotations_path)


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Line Plot Dataset")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="Output directory")
    parser.add_argument("--num-train", type=int, default=1000,
                       help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=200,
                       help="Number of validation samples")
    parser.add_argument("--num-test", type=int, default=200,
                       help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--mask-thickness", type=int, default=3,
                       help="Line thickness in masks (pixels)")
    parser.add_argument("--save-debug-masks", action="store_true",
                       help="Save individual mask images for debugging")
    parser.add_argument("--save-metadata", action="store_true",
                       help="Save generation metadata for each image")
    parser.add_argument("--use-rle", action="store_true",
                       help="Use RLE encoding for annotations")
    
    args = parser.parse_args()
    
    # Create generator
    generator = LinePlotGenerator(
        seed=args.seed,
        mask_line_thickness=args.mask_thickness,
        mask_threshold=0.5  # 50% as specified
    )
    
    print("=" * 60)
    print("Synthetic Line Plot Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Mask thickness: {args.mask_thickness}px")
    print(f"Mask threshold: 50%")
    
    # Generate each split
    splits = [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test),
    ]
    
    for split_name, num_samples in splits:
        if num_samples > 0:
            generator.generate_dataset(
                output_dir=args.output_dir,
                num_samples=num_samples,
                split=split_name,
                save_debug_masks=args.save_debug_masks,
                save_metadata=args.save_metadata,
                use_rle=args.use_rle
            )
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    
    # Print directory structure
    print(f"\nDirectory structure:")
    print(f"  {args.output_dir}/")
    for split_name, num_samples in splits:
        if num_samples > 0:
            print(f"    {split_name}/")
            print(f"      images/")
            print(f"      annotations.json")
            print(f"      stats.json")
    
    if args.save_debug_masks:
        print(f"    debug_masks/")
    if args.save_metadata:
        print(f"    metadata/")


if __name__ == "__main__":
    main()

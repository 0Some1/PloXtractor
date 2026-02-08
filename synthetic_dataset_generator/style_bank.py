"""
Style Bank for Synthetic Line Plot Dataset
Contains visual styling options for lines and plots
"""

import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LineStyle:
    """Visual style for a single line"""
    color: str
    linestyle: str
    linewidth: float
    marker: Optional[str] = None
    markersize: float = 6.0
    alpha: float = 1.0
    label: Optional[str] = None


@dataclass 
class PlotStyle:
    """Visual style for the entire plot"""
    figsize: Tuple[float, float]  # inches
    dpi: int
    background_color: str
    grid: bool
    grid_alpha: float
    grid_linestyle: str
    title: Optional[str]
    title_fontsize: int
    xlabel: Optional[str]
    ylabel: Optional[str]
    axis_label_fontsize: int
    tick_label_fontsize: int
    show_legend: bool
    legend_loc: str
    spine_visible: Dict[str, bool]
    tight_layout: bool


class LegendBank:
    """Bank of legend/label text options"""
    
    SCIENTIFIC_LABELS = [
        "Temperature (°C)", "Temperature (K)", "Pressure (Pa)", "Pressure (atm)",
        "Velocity (m/s)", "Acceleration (m/s²)", "Force (N)", "Energy (J)",
        "Power (W)", "Voltage (V)", "Current (A)", "Resistance (Ω)",
        "Concentration (mol/L)", "Time (s)", "Distance (m)", "Mass (kg)",
        "Density (kg/m³)", "Frequency (Hz)", "Wavelength (nm)", "Intensity",
        "Absorption", "Transmittance", "Reflectance", "pH", "Conductivity",
    ]
    
    BUSINESS_LABELS = [
        "Revenue ($)", "Profit ($)", "Cost ($)", "Sales", "Growth Rate (%)",
        "Market Share (%)", "ROI (%)", "Q1 Results", "Q2 Results", "Q3 Results", "Q4 Results",
        "YoY Change", "MoM Change", "Customer Count", "Conversion Rate",
        "Churn Rate", "LTV", "CAC", "ARPU", "DAU", "MAU", "Retention",
        "Engagement", "NPS Score", "Satisfaction", "Performance Index",
    ]
    
    GENERIC_LABELS = [
        "Series A", "Series B", "Series C", "Series D", "Series E",
        "Data 1", "Data 2", "Data 3", "Data 4", "Data 5",
        "Line 1", "Line 2", "Line 3", "Line 4", "Line 5",
        "Group A", "Group B", "Group C", "Control", "Treatment",
        "Baseline", "Variant 1", "Variant 2", "Model A", "Model B",
        "Predicted", "Actual", "Expected", "Observed", "Theoretical",
    ]
    
    MATH_LABELS = [
        "f(x)", "g(x)", "h(x)", "y₁", "y₂", "y₃",
        "sin(x)", "cos(x)", "tan(x)", "eˣ", "ln(x)",
        "P(x)", "Q(x)", "R(x)", "φ(x)", "ψ(x)",
    ]
    
    AXIS_X_LABELS = [
        "Time", "Time (s)", "Time (min)", "Time (h)", "Date",
        "x", "Position", "Distance", "Index", "Sample",
        "Iteration", "Epoch", "Step", "Frequency", "Wavelength",
    ]
    
    AXIS_Y_LABELS = [
        "Value", "y", "Amplitude", "Magnitude", "Count",
        "Probability", "Density", "Score", "Rate", "Ratio",
        "Level", "Output", "Response", "Measurement", "Result",
    ]
    
    TITLE_TEMPLATES = [
        "{} vs {}",
        "{} over {}",
        "{} as a function of {}",
        "Comparison of {}",
        "Analysis: {}",
        "{} Trends",
        "{} Distribution",
        "Effect of {} on {}",
        "{} Performance",
        "{} Results",
    ]
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
    
    def sample_line_label(self, category: str = None) -> str:
        """Sample a label for a line"""
        if category is None:
            category = random.choice(['scientific', 'business', 'generic', 'math'])
        
        pools = {
            'scientific': self.SCIENTIFIC_LABELS,
            'business': self.BUSINESS_LABELS,
            'generic': self.GENERIC_LABELS,
            'math': self.MATH_LABELS,
        }
        
        return random.choice(pools.get(category, self.GENERIC_LABELS))
    
    def sample_axis_labels(self) -> Tuple[str, str]:
        """Sample x and y axis labels"""
        return random.choice(self.AXIS_X_LABELS), random.choice(self.AXIS_Y_LABELS)
    
    def sample_title(self, y_label: str = None, x_label: str = None) -> str:
        """Sample a plot title"""
        template = random.choice(self.TITLE_TEMPLATES)
        
        if "{}" in template:
            count = template.count("{}")
            if count == 1:
                fill = y_label or random.choice(self.GENERIC_LABELS)
                return template.format(fill)
            elif count == 2:
                fill1 = y_label or random.choice(self.AXIS_Y_LABELS)
                fill2 = x_label or random.choice(self.AXIS_X_LABELS)
                return template.format(fill1, fill2)
        
        return template
    
    def sample_multiple_labels(self, n: int, ensure_unique: bool = True) -> List[str]:
        """Sample n labels, optionally ensuring uniqueness"""
        all_labels = (
            self.SCIENTIFIC_LABELS + 
            self.BUSINESS_LABELS + 
            self.GENERIC_LABELS + 
            self.MATH_LABELS
        )
        
        if ensure_unique and n <= len(all_labels):
            return random.sample(all_labels, n)
        else:
            return [random.choice(all_labels) for _ in range(n)]


class AnnotationBank:
    """Bank of text annotations, callouts, and noise elements for plots"""

    CALLOUT_TEXTS = [
        "max", "min", "peak", "trough", "inflection point",
        "critical point", "optimum", "saturation", "equilibrium",
        "onset", "offset", "threshold", "breakpoint", "anomaly",
        "x = {:.2f}", "y = {:.2f}", "({:.1f}, {:.1f})",
    ]

    REFERENCE_LINE_LABELS = [
        "mean", "median", "threshold", "baseline", "target",
        "upper limit", "lower limit", "avg", "cutoff", "zero",
        "μ", "μ + σ", "μ - σ", "μ + 2σ", "μ - 2σ",
        "95% CI", "99% CI", "reference", "control",
    ]

    TEXTBOX_TEXTS = [
        "R² = {:.3f}", "p < {:.3f}", "n = {}", "σ = {:.2f}",
        "RMSE = {:.3f}", "MAE = {:.3f}", "r = {:.3f}",
        "slope = {:.3f}", "intercept = {:.3f}",
        "AUC = {:.3f}", "F1 = {:.3f}",
        "Model A", "Model B", "Fitted", "Raw data",
        "α = {:.2f}", "β = {:.2f}", "λ = {:.3f}",
    ]

    WATERMARK_TEXTS = [
        "DRAFT", "SAMPLE", "PRELIMINARY", "CONFIDENTIAL",
        "DO NOT DISTRIBUTE", "PREPRINT", "REVIEW COPY",
        "Figure {}", "Fig. {}", "Chart {}",
        "Source: internal", "Source: simulated",
        "Generated by PlotLib", "matplotlib v3.x",
        "© 2024", "© 2025",
    ]

    SHADED_REGION_LABELS = [
        "confidence interval", "std dev", "error band",
        "±1σ", "±2σ", "95% CI", "IQR",
        "region of interest", "operating range",
    ]

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

    def sample_callout(self, x: float = None, y: float = None) -> str:
        text = random.choice(self.CALLOUT_TEXTS)
        if '{}' in text or '{:' in text:
            if x is not None and y is not None:
                count = text.count('{')
                if count == 2:
                    text = text.format(x, y)
                elif count == 1:
                    text = text.format(random.choice([x, y]))
            else:
                text = text.format(random.uniform(-5, 5))
        return text

    def sample_reference_label(self) -> str:
        return random.choice(self.REFERENCE_LINE_LABELS)

    def sample_textbox(self) -> str:
        text = random.choice(self.TEXTBOX_TEXTS)
        if '{}' in text or '{:' in text:
            count = text.count('{')
            vals = [random.uniform(0, 1) if ':' in text else random.randint(10, 500)]
            text = text.format(*[random.uniform(0, 1) if '{:' in text else random.randint(10, 500) for _ in range(count)])
        return text

    def sample_watermark(self) -> str:
        text = random.choice(self.WATERMARK_TEXTS)
        if '{}' in text:
            text = text.format(random.randint(1, 20))
        return text

    def sample_shaded_label(self) -> str:
        return random.choice(self.SHADED_REGION_LABELS)


class StyleBank:
    """
    Bank of visual styles for generating diverse line plots.
    """
    
    # Color palettes
    COLORS_STANDARD = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    COLORS_VIBRANT = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5'
    ]
    
    COLORS_PASTEL = [
        '#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9',
        '#fff2ae', '#f1e2cc', '#cccccc', '#fb9a99', '#fdbf6f'
    ]
    
    COLORS_DARK = [
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
        '#e6ab02', '#a6761d', '#666666', '#a6cee3', '#b2df8a'
    ]
    
    LINE_STYLES = ['-', '--', '-.', ':']
    
    # Markers that look good on lines (not used for masks, but for visual variety)
    MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x']
    
    LEGEND_LOCATIONS = [
        'upper right', 'upper left', 'lower right', 'lower left',
        'right', 'center left', 'center right', 'lower center', 'upper center',
        'best'
    ]
    
    GRID_STYLES = ['-', '--', ':', '-.']
    
    BACKGROUND_COLORS = ['white', '#f8f8f8', '#f0f0f0', '#fafafa', 'whitesmoke']
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        self.legend_bank = LegendBank(seed=seed)
        self.annotation_bank = AnnotationBank(seed=seed)
        self._used_colors = []
        self._used_styles = []
    
    def reset_used(self):
        """Reset tracking of used colors/styles for a new plot"""
        self._used_colors = []
        self._used_styles = []
    
    def sample_color_palette(self) -> List[str]:
        """Sample a color palette to use for a plot"""
        palettes = [
            self.COLORS_STANDARD,
            self.COLORS_VIBRANT,
            self.COLORS_PASTEL,
            self.COLORS_DARK,
        ]
        return random.choice(palettes)
    
    def sample_line_style(
        self, 
        color_palette: List[str] = None,
        ensure_distinguishable: bool = True,
        include_markers: bool = False,
        label: str = None
    ) -> LineStyle:
        """
        Sample a visual style for a line.
        
        Args:
            color_palette: Colors to choose from
            ensure_distinguishable: Try to use different colors/styles
            include_markers: Whether to potentially add markers
            label: Label for the line
        """
        if color_palette is None:
            color_palette = self.COLORS_STANDARD
        
        # Select color
        if ensure_distinguishable and self._used_colors:
            available_colors = [c for c in color_palette if c not in self._used_colors]
            if not available_colors:
                available_colors = color_palette
            color = random.choice(available_colors)
        else:
            color = random.choice(color_palette)
        
        self._used_colors.append(color)
        
        # Select line style
        if ensure_distinguishable and len(self._used_colors) > len(color_palette) // 2:
            # Use different line styles when running low on colors
            available_styles = [s for s in self.LINE_STYLES if s not in self._used_styles]
            if not available_styles:
                available_styles = self.LINE_STYLES
            linestyle = random.choice(available_styles)
        else:
            # Mostly use solid lines
            linestyle = random.choices(
                self.LINE_STYLES,
                weights=[0.7, 0.15, 0.1, 0.05],
                k=1
            )[0]
        
        self._used_styles.append(linestyle)
        
        # Line width
        linewidth = random.uniform(1.5, 3.0)
        
        # Markers (optional, for visual diversity only - won't affect masks)
        marker = None
        markersize = 6.0
        if include_markers and random.random() < 0.3:
            marker = random.choice(self.MARKERS)
            markersize = random.uniform(4, 8)
        
        # Alpha (slight variation)
        alpha = random.uniform(0.85, 1.0)
        
        return LineStyle(
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            alpha=alpha,
            label=label
        )
    
    def sample_plot_style(
        self,
        num_lines: int = 1,
        include_title: bool = None,
        include_legend: bool = None,
        include_grid: bool = None,
        include_axis_labels: bool = None,
    ) -> PlotStyle:
        """
        Sample a visual style for the entire plot.
        
        Args:
            num_lines: Number of lines (affects legend decision)
            include_*: Override random decisions for these elements
        """
        # Figure size (in inches, will be converted to pixels via DPI)
        width = random.uniform(6, 10)
        height = random.uniform(4, 7)
        dpi = random.choice([80, 100, 120])
        
        # Background
        background_color = random.choice(self.BACKGROUND_COLORS)
        
        # Grid
        if include_grid is None:
            include_grid = random.random() < 0.5
        grid_alpha = random.uniform(0.3, 0.7) if include_grid else 0
        grid_linestyle = random.choice(self.GRID_STYLES)
        
        # Title
        if include_title is None:
            include_title = random.random() < 0.7
        
        x_label_text, y_label_text = self.legend_bank.sample_axis_labels()
        title = self.legend_bank.sample_title(y_label_text, x_label_text) if include_title else None
        title_fontsize = random.randint(12, 18)
        
        # Axis labels
        if include_axis_labels is None:
            include_axis_labels = random.random() < 0.8
        
        xlabel = x_label_text if include_axis_labels else None
        ylabel = y_label_text if include_axis_labels else None
        axis_label_fontsize = random.randint(10, 14)
        tick_label_fontsize = random.randint(8, 12)
        
        # Legend
        if include_legend is None:
            include_legend = num_lines > 1 and random.random() < 0.8
        
        legend_loc = random.choice(self.LEGEND_LOCATIONS)
        
        # Spines (axis borders)
        spine_style = random.choice(['all', 'bottom_left', 'none'])
        if spine_style == 'all':
            spine_visible = {'top': True, 'right': True, 'bottom': True, 'left': True}
        elif spine_style == 'bottom_left':
            spine_visible = {'top': False, 'right': False, 'bottom': True, 'left': True}
        else:
            spine_visible = {'top': False, 'right': False, 'bottom': True, 'left': True}
        
        return PlotStyle(
            figsize=(width, height),
            dpi=dpi,
            background_color=background_color,
            grid=include_grid,
            grid_alpha=grid_alpha,
            grid_linestyle=grid_linestyle,
            title=title,
            title_fontsize=title_fontsize,
            xlabel=xlabel,
            ylabel=ylabel,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            show_legend=include_legend,
            legend_loc=legend_loc,
            spine_visible=spine_visible,
            tight_layout=True
        )
    
    def sample_complete_style(
        self,
        num_lines: int,
        include_markers: bool = False,
        **plot_kwargs
    ) -> Tuple[PlotStyle, List[LineStyle]]:
        """
        Sample complete styling for a plot with multiple lines.
        
        Returns:
            Tuple of (PlotStyle, list of LineStyle for each line)
        """
        self.reset_used()
        
        # Get plot style
        plot_style = self.sample_plot_style(num_lines=num_lines, **plot_kwargs)
        
        # Get color palette
        color_palette = self.sample_color_palette()
        
        # Get line styles with labels
        line_labels = self.legend_bank.sample_multiple_labels(num_lines, ensure_unique=True)
        
        line_styles = []
        for i in range(num_lines):
            label = line_labels[i] if plot_style.show_legend else None
            style = self.sample_line_style(
                color_palette=color_palette,
                ensure_distinguishable=True,
                include_markers=include_markers,
                label=label
            )
            line_styles.append(style)
        
        return plot_style, line_styles


# Test the style bank
if __name__ == "__main__":
    bank = StyleBank(seed=42)
    
    print("Testing Style Bank")
    print("=" * 50)
    
    # Sample complete style for 5 lines
    plot_style, line_styles = bank.sample_complete_style(num_lines=5)
    
    print("\nPlot Style:")
    print(f"  Figure size: {plot_style.figsize}")
    print(f"  DPI: {plot_style.dpi}")
    print(f"  Title: {plot_style.title}")
    print(f"  Grid: {plot_style.grid}")
    print(f"  Legend: {plot_style.show_legend} at {plot_style.legend_loc}")
    
    print("\nLine Styles:")
    for i, ls in enumerate(line_styles):
        print(f"  Line {i+1}: color={ls.color}, style={ls.linestyle}, "
              f"width={ls.linewidth:.1f}, label={ls.label}")
    
    print("\n" + "=" * 50)
    print("Legend Bank Samples:")
    legend_bank = LegendBank(seed=42)
    print(f"  Scientific: {legend_bank.sample_line_label('scientific')}")
    print(f"  Business: {legend_bank.sample_line_label('business')}")
    print(f"  Math: {legend_bank.sample_line_label('math')}")
    print(f"  Title: {legend_bank.sample_title()}")

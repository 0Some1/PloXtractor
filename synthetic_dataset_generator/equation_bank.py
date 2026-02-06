"""
Equation Bank for Synthetic Line Plot Dataset
Contains various mathematical functions with randomizable parameters
"""

import numpy as np
from typing import Callable, Dict, Any, Tuple
from dataclasses import dataclass
import random


@dataclass
class Equation:
    """Represents a mathematical equation with its properties"""
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    latex: str  # For display/legend
    category: str
    params: Dict[str, Any]


class EquationBank:
    """
    Bank of mathematical equations for generating diverse line plots.
    Each equation type can be sampled with random parameters.
    """
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define equation categories with their sampling weights
        self.categories = {
            'linear': 1.0,
            'polynomial': 1.5,
            'trigonometric': 1.2,
            'exponential': 0.8,
            'logarithmic': 0.6,
            'power': 0.7,
            'sigmoid': 0.5,
            'composite': 0.8,
            'rational': 0.4,
        }
    
    def sample_equation(self, x_range: Tuple[float, float], category: str = None) -> Equation:
        """
        Sample a random equation from the bank.
        
        Args:
            x_range: (x_min, x_max) for the plot domain
            category: Optional specific category to sample from
            
        Returns:
            Equation object with function and metadata
        """
        if category is None:
            # Weighted random selection
            categories = list(self.categories.keys())
            weights = list(self.categories.values())
            category = random.choices(categories, weights=weights, k=1)[0]
        
        # Dispatch to appropriate generator
        generators = {
            'linear': self._generate_linear,
            'polynomial': self._generate_polynomial,
            'trigonometric': self._generate_trigonometric,
            'exponential': self._generate_exponential,
            'logarithmic': self._generate_logarithmic,
            'power': self._generate_power,
            'sigmoid': self._generate_sigmoid,
            'composite': self._generate_composite,
            'rational': self._generate_rational,
        }
        
        return generators[category](x_range)
    
    def _generate_linear(self, x_range: Tuple[float, float]) -> Equation:
        """Generate y = mx + b"""
        m = random.uniform(-3, 3)
        b = random.uniform(-5, 5)
        
        def func(x):
            return m * x + b
        
        # Format nicely
        if abs(b) < 0.01:
            latex = f"y = {m:.2f}x"
        elif b >= 0:
            latex = f"y = {m:.2f}x + {b:.2f}"
        else:
            latex = f"y = {m:.2f}x - {abs(b):.2f}"
        
        return Equation(
            name='linear',
            func=func,
            latex=latex,
            category='linear',
            params={'m': m, 'b': b}
        )
    
    def _generate_polynomial(self, x_range: Tuple[float, float]) -> Equation:
        """Generate polynomial of degree 2-4"""
        degree = random.randint(2, 4)
        
        # Scale coefficients based on degree to avoid extreme values
        scale = 1.0 / (2 ** (degree - 1))
        coeffs = [random.uniform(-2, 2) * scale for _ in range(degree + 1)]
        
        # Ensure leading coefficient is not too small
        if abs(coeffs[0]) < 0.1:
            coeffs[0] = random.choice([-1, 1]) * random.uniform(0.2, 0.5) * scale
        
        def func(x):
            result = np.zeros_like(x, dtype=float)
            for i, c in enumerate(coeffs):
                result += c * (x ** (degree - i))
            return result
        
        # Build latex string
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if abs(c) < 0.01:
                continue
            if power == 0:
                terms.append(f"{c:.2f}")
            elif power == 1:
                terms.append(f"{c:.2f}x")
            else:
                terms.append(f"{c:.2f}x^{power}")
        
        latex = "y = " + " + ".join(terms) if terms else "y = 0"
        latex = latex.replace("+ -", "- ")
        
        return Equation(
            name=f'polynomial_deg{degree}',
            func=func,
            latex=latex,
            category='polynomial',
            params={'degree': degree, 'coefficients': coeffs}
        )
    
    def _generate_trigonometric(self, x_range: Tuple[float, float]) -> Equation:
        """Generate trigonometric functions: sin, cos, or combinations"""
        trig_type = random.choice(['sin', 'cos', 'sin_cos', 'tan_limited'])
        
        amplitude = random.uniform(0.5, 3.0)
        frequency = random.uniform(0.5, 3.0)
        phase = random.uniform(0, 2 * np.pi)
        vertical_shift = random.uniform(-2, 2)
        
        if trig_type == 'sin':
            def func(x):
                return amplitude * np.sin(frequency * x + phase) + vertical_shift
            latex = f"y = {amplitude:.2f}sin({frequency:.2f}x)"
            
        elif trig_type == 'cos':
            def func(x):
                return amplitude * np.cos(frequency * x + phase) + vertical_shift
            latex = f"y = {amplitude:.2f}cos({frequency:.2f}x)"
            
        elif trig_type == 'sin_cos':
            amp2 = random.uniform(0.3, 1.5)
            freq2 = random.uniform(1.0, 4.0)
            def func(x):
                return amplitude * np.sin(frequency * x) + amp2 * np.cos(freq2 * x) + vertical_shift
            latex = f"y = {amplitude:.2f}sin({frequency:.2f}x) + {amp2:.2f}cos({freq2:.2f}x)"
            
        else:  # tan_limited - use tanh to avoid infinities
            def func(x):
                return amplitude * np.tanh(frequency * x + phase) + vertical_shift
            latex = f"y = {amplitude:.2f}tanh({frequency:.2f}x)"
        
        return Equation(
            name=trig_type,
            func=func,
            latex=latex,
            category='trigonometric',
            params={'amplitude': amplitude, 'frequency': frequency, 'phase': phase}
        )
    
    def _generate_exponential(self, x_range: Tuple[float, float]) -> Equation:
        """Generate exponential functions"""
        exp_type = random.choice(['growth', 'decay', 'shifted'])
        
        # Keep parameters modest to avoid overflow
        a = random.uniform(0.5, 2.0)
        b = random.uniform(0.1, 0.5) * random.choice([-1, 1])
        c = random.uniform(-2, 2)
        
        x_min, x_max = x_range
        
        if exp_type == 'growth':
            b = abs(b) * 0.3  # Slower growth
            def func(x):
                return a * np.exp(b * x) + c
            latex = f"y = {a:.2f}e^{{{b:.2f}x}}"
            
        elif exp_type == 'decay':
            b = -abs(b) * 0.3
            def func(x):
                return a * np.exp(b * x) + c
            latex = f"y = {a:.2f}e^{{{b:.2f}x}}"
            
        else:  # shifted
            h = random.uniform(x_min, x_max)
            def func(x):
                return a * np.exp(b * (x - h)) + c
            latex = f"y = {a:.2f}e^{{{b:.2f}(x-{h:.1f})}}"
        
        return Equation(
            name=exp_type + '_exp',
            func=func,
            latex=latex,
            category='exponential',
            params={'a': a, 'b': b, 'c': c}
        )
    
    def _generate_logarithmic(self, x_range: Tuple[float, float]) -> Equation:
        """Generate logarithmic functions"""
        a = random.uniform(0.5, 3.0) * random.choice([-1, 1])
        b = random.uniform(0.5, 2.0)
        c = random.uniform(-2, 2)
        
        # Shift to ensure positive argument
        x_min, x_max = x_range
        shift = max(0.1, -x_min + 0.5) if x_min <= 0 else 0.1
        
        def func(x):
            arg = b * x + shift
            # Clip to avoid log of negative/zero
            arg = np.maximum(arg, 1e-10)
            return a * np.log(arg) + c
        
        latex = f"y = {a:.2f}ln({b:.2f}x + {shift:.2f})"
        
        return Equation(
            name='logarithmic',
            func=func,
            latex=latex,
            category='logarithmic',
            params={'a': a, 'b': b, 'shift': shift, 'c': c}
        )
    
    def _generate_power(self, x_range: Tuple[float, float]) -> Equation:
        """Generate power functions y = ax^n"""
        a = random.uniform(0.3, 2.0) * random.choice([-1, 1])
        # Use fractional or integer powers
        n = random.choice([0.5, 1.5, 2, 3, -0.5, -1])
        c = random.uniform(-2, 2)
        
        x_min, x_max = x_range
        
        def func(x):
            # Handle negative x for fractional powers
            if n != int(n):
                x_safe = np.abs(x) + 0.01
                return a * np.power(x_safe, n) + c
            else:
                return a * np.power(x, n) + c
        
        if n == int(n):
            latex = f"y = {a:.2f}x^{{{int(n)}}}"
        else:
            latex = f"y = {a:.2f}x^{{{n:.1f}}}"
        
        return Equation(
            name='power',
            func=func,
            latex=latex,
            category='power',
            params={'a': a, 'n': n, 'c': c}
        )
    
    def _generate_sigmoid(self, x_range: Tuple[float, float]) -> Equation:
        """Generate sigmoid/logistic functions"""
        L = random.uniform(1, 5)  # Maximum value
        k = random.uniform(0.5, 3)  # Steepness
        x0 = random.uniform(*x_range)  # Midpoint
        b = random.uniform(-1, 1)  # Vertical shift
        
        def func(x):
            return L / (1 + np.exp(-k * (x - x0))) + b
        
        latex = f"y = {L:.2f}/(1 + e^{{-{k:.2f}(x-{x0:.1f})}})"
        
        return Equation(
            name='sigmoid',
            func=func,
            latex=latex,
            category='sigmoid',
            params={'L': L, 'k': k, 'x0': x0, 'b': b}
        )
    
    def _generate_composite(self, x_range: Tuple[float, float]) -> Equation:
        """Generate composite functions (combinations)"""
        composite_type = random.choice([
            'linear_plus_sin',
            'poly_plus_sin',
            'exp_sin',
            'damped_oscillation'
        ])
        
        if composite_type == 'linear_plus_sin':
            m = random.uniform(-1, 1)
            b = random.uniform(-2, 2)
            amp = random.uniform(0.5, 2)
            freq = random.uniform(1, 3)
            
            def func(x):
                return m * x + b + amp * np.sin(freq * x)
            latex = f"y = {m:.2f}x + {amp:.2f}sin({freq:.2f}x)"
            
        elif composite_type == 'poly_plus_sin':
            a = random.uniform(-0.2, 0.2)
            amp = random.uniform(0.5, 1.5)
            freq = random.uniform(1, 3)
            
            def func(x):
                return a * x**2 + amp * np.sin(freq * x)
            latex = f"y = {a:.2f}x² + {amp:.2f}sin({freq:.2f}x)"
            
        elif composite_type == 'exp_sin':
            a = random.uniform(0.5, 1.5)
            b = random.uniform(0.05, 0.2)
            freq = random.uniform(1, 3)
            
            def func(x):
                return a * np.exp(b * x) * np.sin(freq * x)
            latex = f"y = {a:.2f}e^{{{b:.2f}x}}sin({freq:.2f}x)"
            
        else:  # damped_oscillation
            a = random.uniform(1, 3)
            decay = random.uniform(0.1, 0.5)
            freq = random.uniform(1, 4)
            
            def func(x):
                return a * np.exp(-decay * np.abs(x)) * np.cos(freq * x)
            latex = f"y = {a:.2f}e^{{-{decay:.2f}|x|}}cos({freq:.2f}x)"
        
        return Equation(
            name=composite_type,
            func=func,
            latex=latex,
            category='composite',
            params={'type': composite_type}
        )
    
    def _generate_rational(self, x_range: Tuple[float, float]) -> Equation:
        """Generate rational functions (ratios of polynomials)"""
        rational_type = random.choice(['simple', 'quadratic_over_linear'])
        
        if rational_type == 'simple':
            a = random.uniform(0.5, 3) * random.choice([-1, 1])
            h = random.uniform(*x_range) * 0.5  # Asymptote location
            k = random.uniform(-2, 2)
            
            def func(x):
                denom = x - h
                # Avoid division by zero
                denom = np.where(np.abs(denom) < 0.1, 0.1 * np.sign(denom + 0.001), denom)
                return a / denom + k
            latex = f"y = {a:.2f}/(x - {h:.2f}) + {k:.2f}"
            
        else:  # quadratic_over_linear
            a = random.uniform(-1, 1)
            b = random.uniform(-2, 2)
            c = random.uniform(0.5, 2)
            
            def func(x):
                denom = c + x**2
                return (a * x**2 + b) / denom
            latex = f"y = ({a:.2f}x² + {b:.2f})/({c:.2f} + x²)"
        
        return Equation(
            name=rational_type + '_rational',
            func=func,
            latex=latex,
            category='rational',
            params={'type': rational_type}
        )
    
    def sample_multiple_equations(
        self, 
        n: int, 
        x_range: Tuple[float, float],
        ensure_diversity: bool = True
    ) -> list:
        """
        Sample multiple equations, optionally ensuring category diversity.
        
        Args:
            n: Number of equations to sample
            x_range: Domain for the equations
            ensure_diversity: If True, try to use different categories
            
        Returns:
            List of Equation objects
        """
        equations = []
        used_categories = set()
        
        for i in range(n):
            if ensure_diversity and len(used_categories) < len(self.categories) and i < len(self.categories):
                # Try to pick unused category
                available = [c for c in self.categories.keys() if c not in used_categories]
                if available:
                    category = random.choice(available)
                else:
                    category = None
            else:
                category = None
            
            eq = self.sample_equation(x_range, category=category)
            equations.append(eq)
            used_categories.add(eq.category)
        
        return equations


# Test the equation bank
if __name__ == "__main__":
    bank = EquationBank(seed=42)
    
    print("Testing Equation Bank")
    print("=" * 50)
    
    x = np.linspace(-5, 5, 100)
    
    for category in bank.categories.keys():
        eq = bank.sample_equation((-5, 5), category=category)
        y = eq.func(x)
        print(f"\n{category.upper()}")
        print(f"  LaTeX: {eq.latex}")
        print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    
    print("\n" + "=" * 50)
    print("Sampling 5 diverse equations:")
    equations = bank.sample_multiple_equations(5, (-5, 5), ensure_diversity=True)
    for eq in equations:
        print(f"  [{eq.category}] {eq.latex}")

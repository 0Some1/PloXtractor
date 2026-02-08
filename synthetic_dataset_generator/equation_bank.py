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
            'piecewise': 0.9,
            'gaussian': 0.8,
            'absolute_value': 0.6,
            'logistic_growth': 0.5,
            'fourier': 0.7,
            'spline_noise': 0.8,
            'sqrt_cbrt': 0.5,
            'reciprocal_trig': 0.3,
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
            'piecewise': self._generate_piecewise,
            'gaussian': self._generate_gaussian,
            'absolute_value': self._generate_absolute_value,
            'logistic_growth': self._generate_logistic_growth,
            'fourier': self._generate_fourier,
            'spline_noise': self._generate_spline_noise,
            'sqrt_cbrt': self._generate_sqrt_cbrt,
            'reciprocal_trig': self._generate_reciprocal_trig,
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
    
    def _generate_piecewise(self, x_range: Tuple[float, float]) -> Equation:
        """Generate piecewise functions: step, sawtooth, square wave, triangle wave"""
        pw_type = random.choice(['step', 'sawtooth', 'square', 'triangle'])

        amplitude = random.uniform(0.5, 3.0)
        period = random.uniform(1.0, 4.0)
        vertical_shift = random.uniform(-2, 2)

        if pw_type == 'step':
            num_steps = random.randint(2, 6)
            x_min, x_max = x_range
            step_width = (x_max - x_min) / num_steps
            levels = [random.uniform(-3, 3) for _ in range(num_steps)]

            def func(x):
                result = np.full_like(x, levels[0], dtype=float)
                for s in range(1, num_steps):
                    result = np.where(x >= x_min + s * step_width, levels[s], result)
                return result

            latex = f"step function ({num_steps} levels)"

        elif pw_type == 'sawtooth':
            def func(x):
                phase = x / period
                return amplitude * 2 * (phase - np.floor(phase + 0.5)) + vertical_shift
            latex = f"y = {amplitude:.2f} sawtooth(x/{period:.2f})"

        elif pw_type == 'square':
            def func(x):
                return amplitude * np.sign(np.sin(2 * np.pi * x / period)) + vertical_shift
            latex = f"y = {amplitude:.2f} sgn(sin(2πx/{period:.2f}))"

        else:  # triangle
            def func(x):
                phase = x / period
                return amplitude * 2 * np.abs(2 * (phase - np.floor(phase + 0.5))) - amplitude + vertical_shift
            latex = f"y = {amplitude:.2f} triangle(x/{period:.2f})"

        return Equation(
            name=pw_type,
            func=func,
            latex=latex,
            category='piecewise',
            params={'type': pw_type, 'amplitude': amplitude, 'period': period}
        )

    def _generate_gaussian(self, x_range: Tuple[float, float]) -> Equation:
        """Generate Gaussian / bell curve functions"""
        gauss_type = random.choice(['single', 'bimodal', 'skewed'])

        x_min, x_max = x_range
        x_mid = (x_min + x_max) / 2
        x_span = x_max - x_min

        if gauss_type == 'single':
            amplitude = random.uniform(1, 5)
            mu = random.uniform(x_mid - x_span * 0.3, x_mid + x_span * 0.3)
            sigma = random.uniform(x_span * 0.05, x_span * 0.3)

            def func(x):
                return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            latex = f"y = {amplitude:.2f} exp(-((x-{mu:.1f})/{sigma:.1f})²/2)"

        elif gauss_type == 'bimodal':
            a1 = random.uniform(1, 4)
            a2 = random.uniform(1, 4)
            mu1 = x_mid - random.uniform(x_span * 0.1, x_span * 0.3)
            mu2 = x_mid + random.uniform(x_span * 0.1, x_span * 0.3)
            s1 = random.uniform(x_span * 0.05, x_span * 0.15)
            s2 = random.uniform(x_span * 0.05, x_span * 0.15)

            def func(x):
                return a1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2) + a2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
            latex = f"y = {a1:.1f}N({mu1:.1f},{s1:.1f}) + {a2:.1f}N({mu2:.1f},{s2:.1f})"

        else:  # skewed
            amplitude = random.uniform(1, 5)
            mu = random.uniform(x_mid - x_span * 0.2, x_mid + x_span * 0.2)
            sigma = random.uniform(x_span * 0.05, x_span * 0.25)
            alpha = random.uniform(-5, 5)  # skewness

            def func(x):
                t = (x - mu) / sigma
                gauss = amplitude * np.exp(-0.5 * t ** 2)
                skew = 1 + np.vectorize(lambda v: float(np.math.erf(alpha * v / np.sqrt(2))))(t)
                return gauss * skew
            latex = f"y = {amplitude:.2f} skew-normal(α={alpha:.1f})"

        return Equation(
            name=gauss_type + '_gaussian',
            func=func,
            latex=latex,
            category='gaussian',
            params={'type': gauss_type}
        )

    def _generate_absolute_value(self, x_range: Tuple[float, float]) -> Equation:
        """Generate absolute value functions"""
        av_type = random.choice(['v_shape', 'ramp', 'piecewise_linear'])

        x_min, x_max = x_range

        if av_type == 'v_shape':
            a = random.uniform(0.5, 3) * random.choice([-1, 1])
            h = random.uniform(*x_range) * 0.5
            k = random.uniform(-3, 3)

            def func(x):
                return a * np.abs(x - h) + k
            latex = f"y = {a:.2f}|x - {h:.2f}| + {k:.2f}"

        elif av_type == 'ramp':
            a = random.uniform(0.5, 2)
            threshold = random.uniform(*x_range) * 0.5

            def func(x):
                return a * np.maximum(x - threshold, 0)
            latex = f"y = {a:.2f} max(x - {threshold:.2f}, 0)"

        else:  # piecewise_linear - multiple V-shapes
            num_vs = random.randint(2, 4)
            points_x = sorted(np.random.uniform(x_min, x_max, num_vs))
            points_y = np.random.uniform(-3, 3, num_vs)

            def func(x):
                result = np.interp(x, points_x, points_y)
                return result
            latex = f"piecewise-linear ({num_vs} segments)"

        return Equation(
            name=av_type,
            func=func,
            latex=latex,
            category='absolute_value',
            params={'type': av_type}
        )

    def _generate_logistic_growth(self, x_range: Tuple[float, float]) -> Equation:
        """Generate logistic growth and Gompertz curves"""
        lg_type = random.choice(['logistic', 'gompertz'])

        if lg_type == 'logistic':
            L = random.uniform(2, 10)  # carrying capacity
            k = random.uniform(0.3, 2)  # growth rate
            x0 = random.uniform(*x_range) * 0.5  # midpoint
            C = random.uniform(1, 20)  # initial condition factor

            def func(x):
                return L / (1 + C * np.exp(-k * (x - x0)))
            latex = f"y = {L:.1f}/(1 + {C:.1f}e^{{-{k:.2f}(x-{x0:.1f})}})"

        else:  # gompertz
            a = random.uniform(2, 10)
            b = random.uniform(1, 5)
            c = random.uniform(0.2, 1.5)

            def func(x):
                return a * np.exp(-b * np.exp(-c * x))
            latex = f"y = {a:.1f}e^{{-{b:.1f}e^{{-{c:.2f}x}}}}"

        return Equation(
            name=lg_type,
            func=func,
            latex=latex,
            category='logistic_growth',
            params={'type': lg_type}
        )

    def _generate_fourier(self, x_range: Tuple[float, float]) -> Equation:
        """Generate Fourier series (sum of harmonics)"""
        num_harmonics = random.randint(2, 4)
        base_freq = random.uniform(0.5, 2.0)

        amplitudes = [random.uniform(0.3, 2.0) for _ in range(num_harmonics)]
        phases = [random.uniform(0, 2 * np.pi) for _ in range(num_harmonics)]

        # Decay amplitudes for higher harmonics (more realistic)
        for i in range(num_harmonics):
            amplitudes[i] /= (i + 1)

        def func(x):
            result = np.zeros_like(x, dtype=float)
            for i in range(num_harmonics):
                result += amplitudes[i] * np.sin((i + 1) * base_freq * x + phases[i])
            return result

        terms = [f"{amplitudes[i]:.2f}sin({(i+1)*base_freq:.2f}x)" for i in range(min(2, num_harmonics))]
        latex = "y = " + " + ".join(terms) + (" + ..." if num_harmonics > 2 else "")

        return Equation(
            name='fourier',
            func=func,
            latex=latex,
            category='fourier',
            params={'num_harmonics': num_harmonics, 'base_freq': base_freq}
        )

    def _generate_spline_noise(self, x_range: Tuple[float, float]) -> Equation:
        """Generate random spline through knot points (simulates real noisy data)"""
        from scipy.interpolate import CubicSpline

        num_knots = random.randint(5, 15)
        x_min, x_max = x_range

        # Random knot positions (sorted)
        knot_x = np.sort(np.random.uniform(x_min, x_max, num_knots))
        # Ensure endpoints are included
        knot_x[0] = x_min
        knot_x[-1] = x_max

        # Random knot values with controlled amplitude
        amplitude = random.uniform(1, 5)
        knot_y = np.random.uniform(-amplitude, amplitude, num_knots)

        # Optional trend
        if random.random() < 0.5:
            trend_slope = random.uniform(-0.5, 0.5)
            knot_y += trend_slope * (knot_x - x_min)

        cs = CubicSpline(knot_x, knot_y)

        def func(x):
            return cs(x)

        latex = f"cubic spline ({num_knots} knots)"

        return Equation(
            name='spline_noise',
            func=func,
            latex=latex,
            category='spline_noise',
            params={'num_knots': num_knots}
        )

    def _generate_sqrt_cbrt(self, x_range: Tuple[float, float]) -> Equation:
        """Generate square root and cube root functions"""
        sc_type = random.choice(['sqrt', 'cbrt'])

        a = random.uniform(0.5, 3) * random.choice([-1, 1])
        h = random.uniform(-2, 2)
        k = random.uniform(-2, 2)

        if sc_type == 'sqrt':
            def func(x):
                arg = x + h
                return a * np.sqrt(np.maximum(arg, 0)) + k
            latex = f"y = {a:.2f}√(x + {h:.2f}) + {k:.2f}"

        else:  # cbrt
            def func(x):
                return a * np.sign(x + h) * np.abs(x + h) ** (1/3) + k
            latex = f"y = {a:.2f}∛(x + {h:.2f}) + {k:.2f}"

        return Equation(
            name=sc_type,
            func=func,
            latex=latex,
            category='sqrt_cbrt',
            params={'type': sc_type, 'a': a, 'h': h, 'k': k}
        )

    def _generate_reciprocal_trig(self, x_range: Tuple[float, float]) -> Equation:
        """Generate clamped reciprocal trig functions (sec, csc)"""
        rt_type = random.choice(['sec', 'csc'])

        amplitude = random.uniform(0.5, 2.0)
        frequency = random.uniform(0.5, 2.0)
        clamp_val = random.uniform(3, 8)

        if rt_type == 'sec':
            def func(x):
                cos_val = np.cos(frequency * x)
                # Clamp to avoid infinity
                cos_val = np.where(np.abs(cos_val) < 0.1, 0.1 * np.sign(cos_val + 1e-10), cos_val)
                result = amplitude / cos_val
                return np.clip(result, -clamp_val, clamp_val)
            latex = f"y = {amplitude:.2f}sec({frequency:.2f}x)"

        else:  # csc
            def func(x):
                sin_val = np.sin(frequency * x)
                sin_val = np.where(np.abs(sin_val) < 0.1, 0.1 * np.sign(sin_val + 1e-10), sin_val)
                result = amplitude / sin_val
                return np.clip(result, -clamp_val, clamp_val)
            latex = f"y = {amplitude:.2f}csc({frequency:.2f}x)"

        return Equation(
            name=rt_type,
            func=func,
            latex=latex,
            category='reciprocal_trig',
            params={'type': rt_type, 'amplitude': amplitude, 'frequency': frequency}
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

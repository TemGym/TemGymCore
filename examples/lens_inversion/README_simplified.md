# Simplified Two-Lens Notebook

## Overview

The `two_lenses_simplified.ipynb` notebook provides a streamlined approach to analyzing two-lens optical systems using:

1. **Ray tracing** with temgym_core components
2. **ABCD transfer matrix** computation via JAX automatic differentiation
3. **Collins FFT** for Fresnel diffraction modeling
4. **Bayesian optimization** with Optuna for parameter fitting

## Features

### 1. Forward Model with Ray Tracing

The notebook builds a two-lens optical system and traces rays through it to compute the system's ABCD transfer matrix. The key components are:

- `Lens`: Thin lens with focal length
- `Detector`: Output plane with pixel grid
- `solve_model()`: Computes ABCD matrices via automatic differentiation

```python
model = [
    Lens(z=z1, focal_length=f1),
    Lens(z=z2, focal_length=f2),
    Detector(z=z3, pixel_size=..., shape=...)
]
```

### 2. ABCD Matrix via Differentiation

Instead of manually computing the transfer matrix, we use JAX's `jacobian` to automatically differentiate through the ray tracing:

```python
abcd = get_abcd_matrix(z1, z2, z3, f1, f2)
# Returns 5×5 matrix: [x, y, dx, dy, 1]ᵀ → [x', y', dx', dy', 1]ᵀ
```

The key insight: **B/A = z_defocus** (magnification cancels in defocus)

### 3. Collins FFT Diffraction

The Collins integral propagates a field through an optical system characterized by the ABCD matrix:

$$H(f_x, f_y) = \exp\left(-i\pi\lambda\frac{B}{A}(f_x^2 + f_y^2)\right)$$

where:
- $A$ is the magnification
- $B/A$ is the effective defocus distance
- $\lambda$ is the wavelength

This is equivalent to Fresnel diffraction with effective propagation distance $z_{eff} = B/A$.

### 4. Input/Output Grids

- **Input Grid**: 5 μm × 5 μm (default), 512×512 pixels
  - Contains circular aperture (1 μm diameter)
  - Adequate padding to avoid edge effects
  
- **Output Grid**: 10 mm × 10 mm, 256×256 pixels
  - Zoomed using `jax.image.resize`
  - Accounts for magnification from ABCD matrix

### 5. Bayesian Optimization with Optuna

The notebook includes a template for optimizing lens parameters using Optuna:

```python
def objective(trial):
    params = {
        'z2': trial.suggest_float('z2', 0.05, 0.2),
        'z3': trial.suggest_float('z3', 0.3, 0.8),
        'f1': trial.suggest_float('f1', 0.02, 0.1),
        'f2': trial.suggest_float('f2', 0.1, 0.25),
    }
    loss, _ = forward_model_loss(params, target_intensity)
    return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

## Usage

### Running the Notebook

```bash
cd examples/lens_inversion
jupyter notebook two_lenses_simplified.ipynb
```

Or use JupyterLab:

```bash
jupyter lab two_lenses_simplified.ipynb
```

### Running Tests

```bash
cd examples/lens_inversion
python test_two_lenses_simplified.py
```

Expected output:
```
============================================================
RUNNING TWO-LENS SIMPLIFIED NOTEBOOK TESTS
============================================================
...
✓ ALL TESTS PASSED!
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VOLTAGE` | 300 kV | Electron beam voltage |
| `APERTURE_RADIUS` | 0.5 μm | Aperture radius (1 μm diameter) |
| `INPUT_SIZE` | 5 μm | Input grid physical size |
| `INPUT_PIXELS` | 512 | Input grid resolution |
| `OUTPUT_SIZE` | 10 mm | Output detector size |
| `OUTPUT_PIXELS` | 256 | Output detector resolution |
| `Z1` | 0.0 m | First lens position |
| `Z2` | 0.1 m | Second lens position (100 mm) |
| `Z3` | 0.5 m | Detector position (500 mm) |
| `F1` | 0.05 m | First lens focal length (50 mm) |
| `F2` | 0.15 m | Second lens focal length (150 mm) |

## Physics Background

### ABCD Matrix

The 5×5 ABCD matrix represents a linear transformation of ray coordinates:

$$\begin{bmatrix} x' \\ y' \\ \theta_x' \\ \theta_y' \\ 1 \end{bmatrix} = 
\begin{bmatrix} 
A_{xx} & A_{xy} & B_{x\theta_x} & B_{x\theta_y} & 0 \\
A_{yx} & A_{yy} & B_{y\theta_x} & B_{y\theta_y} & 0 \\
C_{xx} & C_{xy} & D_{x\theta_x} & D_{x\theta_y} & 0 \\
C_{yx} & C_{yy} & D_{y\theta_x} & D_{y\theta_y} & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x \\ y \\ \theta_x \\ \theta_y \\ 1 \end{bmatrix}$$

For rotationally symmetric systems:
- $A_{xx} = A_{yy} = A$ (magnification)
- $B_{x\theta_x} = B_{y\theta_y} = B$ (defocus-related)

### Fresnel Diffraction

The Collins integral reduces to Fresnel diffraction when $B/A$ represents the propagation distance:

$$U(x,y) = \int\int U_0(x_0,y_0) \exp\left(\frac{i\pi}{\lambda z}\left[(x-Ax_0)^2 + (y-Ay_0)^2\right]\right) dx_0 dy_0$$

where $z = B/A$ and $A$ is the magnification.

## Optimization Strategy

### Loss Function

The loss function compares predicted and target intensity patterns:

```python
loss = jnp.mean((predicted_intensity - target_intensity)**2)
```

For better results, consider:
- Normalizing intensities
- Using multiple defocus planes
- Adding regularization for physical constraints

### Parameter Bounds

Set bounds based on physical constraints:

```python
bounds = {
    'z2': (50e-3, 200e-3),   # Second lens 50-200 mm from first
    'z3': (300e-3, 1000e-3), # Detector 300-1000 mm from first lens
    'f1': (20e-3, 100e-3),   # First lens focal length 20-100 mm
    'f2': (100e-3, 300e-3),  # Second lens focal length 100-300 mm
}
```

### Priors

Include manufacturer specs as priors:

```python
from scipy.stats import norm

priors = {
    'f1': norm(loc=50e-3, scale=5e-3),  # 50 mm ± 5 mm
    'f2': norm(loc=150e-3, scale=15e-3), # 150 mm ± 15 mm
}

prior_loss = -sum(prior.logpdf(params[key]) for key, prior in priors.items())
total_loss = data_loss + 0.1 * prior_loss
```

## Comparison with Full Simulation

The Collins FFT approach is much faster than full ray tracing but makes paraxial approximations. For validation:

1. Run Collins FFT (fast, ~ms)
2. Compare with full ray traced simulation (slower, ~seconds)
3. Use Collins FFT for optimization
4. Validate final result with full simulation

## Troubleshooting

### Memory Issues

If you run out of memory:
- Reduce `INPUT_PIXELS` (e.g., 256 instead of 512)
- Reduce `OUTPUT_PIXELS` (e.g., 128 instead of 256)

### Numerical Issues

If optimization fails:
- Check ABCD matrix for singularities ($|A| \approx 0$)
- Ensure parameters are in reasonable ranges
- Add bounds to prevent unphysical values

### Slow Optimization

To speed up:
- Use fewer pixels during optimization
- Use JIT compilation: `jax.jit(forward_model_loss)`
- Run on GPU if available

## References

1. **Collins Integral**: S. A. Collins, "Lens-System Diffraction Integral Written in Terms of Matrix Optics," J. Opt. Soc. Am. 60, 1168-1177 (1970)

2. **ABCD Matrices**: Siegman, A. E. "Lasers" (University Science Books, 1986)

3. **Optuna**: Akiba, T., et al. "Optuna: A Next-generation Hyperparameter Optimization Framework," KDD 2019

4. **JAX**: Bradbury, J., et al. "JAX: composable transformations of Python+NumPy programs" (2018)

## See Also

- `two_lenses.ipynb` - Original detailed notebook with full analysis
- `QUICK_REFERENCE.md` - Quick reference for key concepts
- `bayesian_dual_wobble_guide.md` - Guide for dual wobble optimization

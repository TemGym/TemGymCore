"""
Fitting routine for rotation, magnification, and defocus parameters.

This script fits synthetic images to extract rotation, magnification, and defocus
using JAX-differentiable forward models.
"""

import sys
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import pickle
from pathlib import Path
from functools import partial

jax.config.update("jax_enable_x64", True)


def collins_fft_propagate(
    input_field: jnp.ndarray,
    wavelength: float,
    defocus: float,
    magnification: float,
    rotation: float,
    pixel_size: float
) -> jnp.ndarray:
    """
    Propagate field using Collins FFT with magnification and rotation.
    
    The Collins formula for ABCD systems can be separated:
    - Magnification affects the spatial scaling
    - Defocus affects the Fresnel propagation
    - Rotation can be applied as coordinate transformation
    
    Parameters
    ----------
    input_field : jnp.ndarray
        Input complex field
    wavelength : float
        Electron wavelength (metres)
    defocus : float
        Defocus distance (metres)
    magnification : float
        System magnification
    rotation : float
        Rotation angle (radians)
    pixel_size : float
        Pixel size (metres)
        
    Returns
    -------
    output_field : jnp.ndarray
        Propagated field
    """
    ny, nx = input_field.shape
    
    # Apply rotation to input field via coordinate transformation
    if abs(rotation) > 1e-10:
        input_field = rotate_field(input_field, rotation)
    
    # Fresnel propagation with defocus
    # The key insight: In Collins formula with magnification M and defocus B,
    # the Fresnel kernel depends on B/M, which is the effective defocus
    effective_defocus = defocus / magnification
    
    # Build frequency coordinates
    fx = jnp.fft.fftfreq(nx, d=pixel_size)
    fy = jnp.fft.fftfreq(ny, d=pixel_size)
    FX, FY = jnp.meshgrid(fx, fy)
    
    # Fresnel transfer function
    H = jnp.exp(-1j * jnp.pi * wavelength * effective_defocus * (FX**2 + FY**2))
    
    # Propagate
    U_fft = jnp.fft.fft2(input_field)
    U_out_fft = H * U_fft
    output_field = jnp.fft.ifft2(U_out_fft)
    
    # Apply magnification as spatial scaling (zoom)
    output_field = scale_field(output_field, magnification)
    
    return output_field


def rotate_field(field: jnp.ndarray, angle: float) -> jnp.ndarray:
    """
    Rotate field by given angle using Fourier shift theorem.
    
    For small angles, we can use a simple coordinate transformation.
    For larger angles, we need proper interpolation.
    
    Parameters
    ----------
    field : jnp.ndarray
        Input field
    angle : float
        Rotation angle (radians)
        
    Returns
    -------
    rotated_field : jnp.ndarray
        Rotated field
    """
    ny, nx = field.shape
    
    # Create coordinate grids
    y = jnp.arange(ny) - ny // 2
    x = jnp.arange(nx) - nx // 2
    Y, X = jnp.meshgrid(y, x, indexing='ij')
    
    # Rotation matrix
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    
    # Rotated coordinates
    X_rot = cos_a * X - sin_a * Y
    Y_rot = sin_a * X + cos_a * Y
    
    # Map back to indices (nearest neighbor for JAX compatibility)
    X_idx = jnp.round(X_rot + nx // 2).astype(int)
    Y_idx = jnp.round(Y_rot + ny // 2).astype(int)
    
    # Clip to valid range
    X_idx = jnp.clip(X_idx, 0, nx - 1)
    Y_idx = jnp.clip(Y_idx, 0, ny - 1)
    
    # Resample
    rotated_field = field[Y_idx, X_idx]
    
    return rotated_field


def scale_field(field: jnp.ndarray, scale: float) -> jnp.ndarray:
    """
    Scale field by given factor (magnification).
    
    Uses frequency domain approach for differentiability.
    
    Parameters
    ----------
    field : jnp.ndarray
        Input field
    scale : float
        Scaling factor
        
    Returns
    -------
    scaled_field : jnp.ndarray
        Scaled field
    """
    # For now, use simple resampling
    # In practice, you'd want proper interpolation
    ny, nx = field.shape
    
    # Create coordinate grids for resampling
    y = jnp.arange(ny) - ny // 2
    x = jnp.arange(nx) - nx // 2
    Y, X = jnp.meshgrid(y, x, indexing='ij')
    
    # Scaled coordinates
    X_scaled = X / scale
    Y_scaled = Y / scale
    
    # Map back to indices
    X_idx = jnp.round(X_scaled + nx // 2).astype(int)
    Y_idx = jnp.round(Y_scaled + ny // 2).astype(int)
    
    # Clip and resample
    X_idx = jnp.clip(X_idx, 0, nx - 1)
    Y_idx = jnp.clip(Y_idx, 0, ny - 1)
    
    scaled_field = field[Y_idx, X_idx]
    
    return scaled_field


def forward_model(
    params: Dict,
    input_field: jnp.ndarray,
    wavelength: float,
    pixel_size: float
) -> jnp.ndarray:
    """
    Forward model: propagate input field with given parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary with keys: 'defocus', 'magnification', 'rotation'
    input_field : jnp.ndarray
        Input field
    wavelength : float
        Wavelength
    pixel_size : float
        Pixel size
        
    Returns
    -------
    intensity : jnp.ndarray
        Output intensity
    """
    output_field = collins_fft_propagate(
        input_field,
        wavelength,
        params['defocus'],
        params['magnification'],
        params['rotation'],
        pixel_size
    )
    
    return jnp.abs(output_field) ** 2


def loss_function(
    params_vec: jnp.ndarray,
    target_intensity: jnp.ndarray,
    input_field: jnp.ndarray,
    wavelength: float,
    pixel_size: float
) -> float:
    """
    Loss function for fitting.
    
    Parameters
    ----------
    params_vec : jnp.ndarray
        Parameter vector [defocus, magnification, rotation]
    target_intensity : jnp.ndarray
        Target intensity image
    input_field : jnp.ndarray
        Input field
    wavelength : float
        Wavelength
    pixel_size : float
        Pixel size
        
    Returns
    -------
    loss : float
        Mean squared error
    """
    params = {
        'defocus': params_vec[0],
        'magnification': params_vec[1],
        'rotation': params_vec[2]
    }
    
    predicted_intensity = forward_model(
        params, input_field, wavelength, pixel_size
    )
    
    # Normalize both images
    pred_norm = predicted_intensity / jnp.sum(predicted_intensity)
    target_norm = target_intensity / jnp.sum(target_intensity)
    
    # Mean squared error
    loss = jnp.mean((pred_norm - target_norm) ** 2)
    
    return loss


@partial(jax.jit, static_argnums=(2, 3))
def fit_step(
    params_vec: jnp.ndarray,
    target_intensity: jnp.ndarray,
    wavelength: float,
    pixel_size: float,
    input_field: jnp.ndarray,
    learning_rate: float = 0.01
) -> Tuple[jnp.ndarray, float]:
    """
    Single gradient descent step.
    
    Parameters
    ----------
    params_vec : jnp.ndarray
        Current parameters
    target_intensity : jnp.ndarray
        Target intensity
    wavelength : float
        Wavelength
    pixel_size : float
        Pixel size
    input_field : jnp.ndarray
        Input field
    learning_rate : float
        Learning rate
        
    Returns
    -------
    new_params : jnp.ndarray
        Updated parameters
    loss : float
        Current loss
    """
    loss, grad = jax.value_and_grad(loss_function)(
        params_vec, target_intensity, input_field, wavelength, pixel_size
    )
    
    new_params = params_vec - learning_rate * grad
    
    return new_params, loss


def fit_parameters(
    target_intensity: jnp.ndarray,
    input_field: jnp.ndarray,
    wavelength: float,
    pixel_size: float,
    initial_params: Dict,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = True
) -> Dict:
    """
    Fit parameters to target intensity.
    
    Parameters
    ----------
    target_intensity : jnp.ndarray
        Target intensity image
    input_field : jnp.ndarray
        Input field
    wavelength : float
        Wavelength
    pixel_size : float
        Pixel size
    initial_params : dict
        Initial parameter guess
    n_iterations : int
        Number of optimization iterations
    learning_rate : float
        Learning rate for gradient descent
    verbose : bool
        Print progress
        
    Returns
    -------
    fitted_params : dict
        Fitted parameters
    """
    # Convert to parameter vector
    params_vec = jnp.array([
        initial_params['defocus'],
        initial_params['magnification'],
        initial_params['rotation']
    ])
    
    losses = []
    
    for i in range(n_iterations):
        params_vec, loss = fit_step(
            params_vec, target_intensity, wavelength, pixel_size,
            input_field, learning_rate
        )
        
        losses.append(float(loss))
        
        if verbose and (i % 10 == 0 or i == n_iterations - 1):
            print(f"Iteration {i:3d}: Loss = {loss:.6f}, "
                  f"Params = [{params_vec[0]:.2e}, {params_vec[1]:.2f}, {params_vec[2]:.3f}]")
    
    fitted_params = {
        'defocus': float(params_vec[0]),
        'magnification': float(params_vec[1]),
        'rotation': float(params_vec[2]),
        'losses': losses
    }
    
    return fitted_params


def test_fitting_on_dataset(dataset_path: str, n_test: int = 5):
    """
    Test fitting routine on generated dataset.
    
    Parameters
    ----------
    dataset_path : str
        Path to dataset pickle file
    n_test : int
        Number of samples to test
    """
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    metadata = dataset['metadata']
    wavelength = 12.4e-10 / np.sqrt(metadata['voltage'] / 1e3)  # Approximate
    pixel_size = metadata['detector_size'] / metadata['detector_pixels']
    
    # Create input field (uniform square aperture)
    ny = nx = metadata['detector_pixels']
    input_field = jnp.ones((ny, nx), dtype=jnp.complex128)
    
    results = []
    
    print("\n" + "="*60)
    print("Testing Fitting Routine")
    print("="*60)
    
    for i in range(min(n_test, len(dataset['images']))):
        target_intensity = jnp.array(dataset['images'][i])
        true_params = dataset['parameters'][i]
        
        print(f"\n--- Sample {i+1}/{n_test} ---")
        print(f"True parameters:")
        print(f"  Defocus: {true_params['defocus']:.2e} m")
        print(f"  Magnification: {true_params['magnification']:.2f}")
        print(f"  Rotation: {np.rad2deg(true_params['rotation']):.2f}°")
        
        # Initial guess (perturbed from truth)
        initial_params = {
            'defocus': true_params['defocus'] * 1.2,
            'magnification': true_params['magnification'] * 1.1,
            'rotation': true_params['rotation'] + 0.1
        }
        
        # Fit
        fitted_params = fit_parameters(
            target_intensity,
            input_field,
            wavelength,
            pixel_size,
            initial_params,
            n_iterations=50,
            learning_rate=0.001,
            verbose=False
        )
        
        print(f"\nFitted parameters:")
        print(f"  Defocus: {fitted_params['defocus']:.2e} m "
              f"(error: {abs(fitted_params['defocus'] - true_params['defocus']) / true_params['defocus'] * 100:.1f}%)")
        print(f"  Magnification: {fitted_params['magnification']:.2f} "
              f"(error: {abs(fitted_params['magnification'] - true_params['magnification']) / true_params['magnification'] * 100:.1f}%)")
        print(f"  Rotation: {np.rad2deg(fitted_params['rotation']):.2f}° "
              f"(error: {abs(np.rad2deg(fitted_params['rotation'] - true_params['rotation'])):.2f}°)")
        
        results.append({
            'true': true_params,
            'fitted': fitted_params,
            'image_idx': i
        })
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("Fitting Routine for Rotation/Magnification/Defocus")
    print("="*60)
    
    dataset_path = "forward_model_data/dataset.pkl"
    
    if Path(dataset_path).exists():
        results = test_fitting_on_dataset(dataset_path, n_test=3)
        print("\n" + "="*60)
        print("Fitting tests complete!")
        print("="*60)
    else:
        print(f"\nDataset not found at {dataset_path}")
        print("Please run rotation_magnification_defocus_forward.py first")

"""
Debug script to check forward model and loss calculation.
"""

import sys
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import numpy as np
import pickle

jax.config.update("jax_enable_x64", True)

# Import functions from fitting script
from rotation_magnification_defocus_fitting import (
    forward_model,
    loss_function,
    rotate_field,
    scale_field_simple,
    fresnel_propagate
)

# Load dataset
with open('forward_model_data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

metadata = dataset['metadata']
wavelength = metadata['wavelength']
pixel_size = metadata['pixel_size']
grid_size = metadata['grid_size']

print(f"Metadata:")
print(f"  Wavelength: {wavelength*1e12:.3f} pm")
print(f"  Pixel size: {pixel_size*1e6:.3f} µm")
print(f"  Grid size: {grid_size}x{grid_size}")

# Get first sample
target_intensity = jnp.array(dataset['images'][0])
true_params = dataset['parameters'][0]

print(f"\nTarget image:")
print(f"  Shape: {target_intensity.shape}")
print(f"  Min/Max: {target_intensity.min():.3e} / {target_intensity.max():.3e}")
print(f"  Sum: {target_intensity.sum():.3e}")

# Create input aperture (same as in dataset generation)
input_aperture = jnp.zeros((grid_size, grid_size))
center = grid_size // 2
aperture_size = int(grid_size * 0.3)
half_ap = aperture_size // 2
input_aperture = input_aperture.at[
    center - half_ap:center + half_ap,
    center - half_ap:center + half_ap
].set(1.0)
input_field = input_aperture.astype(jnp.complex128)

print(f"\nInput aperture:")
print(f"  Shape: {input_aperture.shape}")
print(f"  Non-zero pixels: {(input_aperture > 0).sum()}")

# Test forward model with true parameters
true_params_array = jnp.array([
    true_params['defocus'],
    true_params['magnification'],
    true_params['rotation']
])

print(f"\nTrue parameters:")
print(f"  Defocus: {true_params['defocus']:.2e} m")
print(f"  Magnification: {true_params['magnification']:.3f}")
print(f"  Rotation: {np.rad2deg(true_params['rotation']):.2f}°")

# Run forward model
predicted_intensity = forward_model(
    true_params_array, input_field, wavelength, pixel_size
)

print(f"\nPredicted intensity:")
print(f"  Shape: {predicted_intensity.shape}")
print(f"  Min/Max: {predicted_intensity.min():.3e} / {predicted_intensity.max():.3e}")
print(f"  Sum: {predicted_intensity.sum():.3e}")

# Compute loss
loss = loss_function(
    true_params_array, target_intensity, input_field, wavelength, pixel_size
)

print(f"\nLoss with true parameters: {loss:.6e}")

# Test with perturbed parameters
perturbed_params = jnp.array([
    true_params['defocus'] * 1.3,
    true_params['magnification'] * 1.15,
    true_params['rotation'] + 0.2
])

print(f"\nPerturbed parameters:")
print(f"  Defocus: {perturbed_params[0]:.2e} m")
print(f"  Magnification: {perturbed_params[1]:.3f}")
print(f"  Rotation: {np.rad2deg(perturbed_params[2]):.2f}°")

# Run forward model with perturbed params
predicted_perturbed = forward_model(
    perturbed_params, input_field, wavelength, pixel_size
)

print(f"\nPredicted (perturbed):")
print(f"  Shape: {predicted_perturbed.shape}")
print(f"  Min/Max: {predicted_perturbed.min():.3e} / {predicted_perturbed.max():.3e}")
print(f"  Sum: {predicted_perturbed.sum():.3e}")

# Compute loss with perturbed params
loss_perturbed = loss_function(
    perturbed_params, target_intensity, input_field, wavelength, pixel_size
)

print(f"\nLoss with perturbed parameters: {loss_perturbed:.6e}")

# Test gradient
grad_fn = jax.grad(loss_function)
grads = grad_fn(perturbed_params, target_intensity, input_field, wavelength, pixel_size)

print(f"\nGradients at perturbed parameters:")
print(f"  d(loss)/d(defocus): {grads[0]:.6e}")
print(f"  d(loss)/d(mag):     {grads[1]:.6e}")
print(f"  d(loss)/d(rot):     {grads[2]:.6e}")

# Check if images are different
diff = jnp.abs(predicted_intensity - predicted_perturbed)
print(f"\nDifference between true and perturbed predictions:")
print(f"  Max diff: {diff.max():.6e}")
print(f"  Mean diff: {diff.mean():.6e}")

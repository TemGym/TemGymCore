#!/usr/bin/env python
"""
Test script for the simplified two-lens notebook.

This script validates all key functionality of the two-lens system:
1. Forward model with ray tracing
2. ABCD matrix computation via differentiation
3. Collins FFT diffraction
4. Zooming to output grid
5. Optuna optimization

Run with: python test_two_lenses_simplified.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import Lens, Detector
from temgym_core.ray import Ray
from temgym_core.run import solve_model
from temgym_core.constants import energy2wavelength

jax.config.update("jax_enable_x64", True)

# Constants
VOLTAGE = 300e3  # 300 kV
WAVELENGTH = energy2wavelength(VOLTAGE)
APERTURE_RADIUS = 0.5e-6
INPUT_SIZE = 5e-6
INPUT_PIXELS = 128  # Smaller for faster testing
OUTPUT_SIZE = 10e-3
OUTPUT_PIXELS = 64


def build_two_lens_model(z1, z2, z3, f1, f2):
    """Build a two-lens optical system."""
    lens1 = Lens(z=z1, focal_length=f1)
    lens2 = Lens(z=z2, focal_length=f2)
    detector = Detector(
        z=z3,
        pixel_size=(OUTPUT_SIZE/OUTPUT_PIXELS, OUTPUT_SIZE/OUTPUT_PIXELS),
        shape=(OUTPUT_PIXELS, OUTPUT_PIXELS),
        centre=(0.0, 0.0)
    )
    return [lens1, lens2, detector]


def get_abcd_matrix(z1, z2, z3, f1, f2):
    """Compute the ABCD transfer matrix."""
    ray = Ray.origin()
    model = build_two_lens_model(z1, z2, z3, f1, f2)
    abcd_matrices = solve_model(ray, model)
    
    # Compute cumulative ABCD
    cumulative_abcd = abcd_matrices[0]
    for i in range(1, len(abcd_matrices)):
        cumulative_abcd = abcd_matrices[i] @ cumulative_abcd
    
    return cumulative_abcd


def collins_fft_propagation(input_field, input_size, A, B, wavelength):
    """Propagate using Collins integral."""
    N = input_field.shape[0]
    dx = input_size / N
    
    fx = jnp.fft.fftfreq(N, d=dx)
    fy = jnp.fft.fftfreq(N, d=dx)
    FX, FY = jnp.meshgrid(fx, fy)
    
    z_eff = B / A if jnp.abs(A) > 1e-10 else 0.0
    H = jnp.exp(-1j * jnp.pi * wavelength * z_eff * (FX**2 + FY**2))
    
    U_input = jnp.fft.fft2(input_field)
    U_output = H * U_input
    output_field = jnp.fft.ifft2(U_output)
    
    k = 2 * jnp.pi / wavelength
    output_field *= jnp.exp(1j * k * jnp.abs(z_eff))
    
    return output_field


def create_circular_aperture(size, n_pixels, aperture_radius):
    """Create a circular aperture."""
    x = jnp.linspace(-size/2, size/2, n_pixels)
    y = jnp.linspace(-size/2, size/2, n_pixels)
    X, Y = jnp.meshgrid(x, y)
    R = jnp.sqrt(X**2 + Y**2)
    aperture = (R <= aperture_radius).astype(jnp.float32)
    return aperture, X, Y


def zoom_to_output_grid(field, input_size, output_size, output_pixels, magnification):
    """Zoom field to output grid."""
    output_shape = (output_pixels, output_pixels)
    
    if jnp.isrealobj(field):
        output_field = jax.image.resize(field, output_shape, method='bilinear')
    else:
        real_part = jax.image.resize(jnp.real(field), output_shape, method='bilinear')
        imag_part = jax.image.resize(jnp.imag(field), output_shape, method='bilinear')
        output_field = real_part + 1j * imag_part
    
    return output_field


def forward_model_loss(params, target_intensity=None):
    """Compute loss for the forward model."""
    z1, z2, z3 = params['z1'], params['z2'], params['z3']
    f1, f2 = params['f1'], params['f2']
    
    # Get ABCD matrix
    abcd = get_abcd_matrix(z1, z2, z3, f1, f2)
    A_x, B_x = abcd[0, 0], abcd[0, 2]
    
    # Create input field
    input_aperture, _, _ = create_circular_aperture(
        INPUT_SIZE, INPUT_PIXELS, APERTURE_RADIUS
    )
    input_field = input_aperture.astype(jnp.complex64)
    
    # Propagate
    output_field = collins_fft_propagation(
        input_field, INPUT_SIZE, A_x, B_x, WAVELENGTH
    )
    
    # Zoom to output grid
    output_field_zoomed = zoom_to_output_grid(
        output_field, INPUT_SIZE, OUTPUT_SIZE, OUTPUT_PIXELS, A_x
    )
    predicted_intensity = jnp.abs(output_field_zoomed)**2
    
    # Compute loss
    if target_intensity is not None:
        loss = jnp.mean((predicted_intensity - target_intensity)**2)
    else:
        loss = -jnp.max(predicted_intensity)
    
    return loss, predicted_intensity


def test_basic_functionality():
    """Test basic ray tracing and ABCD computation."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    Z1, Z2, Z3 = 0.0, 0.1, 0.5
    F1, F2 = 0.05, 0.15
    
    # Test model building
    model = build_two_lens_model(Z1, Z2, Z3, F1, F2)
    assert len(model) == 3, "Model should have 3 components"
    print(f"✓ Model built with {len(model)} components")
    
    # Test ABCD computation
    abcd = get_abcd_matrix(Z1, Z2, Z3, F1, F2)
    assert abcd.shape == (5, 5), "ABCD should be 5×5"
    print(f"✓ ABCD matrix shape: {abcd.shape}")
    print(f"  A_x = {abcd[0, 0]:.6f}, B_x = {abcd[0, 2]:.6f}")
    
    return True


def test_collins_fft():
    """Test Collins FFT propagation."""
    print("\n" + "="*60)
    print("TEST 2: Collins FFT Propagation")
    print("="*60)
    
    # Create aperture
    input_aperture, _, _ = create_circular_aperture(
        INPUT_SIZE, INPUT_PIXELS, APERTURE_RADIUS
    )
    input_field = input_aperture.astype(jnp.complex64)
    print(f"✓ Input aperture created: {input_aperture.shape}")
    
    # Propagate
    A, B = -6.0, 0.2
    output_field = collins_fft_propagation(input_field, INPUT_SIZE, A, B, WAVELENGTH)
    output_intensity = jnp.abs(output_field)**2
    
    assert output_field.shape == input_field.shape, "Output should match input shape"
    assert jnp.max(output_intensity) > 0, "Output should have non-zero intensity"
    print(f"✓ Propagation successful")
    print(f"  Output shape: {output_field.shape}")
    print(f"  Max intensity: {jnp.max(output_intensity):.6e}")
    
    return True


def test_zoom():
    """Test zooming to output grid."""
    print("\n" + "="*60)
    print("TEST 3: Zoom to Output Grid")
    print("="*60)
    
    # Create test field
    test_field = jnp.ones((INPUT_PIXELS, INPUT_PIXELS), dtype=jnp.complex64)
    
    # Zoom
    zoomed = zoom_to_output_grid(
        test_field, INPUT_SIZE, OUTPUT_SIZE, OUTPUT_PIXELS, -6.0
    )
    
    assert zoomed.shape == (OUTPUT_PIXELS, OUTPUT_PIXELS), \
        f"Zoomed shape should be {OUTPUT_PIXELS}×{OUTPUT_PIXELS}"
    print(f"✓ Zoom successful")
    print(f"  Input shape: {test_field.shape}")
    print(f"  Output shape: {zoomed.shape}")
    
    return True


def test_forward_model():
    """Test complete forward model."""
    print("\n" + "="*60)
    print("TEST 4: Complete Forward Model")
    print("="*60)
    
    params = {'z1': 0.0, 'z2': 0.1, 'z3': 0.5, 'f1': 0.05, 'f2': 0.15}
    
    loss, intensity = forward_model_loss(params)
    
    assert intensity.shape == (OUTPUT_PIXELS, OUTPUT_PIXELS), \
        "Output intensity should match output grid"
    assert not jnp.isnan(loss), "Loss should not be NaN"
    print(f"✓ Forward model successful")
    print(f"  Loss: {loss:.6e}")
    print(f"  Intensity shape: {intensity.shape}")
    print(f"  Max intensity: {jnp.max(intensity):.6e}")
    
    return True


def test_optuna_integration():
    """Test Optuna optimization."""
    print("\n" + "="*60)
    print("TEST 5: Optuna Integration")
    print("="*60)
    
    try:
        import optuna
        
        # Create synthetic target
        target_params = {'z1': 0.0, 'z2': 0.12, 'z3': 0.55, 'f1': 0.055, 'f2': 0.16}
        _, target_intensity = forward_model_loss(target_params)
        
        # Simple optimization test
        def objective(trial):
            params = {
                'z1': 0.0,
                'z2': trial.suggest_float('z2', 0.08, 0.15),
                'z3': trial.suggest_float('z3', 0.4, 0.7),
                'f1': trial.suggest_float('f1', 0.04, 0.07),
                'f2': trial.suggest_float('f2', 0.12, 0.2),
            }
            try:
                loss, _ = forward_model_loss(params, target_intensity)
                return float(loss)
            except:
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=3, show_progress_bar=False)
        
        assert len(study.trials) == 3, "Should complete 3 trials"
        assert study.best_value < float('inf'), "Should find valid solution"
        print(f"✓ Optuna optimization successful")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"  Best loss: {study.best_value:.6e}")
        
        return True
        
    except ImportError:
        print("⚠ Optuna not available, skipping test")
        return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING TWO-LENS SIMPLIFIED NOTEBOOK TESTS")
    print("="*60)
    print(f"Wavelength: {WAVELENGTH*1e12:.4f} pm")
    print(f"Input grid: {INPUT_SIZE*1e6:.2f} μm × {INPUT_PIXELS} pixels")
    print(f"Output grid: {OUTPUT_SIZE*1e3:.2f} mm × {OUTPUT_PIXELS} pixels")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Collins FFT", test_collins_fft),
        ("Zoom", test_zoom),
        ("Forward Model", test_forward_model),
        ("Optuna Integration", test_optuna_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

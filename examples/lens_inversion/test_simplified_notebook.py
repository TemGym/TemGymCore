#!/usr/bin/env python3
"""
Test script for two_lenses_simplified.ipynb

Validates that the notebook's core functionality works correctly.
"""

import sys
sys.path.insert(0, '../../src')

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.constants import energy2wavelength
from temgym_core.transfer_matrices import (
    calculate_z1_and_z2_from_M_and_f, 
    propagation_matrix, 
    lens_matrix
)

jax.config.update("jax_enable_x64", True)

def test_forward_model():
    """Test that the forward model computes A and B correctly."""
    print("Testing forward model...")
    
    # System parameters
    F1_TRUE = 0.003
    F2_TRUE = 0.050
    M1 = -50.0
    M2 = -20.0
    
    z1_obj, z1_img = calculate_z1_and_z2_from_M_and_f(M1, F1_TRUE)  
    z2_obj, z2_img = calculate_z1_and_z2_from_M_and_f(M2, F2_TRUE)
    
    D1_TRUE = abs(z1_obj)
    D2_TRUE = z1_img + abs(z2_obj)
    D3_TRUE = z2_img
    
    # Forward model
    @jax.jit
    def compute_AB_jax(d1, d2, d3, f1, f2):
        P1 = propagation_matrix(d1, xp=jnp)
        L1 = lens_matrix(f1, xp=jnp)
        P2 = propagation_matrix(d2, xp=jnp)
        L2 = lens_matrix(f2, xp=jnp)
        P3 = propagation_matrix(d3, xp=jnp)
        
        M = P3 @ L2 @ P2 @ L1 @ P1
        return M[0, 0], M[0, 1]
    
    A_check, B_check = compute_AB_jax(D1_TRUE, D2_TRUE, D3_TRUE, F1_TRUE, F2_TRUE)
    
    assert abs(A_check - M1*M2) < 0.01, f"A={A_check}, expected {M1*M2}"
    assert abs(B_check) < 1e-10, f"B={B_check}, expected ≈0"
    
    print(f"  ✓ A = {A_check:.4f} (expected {M1*M2:.0f})")
    print(f"  ✓ B = {B_check:.6e} (expected ≈0)")
    print("  ✓ Forward model test passed")


def test_measurements_generation():
    """Test that measurements can be generated correctly."""
    print("\nTesting measurements generation...")
    
    # System parameters
    F1_TRUE = 0.003
    F2_TRUE = 0.050
    M1 = -50.0
    M2 = -20.0
    
    z1_obj, z1_img = calculate_z1_and_z2_from_M_and_f(M1, F1_TRUE)  
    z2_obj, z2_img = calculate_z1_and_z2_from_M_and_f(M2, F2_TRUE)
    
    D1_TRUE = abs(z1_obj)
    D2_TRUE = z1_img + abs(z2_obj)
    D3_TRUE = z2_img
    
    # Forward model
    @jax.jit
    def compute_AB_jax(d1, d2, d3, f1, f2):
        P1 = propagation_matrix(d1, xp=jnp)
        L1 = lens_matrix(f1, xp=jnp)
        P2 = propagation_matrix(d2, xp=jnp)
        L2 = lens_matrix(f2, xp=jnp)
        P3 = propagation_matrix(d3, xp=jnp)
        
        M = P3 @ L2 @ P2 @ L1 @ P1
        return M[0, 0], M[0, 1]
    
    # Generate measurements
    WOBBLE_VALUES = np.array([0.0, 100.0, 200.0])
    Z_DEFOCUS_VALUES = np.array([0.0, 50.0, 100.0])
    
    measurements = []
    for wobble_lens in ['f1', 'f2']:
        for wobble_um in WOBBLE_VALUES:
            for defocus_mm in Z_DEFOCUS_VALUES:
                f1_use = F1_TRUE + (wobble_um * 1e-6 if wobble_lens == 'f1' else 0)
                f2_use = F2_TRUE + (wobble_um * 1e-6 if wobble_lens == 'f2' else 0)
                d3_use = D3_TRUE + defocus_mm * 1e-3
                
                A_meas, B_meas = compute_AB_jax(D1_TRUE, D2_TRUE, d3_use, f1_use, f2_use)
                
                measurements.append({
                    'wobble_lens': wobble_lens,
                    'wobble_um': wobble_um,
                    'defocus_mm': defocus_mm,
                    'A_meas': float(A_meas),
                    'B_meas': float(B_meas),
                })
    
    assert len(measurements) == 18, f"Expected 18 measurements, got {len(measurements)}"
    
    # Check that measurements are diverse (not all the same)
    A_values = [m['A_meas'] for m in measurements]
    B_values = [m['B_meas'] for m in measurements]
    
    assert len(set(A_values)) > 1, "All A values are the same!"
    assert len(set(B_values)) > 1, "All B values are the same!"
    
    print(f"  ✓ Generated {len(measurements)} measurements")
    print(f"  ✓ A values range: [{min(A_values):.2f}, {max(A_values):.2f}]")
    print(f"  ✓ B values range: [{min(B_values):.6e}, {max(B_values):.6e}]")
    print("  ✓ Measurements generation test passed")


def test_objective_function():
    """Test that the objective function computes loss correctly."""
    print("\nTesting objective function...")
    
    # System parameters
    F1_TRUE = 0.003
    F2_TRUE = 0.050
    M1 = -50.0
    M2 = -20.0
    
    z1_obj, z1_img = calculate_z1_and_z2_from_M_and_f(M1, F1_TRUE)  
    z2_obj, z2_img = calculate_z1_and_z2_from_M_and_f(M2, F2_TRUE)
    
    D1_TRUE = abs(z1_obj)
    D2_TRUE = z1_img + abs(z2_obj)
    D3_TRUE = z2_img
    
    # Forward model
    @jax.jit
    def compute_AB_jax(d1, d2, d3, f1, f2):
        P1 = propagation_matrix(d1, xp=jnp)
        L1 = lens_matrix(f1, xp=jnp)
        P2 = propagation_matrix(d2, xp=jnp)
        L2 = lens_matrix(f2, xp=jnp)
        P3 = propagation_matrix(d3, xp=jnp)
        
        M = P3 @ L2 @ P2 @ L1 @ P1
        return M[0, 0], M[0, 1]
    
    # Generate measurements
    WOBBLE_VALUES = np.array([0.0, 100.0, 200.0])
    Z_DEFOCUS_VALUES = np.array([0.0, 50.0, 100.0])
    
    measurements = []
    for wobble_lens in ['f1', 'f2']:
        for wobble_um in WOBBLE_VALUES:
            for defocus_mm in Z_DEFOCUS_VALUES:
                f1_use = F1_TRUE + (wobble_um * 1e-6 if wobble_lens == 'f1' else 0)
                f2_use = F2_TRUE + (wobble_um * 1e-6 if wobble_lens == 'f2' else 0)
                d3_use = D3_TRUE + defocus_mm * 1e-3
                
                A_meas, B_meas = compute_AB_jax(D1_TRUE, D2_TRUE, d3_use, f1_use, f2_use)
                
                measurements.append({
                    'wobble_lens': wobble_lens,
                    'wobble_um': wobble_um,
                    'defocus_mm': defocus_mm,
                    'A_meas': float(A_meas),
                    'B_meas': float(B_meas),
                })
    
    # Create objective function
    @jax.jit
    def compute_residuals_jax(params):
        d1, d2, d3, f1, f2 = params
        A_scale = 1000.0
        B_scale = 0.1
        
        residuals = []
        for m in measurements:
            f1_use = f1 + (m['wobble_um'] * 1e-6 if m['wobble_lens'] == 'f1' else 0)
            f2_use = f2 + (m['wobble_um'] * 1e-6 if m['wobble_lens'] == 'f2' else 0)
            d3_use = d3 + m['defocus_mm'] * 1e-3
            
            A_pred, B_pred = compute_AB_jax(d1, d2, d3_use, f1_use, f2_use)
            
            r_A = (A_pred - m['A_meas']) / A_scale
            r_B = (B_pred - m['B_meas']) / jnp.maximum(B_scale, jnp.abs(m['B_meas']))
            
            residuals.append(r_A**2 + r_B**2)
        
        return jnp.sum(jnp.array(residuals))
    
    # Test with true parameters
    params_true = jnp.array([D1_TRUE, D2_TRUE, D3_TRUE, F1_TRUE, F2_TRUE])
    loss_true = compute_residuals_jax(params_true)
    
    assert loss_true < 1e-6, f"Loss at true parameters should be ~0, got {loss_true}"
    
    # Test with perturbed parameters
    params_perturbed = params_true * jnp.array([1.1, 0.9, 1.05, 0.95, 1.02])
    loss_perturbed = compute_residuals_jax(params_perturbed)
    
    assert loss_perturbed > loss_true, "Loss should increase with perturbation"
    
    print(f"  ✓ Loss at true parameters: {loss_true:.6e}")
    print(f"  ✓ Loss at perturbed parameters: {loss_perturbed:.6e}")
    print("  ✓ Objective function test passed")


def test_n_lens_extensibility():
    """Test that the N-lens framework works."""
    print("\nTesting N-lens extensibility...")
    
    # System parameters
    F1_TRUE = 0.003
    F2_TRUE = 0.050
    M1 = -50.0
    M2 = -20.0
    
    z1_obj, z1_img = calculate_z1_and_z2_from_M_and_f(M1, F1_TRUE)  
    z2_obj, z2_img = calculate_z1_and_z2_from_M_and_f(M2, F2_TRUE)
    
    D1_TRUE = abs(z1_obj)
    D2_TRUE = z1_img + abs(z2_obj)
    D3_TRUE = z2_img
    
    # N-lens function
    def compute_AB_N_lenses(distances, focal_lengths):
        M = propagation_matrix(distances[0], xp=jnp)
        
        for i, f in enumerate(focal_lengths):
            M = lens_matrix(f, xp=jnp) @ M
            M = propagation_matrix(distances[i+1], xp=jnp) @ M
        
        return M[0, 0], M[0, 1]
    
    # Test with 2-lens system
    A_n, B_n = compute_AB_N_lenses(
        jnp.array([D1_TRUE, D2_TRUE, D3_TRUE]),
        jnp.array([F1_TRUE, F2_TRUE])
    )
    
    # Reference 2-lens computation
    @jax.jit
    def compute_AB_jax(d1, d2, d3, f1, f2):
        P1 = propagation_matrix(d1, xp=jnp)
        L1 = lens_matrix(f1, xp=jnp)
        P2 = propagation_matrix(d2, xp=jnp)
        L2 = lens_matrix(f2, xp=jnp)
        P3 = propagation_matrix(d3, xp=jnp)
        
        M = P3 @ L2 @ P2 @ L1 @ P1
        return M[0, 0], M[0, 1]
    
    A_ref, B_ref = compute_AB_jax(D1_TRUE, D2_TRUE, D3_TRUE, F1_TRUE, F2_TRUE)
    
    assert np.allclose(A_n, A_ref), f"A mismatch: N-lens={A_n}, 2-lens={A_ref}"
    assert np.allclose(B_n, B_ref), f"B mismatch: N-lens={B_n}, 2-lens={B_ref}"
    
    print(f"  ✓ A: N-lens={A_n:.6f}, 2-lens={A_ref:.6f}")
    print(f"  ✓ B: N-lens={B_n:.6e}, 2-lens={B_ref:.6e}")
    print("  ✓ N-lens extensibility test passed")


def main():
    """Run all tests."""
    print("="*70)
    print("TESTING TWO_LENSES_SIMPLIFIED.IPYNB FUNCTIONALITY")
    print("="*70)
    
    try:
        test_forward_model()
        test_measurements_generation()
        test_objective_function()
        test_n_lens_extensibility()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

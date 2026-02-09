"""
Definitive Solvability Analysis for the Two-Lens Inverse Problem
================================================================

Question: Given measured (A, B) values at known defocus positions 
and unknown focal length perturbations, can we uniquely recover 
(d1, d2, d3, f1, f2)?

KEY INSIGHT (missed by previous analysis):
==========================================
When defocus Δz is applied AFTER the last lens (changing d3), the 
full ABCD matrix satisfies:

    M(d3 + Δz) = P(Δz) · M(d3)

Therefore:
    A(Δz) = A₀ + C₀ · Δz     (LINEAR in Δz)
    B(Δz) = B₀ + D₀ · Δz     (LINEAR in Δz)

So from ≥2 defocus measurements at the SAME wobble setting, you
recover ALL FOUR ABCD elements by linear regression:
    - A₀, B₀ from intercepts
    - C₀, D₀ from slopes

This means each wobble setting gives 3 INDEPENDENT constraints 
(since det(ABCD)=1), NOT just 2 from (A,B).

This changes the equation counting fundamentally.

Author: Solvability analysis
Date: February 2026
"""

import numpy as np
from scipy.optimize import least_squares
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# FORWARD MODEL
# ================================================================

def propagation_matrix(z):
    """2x2 free-space propagation."""
    return np.array([[1.0, z], [0.0, 1.0]])

def lens_matrix(f):
    """2x2 thin lens."""
    return np.array([[1.0, 0.0], [-1.0/f, 1.0]])

def two_lens_ABCD(d1, d2, d3, f1, f2):
    """Full ABCD matrix for source → L1 → L2 → detector."""
    M = propagation_matrix(d3) @ lens_matrix(f2) @ propagation_matrix(d2) @ \
        lens_matrix(f1) @ propagation_matrix(d1)
    return M[0, 0], M[0, 1], M[1, 0], M[1, 1]  # A, B, C, D

def two_lens_AB(d1, d2, d3, f1, f2):
    """Just A and B."""
    A, B, C, D = two_lens_ABCD(d1, d2, d3, f1, f2)
    return A, B


# ================================================================
# TEST 0: PROVE THAT DEFOCUS GIVES YOU ABCD, NOT JUST AB
# ================================================================

def test0_defocus_gives_ABCD():
    """
    Demonstrate that A(Δz) and B(Δz) are LINEAR in Δz,
    with slopes C and D respectively.
    """
    print("=" * 70)
    print("TEST 0: Defocus at detector → Full ABCD recovery")
    print("=" * 70)
    
    d1, d2, d3 = 3.06e-3, 205.5e-3, 1.05
    f1, f2 = 3.0e-3, 50.0e-3
    
    A0, B0, C0, D0 = two_lens_ABCD(d1, d2, d3, f1, f2)
    
    print(f"\nTrue ABCD at base d3:")
    print(f"  A = {A0:.6f}")
    print(f"  B = {B0:.6e}")
    print(f"  C = {C0:.6f}")
    print(f"  D = {D0:.6f}")
    print(f"  det = {A0*D0 - B0*C0:.10f} (should be 1)")
    
    # Measure A, B at multiple defocus values
    dz_values = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])  # meters
    A_measured = []
    B_measured = []
    
    for dz in dz_values:
        A, B, _, _ = two_lens_ABCD(d1, d2, d3 + dz, f1, f2)
        A_measured.append(A)
        B_measured.append(B)
    
    A_measured = np.array(A_measured)
    B_measured = np.array(B_measured)
    
    # Linear fit: A(Δz) = A₀ + C₀·Δz
    A_slope, A_intercept = np.polyfit(dz_values, A_measured, 1)
    B_slope, B_intercept = np.polyfit(dz_values, B_measured, 1)
    
    print(f"\nRecovered from linear fit of A(Δz) and B(Δz):")
    print(f"  A₀ = {A_intercept:.6f}  (true: {A0:.6f}, error: {abs(A_intercept-A0):.2e})")
    print(f"  C₀ = {A_slope:.6f}  (true: {C0:.6f}, error: {abs(A_slope-C0):.2e})")
    print(f"  B₀ = {B_intercept:.6e}  (true: {B0:.6e}, error: {abs(B_intercept-B0):.2e})")
    print(f"  D₀ = {B_slope:.6f}  (true: {D0:.6f}, error: {abs(B_slope-D0):.2e})")
    
    print(f"\n  → Defocus series recovers ALL FOUR ABCD elements to machine precision!")
    print(f"  → Each wobble setting gives 3 independent constraints (det=1 removes 1 DOF)")
    
    # Verify linearity
    A_pred = A_intercept + A_slope * dz_values
    max_nonlinearity = np.max(np.abs(A_measured - A_pred))
    print(f"\n  Max nonlinearity in A(Δz): {max_nonlinearity:.2e} (should be ~0)")
    print(f"  → A and B are EXACTLY linear in Δz (not approximate!)")
    
    return True


# ================================================================
# TEST 1: EQUATION COUNTING
# ================================================================

def test1_equation_counting():
    """
    Systematic equation counting for all scenarios.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Equation Counting for All Scenarios")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    EQUATION COUNTING TABLE                         │
├─────────────────────────────────────────────────────────────────────┤
│ Scenario             │ Unknowns │ Constraints │ Status             │
├──────────────────────┼──────────┼─────────────┼────────────────────┤
│ AB only, 1 setting   │    5     │     2       │ ✗ Underdetermined  │
│ AB only, K settings  │   3+2K   │    2K       │ ✗ Never solvable!  │
│ (free focal lengths) │          │             │   (always 3 short) │
├──────────────────────┼──────────┼─────────────┼────────────────────┤
│ ABCD, 1 setting      │    5     │     3       │ ✗ 2 DOF free       │
│ ABCD, 2 settings     │    7     │     6       │ ✗ 1 DOF free       │
│  (free f1,f2 each)   │          │             │                    │
│ ABCD, 3 settings     │    9     │     9       │ ~ Exactly det.     │
│  (free f1,f2 each)   │          │             │                    │
│ ABCD, 4+ settings    │  3+2K    │    3K       │ ✓ Overdetermined   │
├──────────────────────┼──────────┼─────────────┼────────────────────┤
│ ABCD, 2 settings     │    6     │     6       │ ✓ Exactly det.     │
│  (wobble 1 lens)     │          │             │   (minimum!)       │
│ ABCD, 3+ settings    │   4+K    │    3K       │ ✓ Overdetermined   │
│  (wobble 1 lens)     │          │             │                    │
├──────────────────────┼──────────┼─────────────┼────────────────────┤
│ ABCD, linear model   │    7     │    3K       │                    │
│  K=2 settings        │    7     │     6       │ ✗ 1 DOF free       │
│  K=3 settings        │    7     │     9       │ ✓ Overdetermined   │
│  K=5 settings        │    7     │    15       │ ✓ Well overdetermined│
├──────────────────────┼──────────┼─────────────┼────────────────────┤
│ AB only, linear      │    7     │    2K       │                    │
│  K=3                 │    7     │     6       │ ✗ 1 DOF free       │
│  K=4                 │    7     │     8       │ ~ Barely overdet.  │
│  K=5                 │    7     │    10       │ ✓ Overdetermined   │
└──────────────────────┴──────────┴─────────────┴────────────────────┘

KEY INSIGHT: With defocus you get ABCD (3 independent per setting),
             without defocus you get only AB (2 per setting).
             This is the difference between solvable and not solvable!

Notation:
  - "free f1,f2" = each setting has completely independent focal lengths
  - "wobble 1 lens" = d1,d2,d3,f2 shared; only f1 changes per setting
  - "linear model" = φ₁(w)=a₁+b₁w, φ₂(w)=a₂+b₂w (7 unknowns total)
  - K = number of different wobble/focal-length settings
  - constraints = 3 per setting (from ABCD with det=1) or 2 (from AB only)
""")
    return True


# ================================================================
# TEST 2: SINGLE ABCD (should fail — 5 unknowns, 3 constraints)
# ================================================================

def test2_single_ABCD():
    """Can we recover (d1,d2,d3,f1,f2) from a single ABCD? (Should fail.)"""
    print("\n" + "=" * 70)
    print("TEST 2: Single ABCD → 5 unknowns, 3 constraints (MUST FAIL)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_t, f2_t = 3.0e-3, 50.0e-3
    
    A_t, B_t, C_t, D_t = two_lens_ABCD(d1_t, d2_t, d3_t, f1_t, f2_t)
    target = np.array([A_t, C_t, D_t])  # 3 independent (B from det=1)
    
    def residual(params):
        d1, d2, d3, f1, f2 = np.abs(params)
        A, B, C, D = two_lens_ABCD(d1, d2, d3, f1, f2)
        return np.array([A - A_t, C - C_t, D - D_t])
    
    n_trials = 20
    solutions = []
    for trial in range(n_trials):
        np.random.seed(trial)
        scale = 0.3 + 1.4 * np.random.rand(5)
        x0 = np.array([d1_t, d2_t, d3_t, f1_t, f2_t]) * scale
        
        # Use 'trf' since system is underdetermined (3 residuals < 5 vars)
        result = least_squares(residual, x0, method='trf', max_nfev=5000,
                              bounds=(1e-6, np.inf))
        if result.cost < 1e-20:
            solutions.append(np.abs(result.x))
    
    print(f"\nFound {len(solutions)} solutions fitting the same ABCD:")
    
    # Check if solutions are distinct
    unique_solutions = []
    for sol in solutions:
        is_new = True
        for usol in unique_solutions:
            if np.allclose(sol, usol, rtol=0.01):
                is_new = False
                break
        if is_new:
            unique_solutions.append(sol)
    
    print(f"  Distinct solutions: {len(unique_solutions)}")
    for i, sol in enumerate(unique_solutions[:5]):
        err = np.abs(sol - np.array([d1_t, d2_t, d3_t, f1_t, f2_t])) / \
              np.array([d1_t, d2_t, d3_t, f1_t, f2_t]) * 100
        A_s, B_s, C_s, D_s = two_lens_ABCD(*sol)
        print(f"\n  Solution {i+1}: d1={sol[0]*1e3:.4f}mm, d2={sol[1]*1e3:.2f}mm, "
              f"d3={sol[2]*1e3:.1f}mm, f1={sol[3]*1e3:.4f}mm, f2={sol[4]*1e3:.2f}mm")
        print(f"    ABCD: A={A_s:.4f}, C={C_s:.4f}, D={D_s:.6f} (matches target: "
              f"{np.allclose([A_s,C_s,D_s], [A_t,C_t,D_t], atol=1e-8)})")
        if np.allclose(sol, [d1_t, d2_t, d3_t, f1_t, f2_t], rtol=0.01):
            print(f"    ← This IS the true solution")
        else:
            print(f"    ← DIFFERENT from true solution (max error: {err.max():.1f}%)")
    
    if len(unique_solutions) > 1:
        print(f"\n  ✗ CONFIRMED: Single ABCD has MULTIPLE solutions → NOT UNIQUE")
    else:
        print(f"\n  ⚠ Only found 1 solution numerically, but system IS underdetermined")
    
    return len(unique_solutions) > 1


# ================================================================
# TEST 3: TWO ABCD (wobble 1 lens) — 6 unknowns, 6 constraints
# ================================================================

def test3_two_ABCD_one_lens():
    """
    Two wobble settings on lens 1 only.
    Unknowns: d1, d2, d3, f1_base, f1_perturbed, f2 = 6
    Constraints: 2 × 3 = 6
    Should be exactly determined (unique solution).
    """
    print("\n" + "=" * 70)
    print("TEST 3: Two ABCD (wobble lens 1) → 6 unknowns, 6 constraints")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_t, f2_t = 3.0e-3, 50.0e-3
    f1_p = 3.1e-3  # Perturbed f1
    
    # Targets: ABCD at two settings
    A1, B1, C1, D1 = two_lens_ABCD(d1_t, d2_t, d3_t, f1_t, f2_t)
    A2, B2, C2, D2 = two_lens_ABCD(d1_t, d2_t, d3_t, f1_p, f2_t)
    
    # Use A, C, D from each (6 values for 6 unknowns)
    target = np.array([A1, C1, D1, A2, C2, D2])
    
    def residual(params):
        d1, d2, d3, f1a, f1b, f2 = np.abs(params)
        A1p, _, C1p, D1p = two_lens_ABCD(d1, d2, d3, f1a, f2)
        A2p, _, C2p, D2p = two_lens_ABCD(d1, d2, d3, f1b, f2)
        return np.array([A1p-A1, C1p-C1, D1p-D1, A2p-A2, C2p-C2, D2p-D2])
    
    n_trials = 50
    solutions = []
    true_params = np.array([d1_t, d2_t, d3_t, f1_t, f1_p, f2_t])
    
    for trial in range(n_trials):
        np.random.seed(trial)
        scale = 0.3 + 1.4 * np.random.rand(6)
        x0 = true_params * scale
        
        result = least_squares(residual, x0, method='lm', max_nfev=10000, 
                              ftol=1e-15, xtol=1e-15)
        if result.cost < 1e-20:
            sol = np.abs(result.x)
            solutions.append(sol)
    
    # Deduplicate
    unique_solutions = []
    for sol in solutions:
        is_new = True
        for usol in unique_solutions:
            if np.allclose(sol, usol, rtol=0.005):
                is_new = False
                break
        if is_new:
            unique_solutions.append(sol)
    
    print(f"\nFound {len(solutions)} converged solutions, {len(unique_solutions)} distinct:")
    
    for i, sol in enumerate(unique_solutions[:8]):
        err = np.abs(sol - true_params) / true_params * 100
        match = "✓ TRUE" if np.allclose(sol, true_params, rtol=0.01) else f"✗ max_err={err.max():.1f}%"
        print(f"  Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, d3={sol[2]*1e3:.1f}, "
              f"f1a={sol[3]*1e3:.4f}, f1b={sol[4]*1e3:.4f}, f2={sol[5]*1e3:.2f} mm  {match}")
    
    # Check Jacobian rank
    J = np.zeros((6, 6))
    eps = 1e-10
    f0 = residual(true_params)
    for i in range(6):
        p_plus = true_params.copy()
        p_plus[i] += eps
        J[:, i] = (residual(p_plus) - f0) / eps
    
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    cond = np.linalg.cond(J)
    sv = np.linalg.svd(J, compute_uv=False)
    
    print(f"\n  Jacobian rank: {rank}/6, condition: {cond:.2e}")
    print(f"  Singular values: {sv}")
    
    if len(unique_solutions) == 1:
        print(f"\n  ✓ UNIQUE SOLUTION found! (2 wobble settings of 1 lens suffices)")
    elif len(unique_solutions) <= 3:
        print(f"\n  ~ {len(unique_solutions)} discrete solutions (finite ambiguity, priors can resolve)")
    else:
        print(f"\n  ✗ Multiple solutions — may need more constraints")
    
    return len(unique_solutions), rank


# ================================================================
# TEST 4: THREE ABCD (wobble 1 lens) — 7 unknowns, 9 constraints
# ================================================================

def test4_three_ABCD_one_lens():
    """
    Three wobble settings on lens 1.
    Unknowns: d1, d2, d3, f1_a, f1_b, f1_c, f2 = 7
    Constraints: 3 × 3 = 9
    Should be overdetermined — most robust.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Three ABCD (wobble lens 1) → 7 unknowns, 9 constraints")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_t, f2_t = 3.0e-3, 50.0e-3
    f1_settings = [3.0e-3, 3.1e-3, 3.2e-3]  # Three f1 values
    
    # Compute ABCD for each setting
    targets = []
    for f1_s in f1_settings:
        A, B, C, D = two_lens_ABCD(d1_t, d2_t, d3_t, f1_s, f2_t)
        targets.extend([A, C, D])
    
    targets = np.array(targets)
    true_params = np.array([d1_t, d2_t, d3_t] + f1_settings + [f2_t])
    
    def residual(params):
        d1, d2, d3 = np.abs(params[:3])
        f1_vals = np.abs(params[3:6])
        f2 = np.abs(params[6])
        
        res = []
        for f1_s in f1_vals:
            A, B, C, D = two_lens_ABCD(d1, d2, d3, f1_s, f2)
            res.extend([A, C, D])
        
        return np.array(res) - targets
    
    n_trials = 50
    solutions = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        scale = 0.3 + 1.4 * np.random.rand(7)
        x0 = true_params * scale
        
        result = least_squares(residual, x0, method='lm', max_nfev=10000,
                              ftol=1e-15, xtol=1e-15)
        if result.cost < 1e-18:
            solutions.append(np.abs(result.x))
    
    # Deduplicate
    unique_solutions = []
    for sol in solutions:
        is_new = True
        for usol in unique_solutions:
            if np.allclose(sol, usol, rtol=0.005):
                is_new = False
                break
        if is_new:
            unique_solutions.append(sol)
    
    print(f"\nFound {len(solutions)} converged, {len(unique_solutions)} distinct solutions:")
    
    for i, sol in enumerate(unique_solutions[:8]):
        err = np.abs(sol - true_params) / true_params * 100
        match = "✓ TRUE" if np.allclose(sol, true_params, rtol=0.01) else f"✗ max_err={err.max():.1f}%"
        print(f"  Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, d3={sol[2]*1e3:.1f}, "
              f"f1s=[{sol[3]*1e3:.3f},{sol[4]*1e3:.3f},{sol[5]*1e3:.3f}], f2={sol[6]*1e3:.2f} mm  {match}")
    
    # Jacobian
    J = np.zeros((9, 7))
    eps = 1e-10
    f0 = residual(true_params)
    for i in range(7):
        p_plus = true_params.copy()
        p_plus[i] += eps
        J[:, i] = (residual(p_plus) - f0) / eps
    
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    cond = np.linalg.cond(J)
    
    print(f"\n  Jacobian rank: {rank}/7, condition: {cond:.2e}")
    
    if len(unique_solutions) == 1:
        print(f"\n  ✓ UNIQUE SOLUTION! Overdetermined system resolves all ambiguity")
    else:
        print(f"\n  ~ {len(unique_solutions)} solutions remain")
    
    return len(unique_solutions), rank


# ================================================================
# TEST 5: LINEAR MODEL — φ₁(w) = a₁ + b₁·w  (wobble 1 lens)
# ================================================================

def test5_linear_model_one_lens():
    """
    Linear focal length model for one lens.
    φ₁(w) = a₁ + b₁·w  (w is KNOWN wobble setting)
    Unknowns: d1, d2, d3, a₁, b₁, f2 = 6
    
    Test with K = 2, 3, 5 wobble settings.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Linear model φ₁(w) = a₁ + b₁·w (wobble 1 lens only)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_base = 3.0e-3
    f2_t = 50.0e-3
    
    a1_t = 1.0 / f1_base      # = 333.33 m⁻¹
    b1_t = 5000.0              # sensitivity: Δφ₁ per unit wobble
    
    true_params_base = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, f2_t])
    
    for K in [2, 3, 5]:
        w_values = np.linspace(0, 0.01, K)  # wobble settings
        
        # Generate targets
        targets = []
        for w in w_values:
            phi1_w = a1_t + b1_t * w
            f1_w = 1.0 / phi1_w
            A, B, C, D = two_lens_ABCD(d1_t, d2_t, d3_t, f1_w, f2_t)
            targets.extend([A, C, D])
        targets = np.array(targets)
        
        def make_residual(w_vals, tgt):
            def residual(params):
                d1, d2, d3, a1, b1, f2 = np.abs(params)
                res = []
                for w in w_vals:
                    phi1 = a1 + b1 * w
                    f1 = 1.0 / phi1
                    A, B, C, D = two_lens_ABCD(d1, d2, d3, f1, f2)
                    res.extend([A, C, D])
                return np.array(res) - tgt
            return residual
        
        residual = make_residual(w_values, targets)
        
        n_constraints = 3 * K
        n_unknowns = 6
        
        # Jacobian
        J = np.zeros((n_constraints, n_unknowns))
        eps = 1e-10
        f0 = residual(true_params_base)
        for i in range(n_unknowns):
            p_plus = true_params_base.copy()
            p_plus[i] += eps
            J[:, i] = (residual(p_plus) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        cond = np.linalg.cond(J)
        
        # Multi-start
        n_trials = 50
        solutions = []
        for trial in range(n_trials):
            np.random.seed(trial + K * 100)
            scale = 0.5 + np.random.rand(6)
            x0 = true_params_base * scale
            
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-18:
                solutions.append(np.abs(result.x))
        
        unique_solutions = []
        for sol in solutions:
            is_new = True
            for usol in unique_solutions:
                if np.allclose(sol, usol, rtol=0.005):
                    is_new = False
                    break
            if is_new:
                unique_solutions.append(sol)
        
        status_eq = "overdetermined" if n_constraints > n_unknowns else \
                    "exactly determined" if n_constraints == n_unknowns else "underdetermined"
        status_uniq = f"✓ UNIQUE" if len(unique_solutions) == 1 else \
                      f"~ {len(unique_solutions)} solutions"
        
        print(f"\n  K={K} settings: {n_constraints} constraints, {n_unknowns} unknowns "
              f"({status_eq})")
        print(f"    Jacobian rank: {rank}/{n_unknowns}, condition: {cond:.2e}")
        print(f"    Converged: {len(solutions)}/50, distinct: {len(unique_solutions)}")
        print(f"    Status: {status_uniq}")
        
        if len(unique_solutions) <= 3:
            for i, sol in enumerate(unique_solutions):
                err = np.abs(sol - true_params_base) / true_params_base * 100
                is_true = "✓ TRUE" if err.max() < 1.0 else f"err={err.max():.1f}%"
                print(f"      Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, "
                      f"d3={sol[2]*1e3:.1f}, a1={sol[3]:.1f}, b1={sol[4]:.1f}, "
                      f"f2={sol[5]*1e3:.2f} mm  {is_true}")


# ================================================================
# TEST 6: AB ONLY (no defocus → no ABCD recovery)
# ================================================================

def test6_AB_only_no_defocus():
    """
    What if you ONLY have A and B (no defocus variation)?
    Each setting gives 2 constraints (not 3).
    
    With free focal lengths: never solvable (2K < 3+2K).
    With linear model on 1 lens: need K ≥ 4.
    """
    print("\n" + "=" * 70)
    print("TEST 6: A and B ONLY (no defocus → no C,D recovery)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_base = 3.0e-3
    f2_t = 50.0e-3
    a1_t = 1.0 / f1_base
    b1_t = 5000.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, f2_t])
    
    print(f"\n  With linear model, 6 unknowns:")
    
    for K in [3, 4, 5, 8]:
        w_values = np.linspace(0, 0.01, K)
        
        targets = []
        for w in w_values:
            phi1_w = a1_t + b1_t * w
            f1_w = 1.0 / phi1_w
            A, B = two_lens_AB(d1_t, d2_t, d3_t, f1_w, f2_t)
            targets.extend([A, B])
        targets = np.array(targets)
        
        def make_residual_AB(w_vals, tgt):
            def residual(params):
                d1, d2, d3, a1, b1, f2 = np.abs(params)
                res = []
                for w in w_vals:
                    phi1 = a1 + b1 * w
                    f1 = 1.0 / phi1
                    A, B = two_lens_AB(d1, d2, d3, f1, f2)
                    res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_residual_AB(w_values, targets)
        n_constraints = 2 * K
        
        # Jacobian
        J = np.zeros((n_constraints, 6))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(6):
            p_plus = true_params.copy()
            p_plus[i] += eps
            J[:, i] = (residual(p_plus) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        cond = np.linalg.cond(J) if rank == 6 else float('inf')
        
        status = "overdetermined" if n_constraints > 6 else \
                 "exactly determined" if n_constraints == 6 else "underdetermined"
        
        # Multi-start
        n_trials = 50
        solutions = []
        for trial in range(n_trials):
            np.random.seed(trial + K * 200)
            scale = 0.5 + np.random.rand(6)
            x0 = true_params * scale
            
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-18:
                solutions.append(np.abs(result.x))
        
        unique_solutions = []
        for sol in solutions:
            is_new = True
            for usol in unique_solutions:
                if np.allclose(sol, usol, rtol=0.005):
                    is_new = False
                    break
            if is_new:
                unique_solutions.append(sol)
        
        status_uniq = f"✓ UNIQUE" if len(unique_solutions) == 1 else \
                      f"~ {len(unique_solutions)} solutions" if unique_solutions else "✗ no convergence"
        
        print(f"\n  K={K}: {n_constraints} constraints ({status}), "
              f"rank={rank}/6, distinct={len(unique_solutions)} → {status_uniq}")


# ================================================================
# TEST 7: THE FULL REALISTIC SCENARIO
# ================================================================

def test7_full_scenario():
    """
    Full realistic scenario matching what the user actually has:
    - Known defocus values (different d3)
    - Unknown wobble of one or both lenses
    - Extract A, B from images at each (wobble, defocus) combination
    - Use the defocus structure to recover ABCD
    - Then solve the inverse problem
    
    This test uses A, B measurements directly (not pre-computed ABCD),
    simulating the real pipeline.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Full Realistic Pipeline (A,B from images at known defocus)")
    print("=" * 70)
    
    # True system
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_base = 3.0e-3
    f2_t = 50.0e-3
    a1_t = 1.0 / f1_base
    b1_t = 5000.0
    
    # Wobble settings (known knob positions)
    K = 3
    w_values = np.array([0.0, 0.005, 0.01])
    
    # Defocus positions (known)
    N = 3
    dz_values = np.array([0.0, 0.05, 0.10])  # meters
    
    print(f"\nSetup:")
    print(f"  {K} wobble settings × {N} defocus positions = {K*N} images")
    print(f"  Wobble: w = {w_values}")
    print(f"  Defocus: Δz = {dz_values} m")
    
    # Generate all A, B measurements
    AB_data = {}  # (wobble_idx, defocus_idx) → (A, B)
    for ki, w in enumerate(w_values):
        phi1_w = a1_t + b1_t * w
        f1_w = 1.0 / phi1_w
        for di, dz in enumerate(dz_values):
            A, B = two_lens_AB(d1_t, d2_t, d3_t + dz, f1_w, f2_t)
            AB_data[(ki, di)] = (A, B)
    
    # Step 1: Recover ABCD for each wobble setting using defocus
    print(f"\nStep 1: Recover ABCD from defocus series")
    ABCD_recovered = {}
    for ki in range(K):
        As = [AB_data[(ki, di)][0] for di in range(N)]
        Bs = [AB_data[(ki, di)][1] for di in range(N)]
        
        # Linear fit
        C_rec, A_rec = np.polyfit(dz_values, As, 1)  # slope=C, intercept=A0
        D_rec, B_rec = np.polyfit(dz_values, Bs, 1)  # slope=D, intercept=B0
        
        ABCD_recovered[ki] = (A_rec, B_rec, C_rec, D_rec)
        
        # Compare with true
        phi1_w = a1_t + b1_t * w_values[ki]
        f1_w = 1.0 / phi1_w
        A_true, B_true, C_true, D_true = two_lens_ABCD(d1_t, d2_t, d3_t, f1_w, f2_t)
        
        print(f"  w={w_values[ki]:.3f}: A={A_rec:.4f} (err={abs(A_rec-A_true):.2e}), "
              f"C={C_rec:.4f} (err={abs(C_rec-C_true):.2e}), "
              f"D={D_rec:.6f} (err={abs(D_rec-D_true):.2e})")
    
    # Step 2: Inverse problem using recovered ABCD
    print(f"\nStep 2: Solve inverse problem from {K} ABCD matrices")
    print(f"  Unknowns: d1, d2, d3, a1, b1, f2 = 6")
    print(f"  Constraints: {3*K} (3 per ABCD)")
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, f2_t])
    
    def residual_ABCD(params):
        d1, d2, d3, a1, b1, f2 = np.abs(params)
        res = []
        for ki in range(K):
            phi1 = a1 + b1 * w_values[ki]
            f1 = 1.0 / phi1
            A_pred, B_pred, C_pred, D_pred = two_lens_ABCD(d1, d2, d3, f1, f2)
            A_meas, B_meas, C_meas, D_meas = ABCD_recovered[ki]
            # Use A, C, D (B is determined by det=1)
            res.extend([A_pred - A_meas, C_pred - C_meas, D_pred - D_meas])
        return np.array(res)
    
    # Also test using just A, B (without ABCD recovery)
    def residual_AB_only(params):
        d1, d2, d3, a1, b1, f2 = np.abs(params)
        res = []
        for ki in range(K):
            phi1 = a1 + b1 * w_values[ki]
            f1 = 1.0 / phi1
            for di in range(N):
                A_pred, B_pred = two_lens_AB(d1, d2, d3 + dz_values[di], f1, f2)
                A_meas, B_meas = AB_data[(ki, di)]
                res.extend([A_pred - A_meas, B_pred - B_meas])
        return np.array(res)
    
    # Multi-start for ABCD approach
    n_trials = 100
    solutions_ABCD = []
    solutions_AB = []
    
    for trial in range(n_trials):
        np.random.seed(trial)
        scale = 0.4 + 1.2 * np.random.rand(6)
        x0 = true_params * scale
        
        # ABCD approach
        result = least_squares(residual_ABCD, x0, method='lm', max_nfev=10000,
                              ftol=1e-15, xtol=1e-15)
        if result.cost < 1e-18:
            solutions_ABCD.append(np.abs(result.x))
        
        # AB-only approach (using raw A,B at all defocus/wobble)
        result2 = least_squares(residual_AB_only, x0, method='lm', max_nfev=10000,
                               ftol=1e-15, xtol=1e-15)
        if result2.cost < 1e-18:
            solutions_AB.append(np.abs(result2.x))
    
    def count_unique(solutions, rtol=0.005):
        unique = []
        for sol in solutions:
            is_new = True
            for usol in unique:
                if np.allclose(sol, usol, rtol=rtol):
                    is_new = False
                    break
            if is_new:
                unique.append(sol)
        return unique
    
    unique_ABCD = count_unique(solutions_ABCD)
    unique_AB = count_unique(solutions_AB)
    
    print(f"\n  ABCD approach ({3*K} constraints):")
    print(f"    Converged: {len(solutions_ABCD)}/{n_trials}")
    print(f"    Distinct solutions: {len(unique_ABCD)}")
    for i, sol in enumerate(unique_ABCD[:5]):
        err = np.abs(sol - true_params) / true_params * 100
        mark = "✓ TRUE" if err.max() < 1.0 else f"max_err={err.max():.1f}%"
        print(f"      Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, "
              f"d3={sol[2]*1e3:.1f}, a1={sol[3]:.1f}, b1={sol[4]:.1f}, "
              f"f2={sol[5]*1e3:.2f} mm  {mark}")
    
    print(f"\n  AB-only approach ({2*K*N} constraints):")
    print(f"    Converged: {len(solutions_AB)}/{n_trials}")
    print(f"    Distinct solutions: {len(unique_AB)}")
    for i, sol in enumerate(unique_AB[:5]):
        err = np.abs(sol - true_params) / true_params * 100
        mark = "✓ TRUE" if err.max() < 1.0 else f"max_err={err.max():.1f}%"
        print(f"      Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, "
              f"d3={sol[2]*1e3:.1f}, a1={sol[3]:.1f}, b1={sol[4]:.1f}, "
              f"f2={sol[5]*1e3:.2f} mm  {mark}")
    
    print(f"\n  Note: ABCD and AB approaches should give IDENTICAL results")
    print(f"  (ABCD is just a convenient way to THINK about the information content)")

    return len(unique_ABCD), len(unique_AB)


# ================================================================
# TEST 8: WHAT ABOUT WOBBLING BOTH LENSES?
# ================================================================

def test8_both_lenses_wobble():
    """
    Both lenses have linear models:
    φ₁(w₁) = a₁ + b₁·w₁
    φ₂(w₂) = a₂ + b₂·w₂
    
    Unknowns: d1, d2, d3, a1, b1, a2, b2 = 7
    
    Test with different numbers of settings.
    """
    print("\n" + "=" * 70)
    print("TEST 8: Both lenses wobbled with linear models")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3     # 333.33
    b1_t = 5000.0
    a2_t = 1.0 / 50.0e-3    # 20.0
    b2_t = 200.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])
    
    # Defocus values
    N = 3
    dz_values = np.array([0.0, 0.05, 0.10])
    
    # Test with different wobble configurations
    configs = [
        ("2 settings L1 only", [(0.0, 0.0), (0.01, 0.0)]),
        ("3 settings L1 only", [(0.0, 0.0), (0.005, 0.0), (0.01, 0.0)]),
        ("2×2 grid", [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)]),
        ("3×1 + 1×2", [(0.0, 0.0), (0.005, 0.0), (0.01, 0.0), (0.0, 0.01)]),
        ("3×2 grid", list(product([0.0, 0.005, 0.01], [0.0, 0.01]))),
    ]
    
    for config_name, wobble_pairs in configs:
        K = len(wobble_pairs)
        
        # Compute all AB measurements
        all_AB = []
        for w1, w2 in wobble_pairs:
            phi1 = a1_t + b1_t * w1
            phi2 = a2_t + b2_t * w2
            f1 = 1.0 / phi1
            f2 = 1.0 / phi2
            for dz in dz_values:
                A, B = two_lens_AB(d1_t, d2_t, d3_t + dz, f1, f2)
                all_AB.extend([A, B])
        all_AB = np.array(all_AB)
        
        def make_residual(wp, dzv, tgt):
            def residual(params):
                d1, d2, d3, a1, b1, a2, b2 = np.abs(params)
                res = []
                for w1, w2 in wp:
                    phi1 = a1 + b1 * w1
                    phi2 = a2 + b2 * w2
                    f1 = 1.0 / phi1
                    f2 = 1.0 / phi2
                    for dz in dzv:
                        A, B = two_lens_AB(d1, d2, d3 + dz, f1, f2)
                        res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_residual(wobble_pairs, dz_values, all_AB)
        n_constraints = 2 * K * N
        
        # Jacobian
        J = np.zeros((n_constraints, 7))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(7):
            p_plus = true_params.copy()
            p_plus[i] += eps
            J[:, i] = (residual(p_plus) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        
        # Multi-start
        n_trials = 50
        solutions = []
        for trial in range(n_trials):
            np.random.seed(trial + hash(config_name) % 1000)
            scale = 0.5 + np.random.rand(7)
            x0 = true_params * scale
            
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-15:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            is_new = True
            for usol in unique:
                if np.allclose(sol, usol, rtol=0.005):
                    is_new = False
                    break
            if is_new:
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.01))
        
        status = f"✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                 f"~ {len(unique)} solutions ({n_true} true)" if unique else "✗ failed"
        
        print(f"\n  {config_name:20s}: {n_constraints:2d} constraints, rank={rank}/7, "
              f"converged={len(solutions):2d}/50, distinct={len(unique)}, {status}")
        
        if len(unique) <= 3:
            for i, sol in enumerate(unique):
                err = np.abs(sol - true_params) / true_params * 100
                mark = "✓" if err.max() < 1.0 else f"err={err.max():.1f}%"
                print(f"      Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, "
                      f"d3={sol[2]*1e3:.1f}, a1={sol[3]:.1f}, b1={sol[4]:.1f}, "
                      f"a2={sol[5]:.2f}, b2={sol[6]:.1f}  {mark}")


# ================================================================
# TEST 9: SENSITIVITY — HOW MUCH WOBBLE IS NEEDED?
# ================================================================

def test9_wobble_sensitivity():
    """
    Test: How large must the wobble perturbation be for recovery to work?
    If the wobble is too small, the measurements become numerically degenerate.
    """
    print("\n" + "=" * 70)
    print("TEST 9: Wobble Sensitivity — How much perturbation is needed?")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_base = 3.0e-3
    f2_t = 50.0e-3
    a1_t = 1.0 / f1_base
    
    N = 3
    dz_values = np.array([0.0, 0.05, 0.10])
    K = 3
    
    print(f"\n  Testing with K={K} wobble settings, N={N} defocus each")
    print(f"  f1_base = {f1_base*1e3:.1f} mm")
    print(f"  Varying the total fractional change Δf1/f1 from 0.01% to 30%\n")
    
    for pct_change in [0.01, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0]:
        # b1 such that max wobble gives pct_change% change in f1
        # At max wobble w_max: f1(w_max) ≈ f1_base * (1 + pct_change/100)
        # φ1(w_max) = 1/f1(w_max) = 1/(f1_base(1+p)) ≈ a1(1-p) = a1 - a1*p
        # So b1 * w_max = -a1 * p → b1 = -a1 * p / w_max
        # Use w = [0, 0.005, 0.01]
        delta_phi = a1_t * (pct_change / 100.0)
        b1_t = delta_phi / 0.01  # max wobble is 0.01
        
        w_values = np.array([0.0, 0.005, 0.01])
        true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, f2_t])
        
        # Generate data
        all_AB = []
        for w in w_values:
            phi1 = a1_t + b1_t * w
            f1 = 1.0 / phi1
            for dz in dz_values:
                A, B = two_lens_AB(d1_t, d2_t, d3_t + dz, f1, f2_t)
                all_AB.extend([A, B])
        all_AB = np.array(all_AB)
        
        def make_res(wv, dzv, tgt, tp):
            def residual(params):
                d1, d2, d3, a1, b1, f2 = np.abs(params)
                res = []
                for w in wv:
                    phi1 = a1 + b1 * w
                    f1 = 1.0 / phi1
                    for dz in dzv:
                        A, B = two_lens_AB(d1, d2, d3 + dz, f1, f2)
                        res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_res(w_values, dz_values, all_AB, true_params)
        
        # Jacobian condition
        J = np.zeros((2 * K * N, 6))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(6):
            p_plus = true_params.copy()
            p_plus[i] += eps
            J[:, i] = (residual(p_plus) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        cond = np.linalg.cond(J) if rank == 6 else float('inf')
        
        # Multi-start
        n_trials = 30
        solutions = []
        for trial in range(n_trials):
            np.random.seed(trial)
            scale = 0.5 + np.random.rand(6)
            x0 = true_params * scale
            
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-15:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            is_new = True
            for usol in unique:
                if np.allclose(sol, usol, rtol=0.005):
                    is_new = False
                    break
            if is_new:
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.02))
        status = "✓" if len(unique) == 1 and n_true == 1 else \
                 f"~{len(unique)}sol" if unique else "✗"
        
        print(f"  Δf1/f1 = {pct_change:5.1f}%: rank={rank}/6, cond={cond:10.2e}, "
              f"converged={len(solutions):2d}/30, distinct={len(unique)}, {status}")


# ================================================================
# MAIN: RUN ALL TESTS
# ================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  DEFINITIVE SOLVABILITY ANALYSIS: Two-Lens Inverse Problem       ║")
    print("╚" + "═" * 68 + "╝")
    
    test0_defocus_gives_ABCD()
    test1_equation_counting()
    test2_single_ABCD()
    test3_two_ABCD_one_lens()
    test4_three_ABCD_one_lens()
    test5_linear_model_one_lens()
    test6_AB_only_no_defocus()
    test7_full_scenario()
    test8_both_lenses_wobble()
    test9_wobble_sensitivity()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║                    FINAL CONCLUSIONS                             ║")
    print("╚" + "═" * 68 + "╝")
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL INSIGHT: DEFOCUS GIVES YOU THE FULL ABCD MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When you measure A and B at N ≥ 2 known defocus positions (changes to d3),
you recover ALL FOUR ABCD elements:

    A(Δz) = A₀ + C · Δz     →  fit intercept = A₀, slope = C
    B(Δz) = B₀ + D · Δz     →  fit intercept = B₀, slope = D

This is EXACT (not approximate). It works because:
    M(d3 + Δz) = P(Δz) · M(d3)

Each wobble setting then gives 3 independent constraints (not 2!),
because det(ABCD) = 1 removes one DOF.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MINIMUM REQUIREMENTS (from equation counting + numerical verification):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For d1, d2, d3 unknown + wobbling ONE lens with independent f values:
  ✓ 2 wobble settings × ≥2 defocus each = 4+ images minimum
    (6 unknowns, 6 constraints → exactly determined)
  ✓ 3+ wobble settings → overdetermined (more robust)

For d1, d2, d3 unknown + linear wobble model φ₁(w) = a₁ + b₁·w:
  ✓ 3 wobble settings × ≥2 defocus each = 6+ images minimum
    (6 unknowns, 9 constraints → overdetermined)

For both lenses wobbled (linear models for both):
  ✓ Need wobble in BOTH lenses (not just one)
  ✓ 3+ settings with variation in both w₁ and w₂
  ✓ ≥2 defocus per setting
  (7 unknowns, ≥12 constraints)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITHOUT DEFOCUS (A,B only, no C,D recovery):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗ With free focal lengths per setting: NEVER solvable (always 3 short)
  ~ With linear model: need K ≥ 4 settings (marginal)
  → DEFOCUS IS ESSENTIAL for robust recovery

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRACTICAL RECIPE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Choose 3-5 wobble settings for one (or both) lenses
2. At EACH wobble setting, measure images at ≥3 known defocus positions
3. For each wobble setting:
   - Linear-fit A vs Δz → get A₀ and C
   - Linear-fit B vs Δz → get B₀ and D
4. Now you have K ABCD matrices (3K constraints for 6-7 unknowns)
5. Solve the nonlinear inverse problem using least-squares (or BFGS)

Total images: K × N = 9-15 (much less than 27!)
Constraint ratio: 9-15 equations for 6-7 unknowns (well overdetermined)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE LINEAR MODEL HELPS BUT IS NOT STRICTLY NECESSARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

With independent focal lengths (no linear model):
  - Each new wobble setting adds 2 unknowns but 3 constraints (+1 net)
  - So K ≥ 3 settings suffice (with defocus)
  
With linear model φ(w) = a + b·w:
  - Constrains all settings through 2 parameters instead of K
  - Reduces unknowns from 3+2K to 6 (one lens) or 7 (both lenses)
  - Gives better conditioning and overdetermination
  - But requires linearity assumption to be valid!

RECOMMENDATION: Use the linear model if wobble is small (≤10-20% Δf/f).
For larger wobbles, treat focal lengths as independent or use quadratic model.
""")


if __name__ == "__main__":
    main()

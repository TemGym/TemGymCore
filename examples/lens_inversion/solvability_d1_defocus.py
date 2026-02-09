"""
Corrected Solvability Analysis: Defocus at d1 (sample plane), NOT d3 (detector)
================================================================================

CRITICAL CORRECTION: The original analysis assumed defocus changes d3 (detector
distance). In reality, the wobble/defocus changes d1 (source-to-first-lens 
distance). This fundamentally changes what information is available.

MATHEMATICS:
  M(d1 + Δz) = P(d3) @ L2 @ P(d2) @ L1 @ [P(d1) @ P(Δz)]
             = M_base @ P(Δz)
             = [[A, AΔz+B], [C, CΔz+D]]

Therefore:
  A'(Δz) = A         ← CONSTANT (unchanged by d1 defocus!)
  B'(Δz) = B + A·Δz  ← linear, but slope is just A (already known)
  C'(Δz) = C         ← CONSTANT  
  D'(Δz) = D + C·Δz  ← linear, but we can't measure D from intensity

CONSEQUENCE: d1-defocus gives us only (A, B₀) per wobble setting = 2 constraints
             NOT 3 as with d3-defocus.

This script tests whether the problem is still solvable.

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
    return np.array([[1.0, z], [0.0, 1.0]])

def lens_matrix(f):
    return np.array([[1.0, 0.0], [-1.0/f, 1.0]])

def two_lens_ABCD(d1, d2, d3, f1, f2):
    M = propagation_matrix(d3) @ lens_matrix(f2) @ propagation_matrix(d2) @ \
        lens_matrix(f1) @ propagation_matrix(d1)
    return M[0, 0], M[0, 1], M[1, 0], M[1, 1]

def two_lens_AB(d1, d2, d3, f1, f2):
    A, B, C, D = two_lens_ABCD(d1, d2, d3, f1, f2)
    return A, B


# ================================================================
# TEST 0: PROVE d1-defocus vs d3-defocus difference
# ================================================================

def test0_d1_vs_d3_defocus():
    """Show exactly what d1 vs d3 defocus gives you."""
    print("=" * 70)
    print("TEST 0: d1-defocus vs d3-defocus — what changes?")
    print("=" * 70)
    
    d1, d2, d3 = 3.06e-3, 205.5e-3, 1.05
    f1, f2 = 3.0e-3, 50.0e-3
    
    A0, B0, C0, D0 = two_lens_ABCD(d1, d2, d3, f1, f2)
    print(f"\nBase ABCD: A={A0:.4f}, B={B0:.4e}, C={C0:.4f}, D={D0:.6f}")
    print(f"det = {A0*D0 - B0*C0:.10f}")
    
    dz_values = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    
    # d3 defocus: M_new = P(Δz) @ M_base
    print(f"\n--- d3 defocus (detector moves) ---")
    print(f"  M(d3+Δz) = P(Δz) · M_base")
    print(f"  A' = A + CΔz (changes!), B' = B + DΔz (changes!)")
    print(f"  C' = C (fixed), D' = D (fixed)")
    for dz in dz_values:
        A, B, C, D = two_lens_ABCD(d1, d2, d3 + dz, f1, f2)
        print(f"  Δz={dz:+.3f}m: A={A:.4f}, B={B:.6e}, C={C:.4f}, D={D:.6f}")
    
    # d1 defocus: M_new = M_base @ P(Δz)
    print(f"\n--- d1 defocus (sample/source plane moves) ---")
    print(f"  M(d1+Δz) = M_base · P(Δz)")
    print(f"  A' = A (FIXED!), B' = B + AΔz (changes), C' = C (fixed), D' = D + CΔz (changes)")
    for dz in dz_values:
        A, B, C, D = two_lens_ABCD(d1 + dz, d2, d3, f1, f2)
        print(f"  Δz={dz:+.3f}m: A={A:.4f}, B={B:.6e}, C={C:.4f}, D={D:.6f}")
    
    # Verify linearity for d1 defocus
    As_d1 = [two_lens_ABCD(d1 + dz, d2, d3, f1, f2)[0] for dz in dz_values]
    Bs_d1 = [two_lens_ABCD(d1 + dz, d2, d3, f1, f2)[1] for dz in dz_values]
    
    A_variation = np.max(np.abs(np.array(As_d1) - A0))
    B_slope, B_intercept = np.polyfit(dz_values, Bs_d1, 1)
    
    print(f"\n  Verification:")
    print(f"  A variation across defocus: {A_variation:.2e} (should be ~0)")
    print(f"  B slope = {B_slope:.4f} (should be A = {A0:.4f})")
    print(f"  B intercept = {B_intercept:.4e} (should be B₀ = {B0:.4e})")
    
    print(f"\n  ╔═══════════════════════════════════════════════════════════════╗")
    print(f"  ║  CONCLUSION: d1-defocus gives A (constant) and B₀ only.     ║")
    print(f"  ║  The slope of B vs Δz is just A (already known).            ║")
    print(f"  ║  C and D are NOT recoverable from d1-defocus.               ║")
    print(f"  ║                                                             ║")
    print(f"  ║  Each wobble setting → 2 independent constraints (A, B₀)    ║")
    print(f"  ║  NOT 3 as claimed in the d3-defocus analysis!               ║")
    print(f"  ╚═══════════════════════════════════════════════════════════════╝")


# ================================================================
# TEST 1: REVISED EQUATION COUNTING
# ================================================================

def test1_equation_counting():
    print("\n" + "=" * 70)
    print("TEST 1: Revised Equation Counting (d1-defocus)")
    print("=" * 70)
    
    print("""
┌──────────────────────────────────────────────────────────────────────┐
│            REVISED EQUATION COUNTING (d1 defocus)                  │
│                                                                    │
│ d1-defocus: A is constant, B varies with known slope A.            │
│ → Each wobble setting gives 2 constraints (A₀, B₀), NOT 3.        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                    │
│ Scenario                         │ Unknowns │ Constr. │ Status     │
│──────────────────────────────────┼──────────┼─────────┼────────────│
│ K settings, free f₁,f₂ each     │  3 + 2K  │   2K    │ ✗ NEVER!   │
│  (always 3 unknowns short)       │          │         │            │
│──────────────────────────────────┼──────────┼─────────┼────────────│
│ K settings, wobble 1 lens only   │  3+K+1   │   2K    │            │
│  (d1,d2,d3 + K f1s + f2)         │          │         │            │
│   K=2 → 6 unkn, 4 constr         │    6     │    4    │ ✗          │
│   K=3 → 7 unkn, 6 constr         │    7     │    6    │ ✗          │
│   K=4 → 8 unkn, 8 constr         │    8     │    8    │ ~ exact    │
│   K=5 → 9 unkn, 10 constr        │    9     │   10    │ ~ +1       │
│──────────────────────────────────┼──────────┼─────────┼────────────│
│ Linear model, wobble 1 lens      │    6     │   2K    │            │
│  (d1,d2,d3,a₁,b₁,f₂)            │          │         │            │
│   K=3 → 6 unkn, 6 constr         │    6     │    6    │ ~ exact    │
│   K=4 → 6 unkn, 8 constr         │    6     │    8    │ ✓ overdet  │
│   K=5 → 6 unkn, 10 constr        │    6     │   10    │ ✓ well     │
│──────────────────────────────────┼──────────┼─────────┼────────────│
│ Linear model, wobble BOTH lenses │    7     │   2K    │            │
│  (d1,d2,d3,a₁,b₁,a₂,b₂)        │          │         │            │
│   K=3 → 7 unkn, 6 constr         │    7     │    6    │ ✗ under    │
│   K=4 → 7 unkn, 8 constr         │    7     │    8    │ ~ +1       │
│   K=5 → 7 unkn, 10 constr        │    7     │   10    │ ✓ overdet  │
│──────────────────────────────────┼──────────┼─────────┼────────────│
│                                                                    │
│ NOTE: With d3-defocus we got 3 constraints per setting (ABCD).     │
│       With d1-defocus we only get 2 (A, B₀).                      │
│       This means we need MORE wobble settings to compensate!       │
└──────────────────────────────────────────────────────────────────────┘
""")


# ================================================================
# TEST 2: SINGLE LENS WOBBLE WITH AB ONLY
# ================================================================

def test2_one_lens_wobble_AB():
    """
    Wobble lens 1 only, linear model φ₁(w) = a₁ + b₁·w.
    Unknowns: d1, d2, d3, a1, b1, f2 = 6
    Each setting gives (A, B) = 2 constraints.
    """
    print("\n" + "=" * 70)
    print("TEST 2: One-lens wobble, AB only (d1-defocus)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1_base = 3.0e-3
    f2_t = 50.0e-3
    a1_t = 1.0 / f1_base
    b1_t = 5000.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, f2_t])
    
    for K in [3, 4, 5, 8, 12]:
        w_values = np.linspace(0, 0.01, K)
        
        # Generate (A, B₀) at each wobble setting (B at base d1, no defocus)
        targets = []
        for w in w_values:
            phi1 = a1_t + b1_t * w
            f1 = 1.0 / phi1
            A, B = two_lens_AB(d1_t, d2_t, d3_t, f1, f2_t)
            targets.extend([A, B])
        targets = np.array(targets)
        
        def make_residual(wv, tgt):
            def residual(params):
                d1, d2, d3, a1, b1, f2 = np.abs(params)
                res = []
                for w in wv:
                    phi1 = a1 + b1 * w
                    f1 = 1.0 / phi1
                    A, B = two_lens_AB(d1, d2, d3, f1, f2)
                    res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_residual(w_values, targets)
        n_constraints = 2 * K
        
        # Jacobian
        J = np.zeros((n_constraints, 6))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(6):
            p = true_params.copy()
            p[i] += eps
            J[:, i] = (residual(p) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        sv = np.linalg.svd(J, compute_uv=False)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-15 else float('inf')
        
        # Multi-start
        n_trials = 50
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
            if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.01))
        status = "✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                 f"✗ {len(unique)} solutions ({n_true} true)"
        
        print(f"  K={K:2d}: {n_constraints:2d} constr, rank={rank}/6, "
              f"sv_min={sv[-1]:.2e}, conv={len(solutions):2d}/50, "
              f"distinct={len(unique)}, {status}")
        
        if len(unique) <= 4:
            for i, sol in enumerate(unique[:4]):
                err = np.abs(sol - true_params) / true_params * 100
                mark = "✓" if err.max() < 1.0 else f"err={err.max():.1f}%"
                print(f"      Sol {i+1}: d2={sol[1]*1e3:.2f}mm, d3={sol[2]*1e3:.1f}mm, "
                      f"f2={sol[5]*1e3:.2f}mm  {mark}")


# ================================================================
# TEST 3: BOTH LENSES WOBBLE WITH AB ONLY
# ================================================================

def test3_both_lenses_wobble_AB():
    """
    Both lenses wobbled with linear models, AB only.
    Unknowns: d1, d2, d3, a1, b1, a2, b2 = 7.
    Each setting gives (A, B) = 2 constraints.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Both-lens wobble, AB only (d1-defocus)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3
    b1_t = 5000.0
    a2_t = 1.0 / 50.0e-3
    b2_t = 200.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])
    
    configs = [
        ("3 settings: (0,0),(w,0),(0,w)",       [(0, 0), (0.01, 0), (0, 0.01)]),
        ("4 settings: 3+diag",                  [(0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01)]),
        ("5 settings: 3+extras",                [(0, 0), (0.005, 0), (0.01, 0), (0, 0.005), (0, 0.01)]),
        ("6 settings: 3×2 grid",                list(product([0.0, 0.005, 0.01], [0.0, 0.01]))),
        ("9 settings: 3×3 grid",                list(product([0.0, 0.005, 0.01], [0.0, 0.005, 0.01]))),
    ]
    
    for config_name, wobble_pairs in configs:
        K = len(wobble_pairs)
        
        # Generate A, B at each setting (no defocus needed; just base d1)
        targets = []
        for w1, w2 in wobble_pairs:
            phi1 = a1_t + b1_t * w1
            phi2 = a2_t + b2_t * w2
            f1 = 1.0 / phi1
            f2 = 1.0 / phi2
            A, B = two_lens_AB(d1_t, d2_t, d3_t, f1, f2)
            targets.extend([A, B])
        targets = np.array(targets)
        
        def make_residual(wp, tgt):
            def residual(params):
                d1, d2, d3, a1, b1, a2, b2 = np.abs(params)
                res = []
                for w1, w2 in wp:
                    phi1 = a1 + b1 * w1
                    phi2 = a2 + b2 * w2
                    f1 = 1.0 / phi1
                    f2 = 1.0 / phi2
                    A, B = two_lens_AB(d1, d2, d3, f1, f2)
                    res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_residual(wobble_pairs, targets)
        n_constraints = 2 * K
        
        # Jacobian
        J = np.zeros((n_constraints, 7))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(7):
            p = true_params.copy()
            p[i] += eps
            J[:, i] = (residual(p) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        sv = np.linalg.svd(J, compute_uv=False)
        
        # Multi-start
        n_trials = 80
        solutions = []
        use_lm = n_constraints >= 7
        for trial in range(n_trials):
            np.random.seed(trial)
            scale = 0.5 + np.random.rand(7)
            x0 = true_params * scale
            if use_lm:
                result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                      ftol=1e-15, xtol=1e-15)
            else:
                result = least_squares(residual, x0, method='trf', max_nfev=10000,
                                      ftol=1e-15, xtol=1e-15,
                                      bounds=(1e-10, np.inf))
            if result.cost < 1e-14:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.01))
        
        status = "✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                 f"~ {len(unique)} solutions ({n_true} true)"
        
        print(f"\n  {config_name}")
        print(f"    {n_constraints} constr, 7 unkn, rank={rank}/7, "
              f"sv_min={sv[-1]:.2e}, conv={len(solutions)}/80, "
              f"distinct={len(unique)} → {status}")
        
        if len(unique) <= 5:
            for i, sol in enumerate(unique[:5]):
                err = np.abs(sol - true_params) / true_params * 100
                mark = "✓ TRUE" if err.max() < 1.0 else f"max_err={err.max():.1f}%"
                print(f"      Sol {i+1}: d1={sol[0]*1e3:.4f}, d2={sol[1]*1e3:.2f}, "
                      f"d3={sol[2]*1e3:.1f}, a1={sol[3]:.1f}, b1={sol[4]:.1f}, "
                      f"a2={sol[5]:.2f}, b2={sol[6]:.1f}  {mark}")


# ================================================================
# TEST 4: d1 DEFOCUS AT EACH WOBBLE SETTING
# ================================================================

def test4_d1_defocus_with_wobble():
    """
    Using d1-defocus: multiple Δz at each wobble setting.
    Even though more images are taken, A is constant per setting,
    so redundant images only improve B₀ estimate (noise averaging).
    
    Test: does adding d1-defocus images help beyond AB?
    Answer should be NO — same (A, B₀) information per setting.
    """
    print("\n" + "=" * 70)
    print("TEST 4: d1-defocus at each wobble setting (does it help?)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3
    b1_t = 5000.0
    a2_t = 1.0 / 50.0e-3
    b2_t = 200.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])
    
    # 4 wobble settings with both lenses
    wobble_pairs = [(0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01)]
    K = len(wobble_pairs)
    
    # d1-defocus values
    dz_values_list = [
        ("N=1 (no defocus)", [0.0]),
        ("N=3 defocus",      [0.0, 0.5e-3, 1.0e-3]),
        ("N=5 defocus",      [0.0, 0.25e-3, 0.5e-3, 0.75e-3, 1.0e-3]),
    ]
    
    for dz_name, dz_values in dz_values_list:
        N = len(dz_values)
        
        # Generate all A, B measurements
        targets = []
        for w1, w2 in wobble_pairs:
            phi1 = a1_t + b1_t * w1
            phi2 = a2_t + b2_t * w2
            f1 = 1.0 / phi1
            f2 = 1.0 / phi2
            for dz in dz_values:
                A, B = two_lens_AB(d1_t + dz, d2_t, d3_t, f1, f2)
                targets.extend([A, B])
        targets = np.array(targets)
        
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
                        A, B = two_lens_AB(d1 + dz, d2, d3, f1, f2)
                        res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_residual(wobble_pairs, dz_values, targets)
        n_constraints = 2 * K * N
        
        # Jacobian
        J = np.zeros((n_constraints, 7))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(7):
            p = true_params.copy()
            p[i] += eps
            J[:, i] = (residual(p) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        sv = np.linalg.svd(J, compute_uv=False)
        
        # Multi-start
        n_trials = 50
        solutions = []
        for trial in range(n_trials):
            np.random.seed(trial)
            scale = 0.5 + np.random.rand(7)
            x0 = true_params * scale
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-15:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.01))
        status = "✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                 f"~ {len(unique)} solutions"
        
        print(f"\n  {dz_name}: {K}×{N} = {K*N} images, {n_constraints} constr, "
              f"rank={rank}/7, distinct={len(unique)} → {status}")


# ================================================================
# TEST 5: DO WE ACTUALLY NEED DEFOCUS AT ALL?
# ================================================================

def test5_no_defocus_needed():
    """
    If d1-defocus doesn't add independent information (just noise averaging),
    then we should get the same result with NO defocus at all — just more
    wobble settings at the base d1.
    
    This test compares:
    - K settings, no defocus (1 image per setting)
    - K settings, with d1 defocus (N images per setting)
    Both should find the same number of solutions.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Is d1-defocus actually useful for uniqueness?")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3
    b1_t = 5000.0
    a2_t = 1.0 / 50.0e-3
    b2_t = 200.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])
    
    # Use 4 wobble settings (both lenses)
    wobble_pairs = [(0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01)]
    K = len(wobble_pairs)
    
    # Case A: No defocus (1 image each)
    targets_nodf = []
    for w1, w2 in wobble_pairs:
        phi1 = a1_t + b1_t * w1
        phi2 = a2_t + b2_t * w2
        A, B = two_lens_AB(d1_t, d2_t, d3_t, 1.0/phi1, 1.0/phi2)
        targets_nodf.extend([A, B])
    targets_nodf = np.array(targets_nodf)
    
    def residual_nodf(params):
        d1, d2, d3, a1, b1, a2, b2 = np.abs(params)
        res = []
        for w1, w2 in wobble_pairs:
            phi1 = a1 + b1 * w1
            phi2 = a2 + b2 * w2
            A, B = two_lens_AB(d1, d2, d3, 1.0/phi1, 1.0/phi2)
            res.extend([A, B])
        return np.array(res) - targets_nodf
    
    # Case B: With d1-defocus (3 images each)
    dz_values = [0.0, 0.5e-3, 1.0e-3]
    targets_df = []
    for w1, w2 in wobble_pairs:
        phi1 = a1_t + b1_t * w1
        phi2 = a2_t + b2_t * w2
        for dz in dz_values:
            A, B = two_lens_AB(d1_t + dz, d2_t, d3_t, 1.0/phi1, 1.0/phi2)
            targets_df.extend([A, B])
    targets_df = np.array(targets_df)
    
    def residual_df(params):
        d1, d2, d3, a1, b1, a2, b2 = np.abs(params)
        res = []
        for w1, w2 in wobble_pairs:
            phi1 = a1 + b1 * w1
            phi2 = a2 + b2 * w2
            for dz in dz_values:
                A, B = two_lens_AB(d1 + dz, d2, d3, 1.0/phi1, 1.0/phi2)
                res.extend([A, B])
        return np.array(res) - targets_df
    
    for label, residual_fn, nc in [("No defocus (4 images)", residual_nodf, 8),
                                     ("d1 defocus (12 images)", residual_df, 24)]:
        # Jacobian
        J = np.zeros((nc, 7))
        eps = 1e-10
        f0 = residual_fn(true_params)
        for i in range(7):
            p = true_params.copy()
            p[i] += eps
            J[:, i] = (residual_fn(p) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        
        # Multi-start
        solutions = []
        for trial in range(80):
            np.random.seed(trial)
            x0 = true_params * (0.5 + np.random.rand(7))
            result = least_squares(residual_fn, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-15:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.01))
        status = "✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                 f"~ {len(unique)} solutions ({n_true} true)"
        
        print(f"\n  {label}: {nc} constr, rank={rank}/7, "
              f"conv={len(solutions)}/80, distinct={len(unique)} → {status}")
    
    print(f"\n  (If both give same # solutions, d1-defocus adds NO information.)")
    print(f"  (If d1-defocus has fewer solutions, it DOES help beyond AB.)")


# ================================================================
# TEST 6: SCALING AMBIGUITY CHECK
# ================================================================

def test6_scaling_check():
    """
    Check if there's a scaling symmetry: if we scale (d1,d2,d3,1/a1,1/a2) 
    by factor α, do we get the same A,B?
    """
    print("\n" + "=" * 70)
    print("TEST 6: Scaling symmetry check")
    print("=" * 70)
    
    d1, d2, d3 = 3.06e-3, 205.5e-3, 1.05
    f1, f2 = 3.0e-3, 50.0e-3
    
    A0, B0 = two_lens_AB(d1, d2, d3, f1, f2)
    
    for alpha in [0.5, 2.0, 10.0]:
        A_s, B_s = two_lens_AB(alpha*d1, alpha*d2, alpha*d3, alpha*f1, alpha*f2)
        print(f"  α={alpha:4.1f}: A={A_s:.4f} (orig={A0:.4f}, same={np.isclose(A_s,A0)}), "
              f"B={B_s:.4e} (orig={B0:.4e}, ratio={B_s/B0 if B0 != 0 else 'N/A'})")
    
    # Check: A is dimensionless, B has units of length
    # A = 1 - d2/f1 - d3/f2 + d2*d3/(f1*f2) + ... → scale-invariant for distances/focal lengths
    # B has units of length → scales with α
    
    # So A is scale-invariant but B is not! B breaks the scaling.
    print(f"\n  A is scale-invariant (dimensionless)")
    print(f"  B scales with α (has units of length)")
    print(f"  → B breaks the scaling symmetry")
    print(f"  → NO scaling ambiguity in (A, B) measurements")


# ================================================================
# TEST 7: WHAT SPECIFIC DEGENERACY EXISTS?
# ================================================================

def test7_analyze_degeneracy():
    """
    For the single-lens wobble case where many solutions exist,
    analyze the relationship between the spurious solutions to
    understand the degeneracy structure.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Degeneracy structure (what are the spurious solutions?)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3
    b1_t = 5000.0
    f2_t = 50.0e-3
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, f2_t])
    
    # 5 wobble settings, 1 lens
    K = 5
    w_values = np.linspace(0, 0.01, K)
    
    targets = []
    for w in w_values:
        phi1 = a1_t + b1_t * w
        A, B = two_lens_AB(d1_t, d2_t, d3_t, 1.0/phi1, f2_t)
        targets.extend([A, B])
    targets = np.array(targets)
    
    def residual(params):
        d1, d2, d3, a1, b1, f2 = np.abs(params)
        res = []
        for w in w_values:
            phi1 = a1 + b1 * w
            A, B = two_lens_AB(d1, d2, d3, 1.0/phi1, f2)
            res.extend([A, B])
        return np.array(res) - targets
    
    # Collect many solutions
    solutions = []
    for trial in range(200):
        np.random.seed(trial)
        scale = 0.3 + 1.4 * np.random.rand(6)
        x0 = true_params * scale
        result = least_squares(residual, x0, method='lm', max_nfev=10000,
                              ftol=1e-15, xtol=1e-15)
        if result.cost < 1e-15:
            solutions.append(np.abs(result.x))
    
    if len(solutions) < 2:
        print("  Not enough solutions found for analysis.")
        return
    
    solutions = np.array(solutions)
    
    print(f"  Found {len(solutions)} solutions fitting the data perfectly.\n")
    
    # Check if d1 is well-determined
    print(f"  Parameter ranges across solutions:")
    names = ['d1', 'd2', 'd3', 'a1', 'b1', 'f2']
    for i, name in enumerate(names):
        vals = solutions[:, i]
        print(f"    {name:3s}: min={vals.min():.4e}, max={vals.max():.4e}, "
              f"std/mean={vals.std()/vals.mean()*100:.1f}%")
    
    # Check correlations
    print(f"\n  Checking d2 vs f2 correlation (the suspected degeneracy):")
    d2_vals = solutions[:, 1]
    f2_vals = solutions[:, 5]
    
    # Are d2 and f2 proportional?
    ratio = d2_vals / f2_vals
    print(f"    d2/f2 ratio: mean={ratio.mean():.4f}, std={ratio.std():.4f}, "
          f"CV={ratio.std()/ratio.mean()*100:.1f}%")
    
    # Check d2+d3 (maybe total path is fixed?)
    d3_vals = solutions[:, 2]
    total = d2_vals + d3_vals
    print(f"    d2+d3: mean={total.mean():.4f}, std={total.std():.4f}, "
          f"CV={total.std()/total.mean()*100:.1f}%")
    
    # Check if there's a functional relationship
    print(f"\n  Sample solutions:")
    # Sort by d2
    idx = np.argsort(solutions[:, 1])
    for i in idx[::max(1, len(idx)//8)][:8]:
        sol = solutions[i]
        err = np.abs(sol - true_params) / true_params * 100
        print(f"    d1={sol[0]*1e3:.4f}mm, d2={sol[1]*1e3:.2f}mm, d3={sol[2]*1e3:.1f}mm, "
              f"a1={sol[3]:.1f}, b1={sol[4]:.1f}, f2={sol[5]*1e3:.2f}mm")


# ================================================================
# TEST 8: WHAT IF WE KNOW d3? (OR d1, OR d2?)
# ================================================================

def test8_fix_one_distance():
    """
    Can we solve the problem if ONE of the distances is known?
    """
    print("\n" + "=" * 70)
    print("TEST 8: Fix one distance — does it become solvable?")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3
    b1_t = 5000.0
    a2_t = 1.0 / 50.0e-3
    b2_t = 200.0
    
    K = 4
    
    # Test single-lens wobble with one distance fixed
    configs = [
        ("Wobble L1, fix d3, linear model",
         np.array([d1_t, d2_t, a1_t, b1_t, 50.0e-3]),  # d1,d2,a1,b1,f2
         [(0,), (0.003,), (0.006,), (0.01,)]),
        
        ("Wobble both, fix d3, linear models",
         np.array([d1_t, d2_t, a1_t, b1_t, a2_t, b2_t]),  # d1,d2,a1,b1,a2,b2
         [(0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01)]),
    ]
    
    for config_name, true_p, wobble_list in configs:
        n_unknowns = len(true_p)
        n_constraints = 2 * len(wobble_list)
        
        targets = []
        if len(wobble_list[0]) == 1:
            # Single lens wobble
            for (w,) in wobble_list:
                phi1 = true_p[2] + true_p[3] * w  # a1 + b1*w
                f1 = 1.0 / phi1
                A, B = two_lens_AB(true_p[0], true_p[1], d3_t, f1, true_p[4])
                targets.extend([A, B])
            targets = np.array(targets)
            
            def make_res_1lens(wl, d3_fixed, tgt):
                def residual(params):
                    d1, d2, a1, b1, f2 = np.abs(params)
                    res = []
                    for (w,) in wl:
                        phi1 = a1 + b1 * w
                        A, B = two_lens_AB(d1, d2, d3_fixed, 1.0/phi1, f2)
                        res.extend([A, B])
                    return np.array(res) - tgt
                return residual
            residual = make_res_1lens(wobble_list, d3_t, targets)
        else:
            # Both lenses wobble
            for w1, w2 in wobble_list:
                phi1 = true_p[2] + true_p[3] * w1
                phi2 = true_p[4] + true_p[5] * w2
                A, B = two_lens_AB(true_p[0], true_p[1], d3_t, 1.0/phi1, 1.0/phi2)
                targets.extend([A, B])
            targets = np.array(targets)
            
            def make_res_2lens(wl, d3_fixed, tgt):
                def residual(params):
                    d1, d2, a1, b1, a2, b2 = np.abs(params)
                    res = []
                    for w1, w2 in wl:
                        phi1 = a1 + b1 * w1
                        phi2 = a2 + b2 * w2
                        A, B = two_lens_AB(d1, d2, d3_fixed, 1.0/phi1, 1.0/phi2)
                        res.extend([A, B])
                    return np.array(res) - tgt
                return residual
            residual = make_res_2lens(wobble_list, d3_t, targets)
        
        # Jacobian
        J = np.zeros((n_constraints, n_unknowns))
        eps = 1e-10
        f0 = residual(true_p)
        for i in range(n_unknowns):
            p = true_p.copy()
            p[i] += eps
            J[:, i] = (residual(p) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        
        # Multi-start
        solutions = []
        for trial in range(80):
            np.random.seed(trial)
            x0 = true_p * (0.5 + np.random.rand(n_unknowns))
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-15:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_p, rtol=0.01))
        status = "✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                 f"~ {len(unique)} solutions ({n_true} true)"
        
        print(f"\n  {config_name}")
        print(f"    {n_constraints} constr, {n_unknowns} unkn, rank={rank}/{n_unknowns}, "
              f"distinct={len(unique)} → {status}")
        
        if len(unique) <= 5:
            for i, sol in enumerate(unique[:5]):
                err = np.abs(sol - true_p) / true_p * 100
                mark = "✓" if err.max() < 1.0 else f"max_err={err.max():.1f}%"
                print(f"      Sol {i+1}: {sol}  {mark}")


# ================================================================
# TEST 9: SYMBOLIC ANALYSIS OF WHAT A AND B DEPEND ON
# ================================================================

def test9_symbolic_structure():
    """
    Derive the explicit dependence of A and B on parameters.
    This helps understand WHY certain degeneracies exist.
    """
    print("\n" + "=" * 70)
    print("TEST 9: Symbolic structure of A and B")
    print("=" * 70)
    
    # ABCD matrix = P(d3) @ L(f2) @ P(d2) @ L(f1) @ P(d1)
    # Let φ₁ = 1/f₁, φ₂ = 1/f₂
    
    # Step by step:
    # P(d1) = [[1, d1], [0, 1]]
    # L(f1) @ P(d1) = [[1-d1φ₁, d1], [-φ₁, 1]]  
    #   Wait, more carefully:
    # L(f1) = [[1, 0], [-φ₁, 1]]
    # L(f1) @ P(d1) = [[1, d1], [-φ₁, 1-d1φ₁]]
    # P(d2) @ L(f1) @ P(d1) = [[1-d2φ₁, d1+d2-d1d2φ₁], [-φ₁, 1-d1φ₁]]
    # ... this gets complex. Let me just compute numerically.
    
    print("""
  The full ABCD matrix for P(d3)·L(φ₂)·P(d2)·L(φ₁)·P(d1):
  
  A = 1 - d₂φ₁ - d₃φ₂ - d₃d₂φ₁φ₂ + d₃φ₁φ₂d₂  ... (complex)
  
  But the KEY structural fact is:

  A depends on: d₂, d₃, φ₁, φ₂           (NOT on d₁!)
  B depends on: d₁, d₂, d₃, φ₁, φ₂       (ALL parameters)

  This is because A is the position-to-position transfer element,
  and d₁ only multiplies the input ANGLE (which A doesn't couple to).

  Formally: M = M_rest · P(d1), so A_total = A_rest · 1 + B_rest · 0 = A_rest
  B_total = A_rest · d1 + B_rest

  This means:
  ┌─────────────────────────────────────────────────────────────────┐
  │  A does NOT depend on d₁                                      │
  │  B = A · d₁ + B_rest(d₂, d₃, φ₁, φ₂)                        │
  │                                                               │
  │  Since A doesn't depend on d₁, and B is linear in d₁          │
  │  with a known coefficient (A), fitting B effectively gives     │
  │  B_rest = B - A·d₁ which depends on d₂, d₃, φ₁, φ₂           │
  │                                                               │
  │  So each wobble setting gives:                                │
  │    A(d₂, d₃, φ₁, φ₂) = measured A                           │
  │    B_rest(d₂, d₃, φ₁, φ₂) = measured B - A·d₁               │
  │                                                               │
  │  BUT d₁ is unknown! So B and d₁ are entangled:               │
  │  we measure B = A·d₁ + B_rest, but we don't know d₁.         │
  │                                                               │
  │  HOWEVER: d₁ is the SAME for all measurements, while A and   │
  │  B_rest change with wobble. So d₁ is identifiable from the    │
  │  system of equations as long as we have enough settings.      │
  └─────────────────────────────────────────────────────────────────┘
""")
    
    # Verify A doesn't depend on d1
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    f1, f2 = 3.0e-3, 50.0e-3
    
    print("  Numerical verification: A vs d₁")
    for d1_test in [1e-3, 3e-3, 5e-3, 10e-3, 50e-3]:
        A, B = two_lens_AB(d1_test, d2_t, d3_t, f1, f2)
        print(f"    d₁={d1_test*1e3:.1f}mm: A={A:.6f}")
    
    A_ref, _ = two_lens_AB(d1_t, d2_t, d3_t, f1, f2)
    print(f"\n  → A is CONSTANT regardless of d₁ ✓")
    print(f"  → A = {A_ref:.4f} for all d₁ values")
    
    # Show B = A·d₁ + B_rest
    print(f"\n  B decomposition: B = A·d₁ + B_rest")
    for d1_test in [1e-3, 3e-3, 5e-3, 10e-3]:
        A, B = two_lens_AB(d1_test, d2_t, d3_t, f1, f2)
        B_rest = B - A * d1_test
        print(f"    d₁={d1_test*1e3:.1f}mm: B={B:.6e}, A·d₁={A*d1_test:.6e}, "
              f"B_rest={B_rest:.6e}")
    
    print(f"\n  → B_rest is the same for all d₁ ✓ (depends only on d₂,d₃,φ₁,φ₂)")


# ================================================================
# TEST 10: MINIMUM SETTINGS FOR DUAL WOBBLE, AB ONLY
# ================================================================

def test10_minimum_dual_wobble():
    """
    For dual-lens wobble with linear models (7 unknowns), 
    systematically test the minimum number of settings needed.
    
    With AB-only: need K ≥ 4 settings (8 constraints for 7 unknowns).
    But does K=4 actually give unique solution?
    """
    print("\n" + "=" * 70)
    print("TEST 10: Minimum K for dual-lens wobble uniqueness (AB only)")
    print("=" * 70)
    
    d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
    a1_t = 1.0 / 3.0e-3
    b1_t = 5000.0
    a2_t = 1.0 / 50.0e-3
    b2_t = 200.0
    
    true_params = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])
    
    # Various wobble configurations with K settings
    configs = [
        ("K=3: (0,0),(w,0),(0,w)",
         [(0, 0), (0.01, 0), (0, 0.01)]),
        
        ("K=4: + diagonal",
         [(0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01)]),
        
        ("K=4: L-shape + mid",
         [(0, 0), (0.01, 0), (0, 0.01), (0.005, 0.005)]),
        
        ("K=5: + 2 mid-points",
         [(0, 0), (0.01, 0), (0, 0.01), (0.005, 0.005), (0.01, 0.01)]),
        
        ("K=6: 3×2 grid",
         list(product([0.0, 0.005, 0.01], [0.0, 0.01]))),
        
        ("K=9: 3×3 grid",
         list(product([0.0, 0.005, 0.01], [0.0, 0.005, 0.01]))),
        
        ("K=5: wider wobble",
         [(0, 0), (0.02, 0), (0, 0.02), (0.01, 0.01), (0.02, 0.02)]),
        
        ("K=4: wider 2×2",
         [(0, 0), (0.02, 0), (0, 0.02), (0.02, 0.02)]),
    ]
    
    print(f"\n  All tests use: AB only (no ABCD recovery, no d1-defocus)")
    print(f"  Unknowns = 7: d₁, d₂, d₃, a₁, b₁, a₂, b₂\n")
    
    for config_name, wobble_pairs in configs:
        K = len(wobble_pairs)
        n_constraints = 2 * K
        
        targets = []
        for w1, w2 in wobble_pairs:
            phi1 = a1_t + b1_t * w1
            phi2 = a2_t + b2_t * w2
            A, B = two_lens_AB(d1_t, d2_t, d3_t, 1.0/phi1, 1.0/phi2)
            targets.extend([A, B])
        targets = np.array(targets)
        
        def make_res(wp, tgt):
            def residual(params):
                d1, d2, d3, a1, b1, a2, b2 = np.abs(params)
                res = []
                for w1, w2 in wp:
                    phi1 = a1 + b1 * w1
                    phi2 = a2 + b2 * w2
                    A, B = two_lens_AB(d1, d2, d3, 1.0/phi1, 1.0/phi2)
                    res.extend([A, B])
                return np.array(res) - tgt
            return residual
        
        residual = make_res(wobble_pairs, targets)
        
        # Jacobian
        J = np.zeros((n_constraints, 7))
        eps = 1e-10
        f0 = residual(true_params)
        for i in range(7):
            p = true_params.copy()
            p[i] += eps
            J[:, i] = (residual(p) - f0) / eps
        
        rank = np.linalg.matrix_rank(J, tol=1e-8)
        sv = np.linalg.svd(J, compute_uv=False)
        
        # Multi-start
        n_trials = 80
        solutions = []
        for trial in range(n_trials):
            np.random.seed(trial)
            scale = 0.5 + np.random.rand(7)
            x0 = true_params * scale
            result = least_squares(residual, x0, method='lm', max_nfev=10000,
                                  ftol=1e-15, xtol=1e-15)
            if result.cost < 1e-14:
                solutions.append(np.abs(result.x))
        
        unique = []
        for sol in solutions:
            if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
                unique.append(sol)
        
        n_true = sum(1 for s in unique if np.allclose(s, true_params, rtol=0.01))
        
        status_sym = "✓ UNIQUE" if len(unique) == 1 and n_true == 1 else \
                     f"~ {len(unique)} sol ({n_true} true)"
        
        print(f"  {config_name:30s}: {n_constraints:2d} constr, rank={rank}/7, "
              f"conv={len(solutions):2d}/80, distinct={len(unique):2d} → {status_sym}")
        
        if len(unique) == 1:
            sol = unique[0]
            err = np.abs(sol - true_params) / true_params * 100
            print(f"    → max_err = {err.max():.6f}%")


# ================================================================
# MAIN
# ================================================================

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  CORRECTED SOLVABILITY: d1 defocus (sample plane moves)         ║")
    print("╚" + "═" * 68 + "╝")
    
    test0_d1_vs_d3_defocus()
    test1_equation_counting()
    test9_symbolic_structure()
    test2_one_lens_wobble_AB()
    test3_both_lenses_wobble_AB()
    test4_d1_defocus_with_wobble()
    test5_no_defocus_needed()
    test6_scaling_check()
    test7_analyze_degeneracy()
    test8_fix_one_distance()
    test10_minimum_dual_wobble()
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║                    FINAL CONCLUSIONS                             ║")
    print("╚" + "═" * 68 + "╝")
    
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORRECTION: d1-DEFOCUS vs d3-DEFOCUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

With d3-defocus (detector moves):
  M(d3+Δz) = P(Δz)·M → A and B both change → slope gives C, D
  → Each setting gives 3 independent constraints (A,B,C or A,B,D)
  → 3 settings × 3 constraints = 9 for 7 unknowns → easily solvable

With d1-defocus (sample plane moves):
  M(d1+Δz) = M·P(Δz) → A stays constant, B changes but slope = A (known)
  → Each setting gives only 2 independent constraints (A, B₀)
  → Need MORE wobble settings to compensate
  → d1-defocus adds NO independent information beyond noise averaging

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT WORKS AND WHAT DOESN'T:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✗ One-lens wobble (any number of settings):
  The d₂-φ₂ degeneracy makes the problem have infinitely many solutions.
  Adding d1-defocus doesn't help.

? Both-lens wobble with linear models (7 unknowns):
  Need K ≥ 4 settings for equation counting.
  Uniqueness depends on specific wobble configuration.
  See TEST 10 results for minimum K.

✓ Both-lens wobble + fix d₃ (6 unknowns):
  With known d₃, fewer unknowns makes the problem easier.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRACTICAL IMPLICATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. d1-defocus images are useful for:
   - Noise averaging on A and B estimates
   - Verifying A is truly constant (consistency check)
   - NOT for recovering additional ABCD information

2. The number of WOBBLE SETTINGS (distinct focal length combinations)
   is what determines solvability, not the number of images.

3. You MUST wobble both lenses to break the d₂-φ₂ degeneracy.

4. With both lenses wobbled (linear model), need ≥ 4 distinct settings.

5. If d₃ can be measured directly, the problem becomes much easier.
""")


if __name__ == "__main__":
    main()

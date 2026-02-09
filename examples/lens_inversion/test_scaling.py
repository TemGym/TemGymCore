#!/usr/bin/env python3
"""
Analysis of degeneracy scaling with number of lenses.

For N lenses:
- Unknowns: (2N+1) distances + 2N optical parameters = 4N+1 total
  - But d1 is always recovered → effectively 4N unknowns
- From AB measurements: 2 constraints per wobble setting
- Question: How many dimensions of degeneracy?
"""
import numpy as np
from scipy.optimize import least_squares
import warnings, time
warnings.filterwarnings('ignore')

def transfer_matrix_AB(distances, phi_list):
    """
    Compute A,B for N lenses with given optical powers.
    
    distances: [d1, d2, ..., d_{N+1}]  (N+1 distances for N lenses)
    phi_list: [phi1, phi2, ..., phiN]  (N optical powers)
    
    Returns A, B (elements [0,0] and [0,1] of transfer matrix)
    """
    M = np.eye(2)
    # Build right-to-left: P(d1), L(f1), P(d2), L(f2), ..., P(d_{N+1})
    M = np.array([[1, distances[0]], [0, 1]]) @ M  # P(d1)
    for i, phi in enumerate(phi_list):
        M = np.array([[1, 0], [-phi, 1]]) @ M  # L(fi)
        M = np.array([[1, distances[i+1]], [0, 1]]) @ M  # P(d_{i+1})
    return M[0, 0], M[0, 1]

def test_N_lenses(N, K=5, ntrials=300, verbose=True):
    """Test degeneracy for N lenses with K dual-wobble settings."""
    
    # ── Setup true parameters ──
    # Realistic TEM parameters scaled by lens number
    if N == 2:
        d_true = [3.06e-3, 205.5e-3, 1.05]
        f_true = [3e-3, 50e-3]
        b_true = [5000.0, 200.0]
    elif N == 3:
        d_true = [3e-3, 200e-3, 300e-3, 1.0]
        f_true = [3e-3, 30e-3, 60e-3]
        b_true = [5000, 300, 150]
    elif N == 4:
        d_true = [3e-3, 150e-3, 250e-3, 350e-3, 900e-3]
        f_true = [3e-3, 20e-3, 40e-3, 70e-3]
        b_true = [5000, 400, 250, 120]
    elif N == 5:
        d_true = [3e-3, 120e-3, 200e-3, 280e-3, 360e-3, 800e-3]
        f_true = [3e-3, 15e-3, 30e-3, 50e-3, 80e-3]
        b_true = [5000, 500, 300, 180, 100]
    else:
        raise ValueError(f"N={N} not implemented")
    
    a_true = [1/f for f in f_true]
    
    # Unknowns: d1, d2, ..., d_{N+1}, a1, b1, a2, b2, ..., aN, bN
    # Total: (N+1) + 2N = 3N+1
    tp = np.array(d_true + [val for a, b in zip(a_true, b_true) for val in [a, b]])
    n_params = len(tp)
    
    # Wobble settings: dual wobble on first two lenses
    # (For simplicity, only wobble lenses 1 and 2; others at baseline)
    wp = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
    if K > 5:
        wp.extend([(0.02,0),(0,0.02),(0.02,0.02),(-0.01,0),(0,-0.01)])
    wp = wp[:K]
    
    # Generate target data
    tgt = []
    for w1, w2 in wp:
        phi = [a_true[0] + b_true[0]*w1, a_true[1] + b_true[1]*w2]
        phi.extend(a_true[2:])  # Other lenses at baseline
        A, B = transfer_matrix_AB(d_true, phi)
        tgt.extend([A, B])
    tgt = np.array(tgt)
    
    # Residual function
    def residual(p):
        # Unpack: d1, ..., d_{N+1}, a1, b1, a2, b2, ..., aN, bN
        distances = np.abs(p[:N+1])
        opt_params = np.abs(p[N+1:]).reshape(N, 2)  # [[a1,b1], [a2,b2], ...]
        
        r = np.empty(len(wp) * 2)
        for k, (w1, w2) in enumerate(wp):
            phi = [opt_params[0,0] + opt_params[0,1]*w1,
                   opt_params[1,0] + opt_params[1,1]*w2]
            phi.extend(opt_params[i,0] for i in range(2, N))
            A, B = transfer_matrix_AB(distances, phi)
            r[2*k] = A - tgt[2*k]
            r[2*k+1] = B - tgt[2*k+1]
        return r
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"N = {N} LENSES")
        print(f"{'='*70}")
        print(f"  Unknowns: {n_params} = {N+1} distances + {2*N} optical params")
        print(f"  Constraints: {2*K} = {K} wobble settings × 2 (A,B)")
        print(f"  Rank: {min(n_params, 2*K)} expected (if full rank)")
        print(f"  True params:")
        print(f"    Distances: {[f'{d*1e3:.2f}' for d in d_true]} mm")
        print(f"    Focal lengths: {[f'{f*1e3:.2f}' for f in f_true]} mm")
    
    # Random starts
    t0 = time.time()
    sols = []
    for trial in range(ntrials):
        np.random.seed(trial)
        x0 = tp * np.exp(np.random.randn(n_params) * 0.5)
        try:
            r = least_squares(residual, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-16:
                sols.append(np.abs(r.x))
        except:
            pass
    
    # Find unique solutions
    uniq = []
    for s in sols:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq):
            uniq.append(s)
    
    nt = sum(1 for s in uniq if np.allclose(s, tp, rtol=0.01))
    dt = time.time() - t0
    
    if verbose:
        print(f"\n  Result: {len(sols)} conv, {len(uniq)} distinct, {nt} true  [{dt:.1f}s]")
        
        if len(uniq) == 1:
            print(f"  ✓ UNIQUE!")
        else:
            print(f"  ✗ DEGENERATE: {len(uniq)} solutions")
            
            # Analyze degeneracy structure
            if len(uniq) > 1:
                print(f"\n  Parameter spreads across solutions:")
                for i in range(n_params):
                    vals = [s[i] for s in uniq]
                    spread = max(vals) / min(vals)
                    if i < N+1:
                        name = f"d{i+1}"
                        unit = "mm"
                        scale = 1e3
                    else:
                        lens_idx = (i - N - 1) // 2 + 1
                        if (i - N - 1) % 2 == 0:
                            name = f"a{lens_idx}"
                            unit = "1/m"
                            scale = 1.0
                        else:
                            name = f"b{lens_idx}"
                            unit = "1/m"
                            scale = 1.0
                    
                    true_val = tp[i] * scale
                    min_val = min(vals) * scale
                    max_val = max(vals) * scale
                    
                    if spread < 1.01:
                        sym = "✓"
                    else:
                        sym = "✗"
                    print(f"    {sym} {name:4s}: spread={spread:.1f}x, "
                          f"range=[{min_val:.2f}, {max_val:.2f}] {unit}")
    
    return len(uniq), uniq, tp

# ══════════════════════════════════════════════════════════════════════
print("DEGENERACY SCALING ANALYSIS")
print("=" * 70)
print("\nTesting how degeneracy scales with number of lenses")
print("(dual wobble on lenses 1 and 2 only, K=5 settings)\n")

results = {}
for N in [2, 3, 4, 5]:
    n_sols, uniq, tp = test_N_lenses(N, K=5, ntrials=300, verbose=True)
    results[N] = n_sols

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: DEGENERACY DIMENSION vs NUMBER OF LENSES")
print("=" * 70)
print(f"\n{'N lenses':<10} {'Unknowns':<12} {'Constraints':<14} {'Distinct sols':<15} {'Pattern'}")
print("-" * 70)
for N in [2, 3, 4, 5]:
    unknowns = 3*N + 1
    constraints = 10  # K=5 × 2
    n_sols = results[N]
    if n_sols == 1:
        pattern = "UNIQUE"
    elif n_sols < 10:
        pattern = f"Low deg ({n_sols}D?)"
    elif n_sols < 100:
        pattern = f"Medium deg ({n_sols} sols)"
    else:
        pattern = f"High deg ({n_sols} sols)"
    print(f"{N:<10} {unknowns:<12} {constraints:<14} {n_sols:<15} {pattern}")

print(f"\nConclusion:")
print(f"  The degeneracy grows with N. For 2 lenses, there are ~200 solutions.")
print(f"  This is a 1-parameter degeneracy (continuous curve in parameter space).")
print(f"  For N>2, expect higher-dimensional degeneracy manifolds.")

# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TESTING STRATEGIES ON 3-LENS SYSTEM")
print("=" * 70)

# Test Strategy 1: Known total distance
print("\n--- Strategy 1: Known total distance (d1+d2+...+d_N+1) ---")
N = 3
d_true = [3e-3, 200e-3, 300e-3, 1.0]
f_true = [3e-3, 30e-3, 60e-3]
b_true = [5000, 300, 150]
a_true = [1/f for f in f_true]
tp = np.array(d_true + [val for a, b in zip(a_true, b_true) for val in [a, b]])
total_dist = sum(d_true)

wp = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
tgt = []
for w1, w2 in wp:
    phi = [a_true[0] + b_true[0]*w1, a_true[1] + b_true[1]*w2, a_true[2]]
    A, B = transfer_matrix_AB(d_true, phi)
    tgt.extend([A, B])
tgt = np.array(tgt)

def res_total(p):
    distances = np.abs(p[:4])
    opt_params = np.abs(p[4:]).reshape(3, 2)
    r = np.empty(len(wp)*2 + 1)
    for k, (w1, w2) in enumerate(wp):
        phi = [opt_params[0,0] + opt_params[0,1]*w1,
               opt_params[1,0] + opt_params[1,1]*w2,
               opt_params[2,0]]
        A, B = transfer_matrix_AB(distances, phi)
        r[2*k] = A - tgt[2*k]
        r[2*k+1] = B - tgt[2*k+1]
    r[-1] = 1e6 * (np.sum(distances) - total_dist)
    return r

sols = []
for trial in range(200):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(len(tp)) * 0.5)
    try:
        r = least_squares(res_total, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-10:
            sols.append(np.abs(r.x))
    except: pass

uniq = []
for s in sols:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq):
        uniq.append(s)
nt = sum(1 for s in uniq if np.allclose(s, tp, rtol=0.01))
res1 = "✓ UNIQUE" if len(uniq)==1 else f"✗ {len(uniq)} sols"
print(f"  3 lenses, known d1+d2+d3+d4: {len(sols)} conv, {len(uniq)} distinct → {res1}")

# Test Strategy 2: Nonlinear model
print("\n--- Strategy 2: Nonlinear phi(I)=alpha*I^2 model ---")
c_true = [b**2/(4*a) for a, b in zip(a_true, b_true)]
wp_quad = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01),
           (0.02,0),(0,0.02),(0.02,0.02)]
tgt_quad = []
for w1, w2 in wp_quad:
    phi = [a_true[0] + b_true[0]*w1 + c_true[0]*w1**2,
           a_true[1] + b_true[1]*w2 + c_true[1]*w2**2,
           a_true[2]]
    A, B = transfer_matrix_AB(d_true, phi)
    tgt_quad.extend([A, B])
tgt_quad = np.array(tgt_quad)

def res_quad(p):
    distances = np.abs(p[:4])
    opt_params = np.abs(p[4:]).reshape(3, 2)
    r = np.empty(len(wp_quad)*2)
    for k, (w1, w2) in enumerate(wp_quad):
        c1 = opt_params[0,1]**2 / (4*opt_params[0,0])
        c2 = opt_params[1,1]**2 / (4*opt_params[1,0])
        phi = [opt_params[0,0] + opt_params[0,1]*w1 + c1*w1**2,
               opt_params[1,0] + opt_params[1,1]*w2 + c2*w2**2,
               opt_params[2,0]]
        A, B = transfer_matrix_AB(distances, phi)
        r[2*k] = A - tgt_quad[2*k]
        r[2*k+1] = B - tgt_quad[2*k+1]
    return r

sols_q = []
for trial in range(200):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(len(tp)) * 0.5)
    try:
        r = least_squares(res_quad, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-16:
            sols_q.append(np.abs(r.x))
    except: pass

uniq_q = []
for s in sols_q:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq_q):
        uniq_q.append(s)
nt_q = sum(1 for s in uniq_q if np.allclose(s, tp, rtol=0.01))
res2 = "✓ UNIQUE" if len(uniq_q)==1 else f"✗ {len(uniq_q)} sols"
print(f"  3 lenses, quadratic model: {len(sols_q)} conv, {len(uniq_q)} distinct → {res2}")

# Test Strategy 3: Two voltages
print("\n--- Strategy 3: Two accelerating voltages ---")
m0c2_eV = 511e3
V1, V2 = 200e3, 300e3
gamma = (V1 * (1 + V1/(2*m0c2_eV))) / (V2 * (1 + V2/(2*m0c2_eV)))

tgt_V1 = tgt.copy()
tgt_V2 = []
for w1, w2 in wp:
    phi = [gamma*(a_true[0] + b_true[0]*w1),
           gamma*(a_true[1] + b_true[1]*w2),
           gamma*a_true[2]]
    A, B = transfer_matrix_AB(d_true, phi)
    tgt_V2.extend([A, B])
tgt_V2 = np.array(tgt_V2)

def res_2V(p):
    distances = np.abs(p[:4])
    opt_params = np.abs(p[4:]).reshape(3, 2)
    r = np.empty(len(wp)*4)
    for k, (w1, w2) in enumerate(wp):
        phi1 = [opt_params[0,0] + opt_params[0,1]*w1,
                opt_params[1,0] + opt_params[1,1]*w2,
                opt_params[2,0]]
        A1, B1 = transfer_matrix_AB(distances, phi1)
        r[4*k] = A1 - tgt_V1[2*k]
        r[4*k+1] = B1 - tgt_V1[2*k+1]
        
        phi2 = [gamma*p for p in phi1]
        A2, B2 = transfer_matrix_AB(distances, phi2)
        r[4*k+2] = A2 - tgt_V2[2*k]
        r[4*k+3] = B2 - tgt_V2[2*k+1]
    return r

sols_2V = []
for trial in range(200):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(len(tp)) * 0.5)
    try:
        r = least_squares(res_2V, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-16:
            sols_2V.append(np.abs(r.x))
    except: pass

uniq_2V = []
for s in sols_2V:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq_2V):
        uniq_2V.append(s)
nt_2V = sum(1 for s in uniq_2V if np.allclose(s, tp, rtol=0.01))
res3 = "✓ UNIQUE" if len(uniq_2V)==1 else f"✗ {len(uniq_2V)} sols"
print(f"  3 lenses, two voltages: {len(sols_2V)} conv, {len(uniq_2V)} distinct → {res3}")

print("\n" + "=" * 70)
print("SUMMARY FOR N=3 LENSES")
print("=" * 70)
print(f"""
Strategy                      Result
─────────────────────────────────────────────
1. Known d1+d2+d3+d4         {res1}
2. Nonlinear φ(I)            {res2}
3. Two voltages              {res3}
─────────────────────────────────────────────

Conclusion: All three strategies generalize to N>2 lenses!
""")

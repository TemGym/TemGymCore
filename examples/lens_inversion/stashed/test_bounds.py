"""
Test: Do physical bounds on (d1,d2,d3,f1,f2) eliminate spurious solutions?

The 7-unknown dual-wobble problem has ~70-80 global solutions with AB-only.
If all spurious solutions violate reasonable physical bounds, the problem 
becomes practically solvable.
"""
import numpy as np
from scipy.optimize import least_squares
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def AB_fast(d1, d2, d3, phi1, phi2):
    A = (1.0 - d2 * phi1) * (1.0 - d3 * phi2) - d3 * phi1
    B_rest = d2 * (1.0 - d3 * phi2) + d3
    B = d1 * A + B_rest
    return A, B


# ═══════════════════════════════════════════
# TRUE PARAMS
# ═══════════════════════════════════════════
d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
a1_t, b1_t = 1/3.0e-3, 5000.0
a2_t, b2_t = 1/50.0e-3, 200.0
true7 = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])

names = ['d1', 'd2', 'd3', 'a1(=1/f1)', 'b1', 'a2(=1/f2)', 'b2']
units_scale = [1e3, 1e3, 1e3, 1, 1, 1, 1]  # mm for distances
units_label = ['mm', 'mm', 'mm', '1/m', '1/m', '1/m', '1/m']


# ═══════════════════════════════════════════
# COLLECT ALL SPURIOUS SOLUTIONS (dual wobble K=5)
# ═══════════════════════════════════════════
print("=" * 70)
print("ANALYZING SPURIOUS SOLUTIONS (dual wobble K=5, AB only)")
print("=" * 70)

wp = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
K = len(wp)

tgt = []
for w1,w2 in wp:
    A,B = AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w1, a2_t+b2_t*w2)
    tgt.extend([A,B])
tgt = np.array(tgt)

def residual(params):
    d1,d2,d3,a1,b1,a2,b2 = np.abs(params)
    r = np.empty(K*2)
    for k,(w1,w2) in enumerate(wp):
        A,B = AB_fast(d1,d2,d3, a1+b1*w1, a2+b2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    return r

# Collect solutions with MORE trials for good coverage
n_trials = 500
all_solutions = []
for trial in range(n_trials):
    np.random.seed(trial)
    x0 = true7 * (0.3 + 1.4 * np.random.rand(7))
    try:
        r = least_squares(residual, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-14:
            all_solutions.append(np.abs(r.x))
    except:
        pass

# Deduplicate
unique = []
for sol in all_solutions:
    if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
        unique.append(sol)

print(f"\nCollected {len(all_solutions)} solutions, {len(unique)} distinct.\n")


# ═══════════════════════════════════════════
# PARAMETER RANGES ACROSS ALL SOLUTIONS
# ═══════════════════════════════════════════
print("PARAMETER RANGES ACROSS ALL SPURIOUS SOLUTIONS:")
print("-" * 70)
sols_arr = np.array(unique)

print(f"{'Param':12s} {'True':>12s} {'Min':>12s} {'Max':>12s} {'Ratio max/min':>14s}")
print("-" * 70)
for i, (nm, sc, ul) in enumerate(zip(names, units_scale, units_label)):
    vals = sols_arr[:, i] * sc
    true_val = true7[i] * sc
    ratio = vals.max() / vals.min() if vals.min() > 0 else float('inf')
    print(f"{nm:12s} {true_val:12.4f} {vals.min():12.4f} {vals.max():12.4f} {ratio:14.1f}x  {ul}")

# Convert to physical quantities
print(f"\n\nPHYSICAL RANGES (focal lengths instead of optical powers):")
print("-" * 70)
f1_vals = 1.0 / sols_arr[:, 3] * 1e3  # mm
f2_vals = 1.0 / sols_arr[:, 5] * 1e3  # mm
d1_vals = sols_arr[:, 0] * 1e3  # mm
d2_vals = sols_arr[:, 1] * 1e3  # mm
d3_vals = sols_arr[:, 2] * 1e3  # mm

for nm, vals, true_v in [
    ("d1 (mm)", d1_vals, d1_t*1e3),
    ("d2 (mm)", d2_vals, d2_t*1e3),
    ("d3 (mm)", d3_vals, d3_t*1e3),
    ("f1 (mm)", f1_vals, 3.0),
    ("f2 (mm)", f2_vals, 50.0),
]:
    print(f"  {nm:10s}: true={true_v:10.3f}, range=[{vals.min():10.3f}, {vals.max():10.3f}], "
          f"spread={vals.max()/vals.min():.1f}x")


# ═══════════════════════════════════════════
# TEST WITH BOUNDS
# ═══════════════════════════════════════════
print("\n\n" + "=" * 70)
print("BOUNDED OPTIMIZATION: Can physical bounds eliminate spurious solutions?")
print("=" * 70)

# Define reasonable physical bounds
# Typical TEM: d1 ~ 1-10mm, d2 ~ 50-500mm, d3 ~ 200-2000mm
# f1 ~ 1-10mm (strong obj lens), f2 ~ 10-200mm (intermediate lens)
bound_scenarios = [
    ("Very loose (10x)",
     {'d1': (0.3e-3, 30e-3), 'd2': (20e-3, 2000e-3), 'd3': (0.1, 10.0),
      'f1': (0.3e-3, 30e-3), 'f2': (5e-3, 500e-3), 'b1': (100, 50000), 'b2': (5, 20000)}),
    
    ("Loose (5x)",
     {'d1': (0.6e-3, 15e-3), 'd2': (40e-3, 1000e-3), 'd3': (0.2, 5.0),
      'f1': (0.6e-3, 15e-3), 'f2': (10e-3, 250e-3), 'b1': (500, 25000), 'b2': (20, 10000)}),
    
    ("Moderate (3x)",
     {'d1': (1e-3, 10e-3), 'd2': (70e-3, 600e-3), 'd3': (0.35, 3.0),
      'f1': (1e-3, 10e-3), 'f2': (15e-3, 150e-3), 'b1': (1000, 15000), 'b2': (50, 5000)}),
    
    ("Tight (2x)",
     {'d1': (1.5e-3, 6e-3), 'd2': (100e-3, 400e-3), 'd3': (0.5, 2.0),
      'f1': (1.5e-3, 6e-3), 'f2': (25e-3, 100e-3), 'b1': (2000, 10000), 'b2': (80, 2000)}),
    
    ("Very tight (1.5x)",
     {'d1': (2e-3, 5e-3), 'd2': (140e-3, 300e-3), 'd3': (0.7, 1.5),
      'f1': (2e-3, 5e-3), 'f2': (33e-3, 75e-3), 'b1': (3000, 8000), 'b2': (100, 500)}),
]

for scenario_name, bounds in bound_scenarios:
    # Count how many of the unbounded solutions fall within these bounds
    n_within = 0
    within_solutions = []
    for sol in unique:
        d1, d2, d3, a1, b1, a2, b2 = sol
        f1 = 1.0 / a1
        f2 = 1.0 / a2
        
        in_bounds = (
            bounds['d1'][0] <= d1 <= bounds['d1'][1] and
            bounds['d2'][0] <= d2 <= bounds['d2'][1] and
            bounds['d3'][0] <= d3 <= bounds['d3'][1] and
            bounds['f1'][0] <= f1 <= bounds['f1'][1] and
            bounds['f2'][0] <= f2 <= bounds['f2'][1] and
            bounds['b1'][0] <= b1 <= bounds['b1'][1] and
            bounds['b2'][0] <= b2 <= bounds['b2'][1]
        )
        if in_bounds:
            n_within += 1
            within_solutions.append(sol)
    
    # Also run bounded optimization from scratch using TRF
    lb = np.array([bounds['d1'][0], bounds['d2'][0], bounds['d3'][0],
                   1.0/bounds['f1'][1], bounds['b1'][0],
                   1.0/bounds['f2'][1], bounds['b2'][0]])
    ub = np.array([bounds['d1'][1], bounds['d2'][1], bounds['d3'][1],
                   1.0/bounds['f1'][0], bounds['b1'][1],
                   1.0/bounds['f2'][0], bounds['b2'][1]])
    
    bounded_solutions = []
    for trial in range(200):
        np.random.seed(trial)
        x0 = lb + np.random.rand(7) * (ub - lb)
        try:
            r = least_squares(residual, x0, method='trf', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15, bounds=(lb, ub))
            if r.cost < 1e-14:
                bounded_solutions.append(np.abs(r.x))
        except:
            pass
    
    bounded_unique = []
    for sol in bounded_solutions:
        if all(not np.allclose(sol, u, rtol=0.005) for u in bounded_unique):
            bounded_unique.append(sol)
    
    n_true_b = sum(1 for s in bounded_unique if np.allclose(s, true7, rtol=0.01))
    
    # Report
    status = "✓ UNIQUE" if len(bounded_unique) == 1 and n_true_b == 1 else \
             f"✗ {len(bounded_unique)} sol ({n_true_b} true)"
    
    print(f"\n  {scenario_name}:")
    print(f"    Unbounded solutions within bounds: {n_within}/{len(unique)}")
    print(f"    Bounded TRF optimization (200 starts): conv={len(bounded_solutions)}, "
          f"distinct={len(bounded_unique)} → {status}")
    
    if len(within_solutions) <= 8:
        for i, sol in enumerate(within_solutions):
            d1,d2,d3,a1,b1,a2,b2 = sol
            err = np.abs(sol - true7) / true7 * 100
            is_true = "✓TRUE" if err.max() < 1.0 else f"err={err.max():.1f}%"
            print(f"      Unbound sol {i+1}: d1={d1*1e3:.3f}mm d2={d2*1e3:.1f}mm d3={d3*1e3:.0f}mm "
                  f"f1={1/a1*1e3:.3f}mm f2={1/a2*1e3:.1f}mm  {is_true}")
    
    if len(bounded_unique) <= 8:
        for i, sol in enumerate(bounded_unique):
            d1,d2,d3,a1,b1,a2,b2 = sol
            err = np.abs(sol - true7) / true7 * 100
            is_true = "✓TRUE" if err.max() < 1.0 else f"err={err.max():.1f}%"
            print(f"      Bounded sol {i+1}: d1={d1*1e3:.3f}mm d2={d2*1e3:.1f}mm d3={d3*1e3:.0f}mm "
                  f"f1={1/a1*1e3:.3f}mm f2={1/a2*1e3:.1f}mm b1={b1:.0f} b2={b2:.0f}  {is_true}")


# ═══════════════════════════════════════════
# CHARACTERIZE THE DEGENERACY
# ═══════════════════════════════════════════
print("\n\n" + "=" * 70)
print("DEGENERACY STRUCTURE: How do spurious solutions relate?")
print("=" * 70)

# Check which parameter pairs are most correlated among solutions
print("\nCorrelation matrix of log(parameters) across solutions:")
log_sols = np.log(sols_arr)
corr = np.corrcoef(log_sols.T)
param_short = ['d1', 'd2', 'd3', 'φ1', 'b1', 'φ2', 'b2']
print(f"{'':6s}", end="")
for p in param_short:
    print(f"{p:>6s}", end="")
print()
for i, p in enumerate(param_short):
    print(f"{p:6s}", end="")
    for j in range(7):
        c = corr[i, j]
        print(f"{c:6.2f}", end="")
    print()

# Check: are there clear functional relationships?
print(f"\n\nKey relationships among spurious solutions:")
print(f"  d2 vs d3 correlation:  r = {corr[1,2]:.3f}")
print(f"  d2 vs φ2 correlation:  r = {corr[1,5]:.3f}")
print(f"  d3 vs φ2 correlation:  r = {corr[2,5]:.3f}")
print(f"  d1 vs φ1 correlation:  r = {corr[0,3]:.3f}")
print(f"  φ1 vs φ2 correlation:  r = {corr[3,5]:.3f}")

# Check if d3 is the main ambiguous parameter
print(f"\n  d3 range: {sols_arr[:,2].min()*1e3:.1f} to {sols_arr[:,2].max()*1e3:.1f} mm "
      f"(true={d3_t*1e3:.0f}mm)")
print(f"  d3 std/mean: {sols_arr[:,2].std()/sols_arr[:,2].mean()*100:.1f}%")

# Show: for solutions with d3 near truth, are other params correct?
near_d3 = [sol for sol in unique if abs(sol[2] - d3_t) / d3_t < 0.1]
print(f"\n  Solutions with d3 within 10% of truth: {len(near_d3)}")
for i, sol in enumerate(near_d3[:5]):
    err = np.abs(sol - true7) / true7 * 100
    print(f"    Sol: d3={sol[2]*1e3:.1f}mm, max_err={err.max():.2f}%, "
          f"which param: {names[np.argmax(err)]}")


# ═══════════════════════════════════════════
# ROBUSTNESS: Test bounds across different parameter regimes
# ═══════════════════════════════════════════
print("\n\n" + "=" * 70)
print("ROBUSTNESS: Bounds test across parameter regimes")
print("=" * 70)
print("Using 'moderate (3x)' bounds centered on each regime's true values\n")

wp5 = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]

regimes = [
    ("Default f1=3mm f2=50mm",   3.06e-3, 205.5e-3, 1.05, 3e-3, 50e-3, 5000, 200),
    ("Equal f1=f2=10mm",         5e-3, 100e-3, 500e-3, 10e-3, 10e-3, 3000, 3000),
    ("Strong f1=1mm f2=100mm",   1.5e-3, 150e-3, 2.0, 1e-3, 100e-3, 10000, 100),
    ("Weak f1=50mm f2=200mm",    60e-3, 500e-3, 3.0, 50e-3, 200e-3, 1000, 50),
]

for nm, d1, d2, d3, f1, f2, b1, b2 in regimes:
    a1 = 1/f1; a2 = 1/f2
    tp = np.array([d1, d2, d3, a1, b1, a2, b2])
    
    tgt_r = []
    for w1,w2 in wp5:
        A,B = AB_fast(d1,d2,d3, a1+b1*w1, a2+b2*w2)
        tgt_r.extend([A,B])
    tgt_r = np.array(tgt_r)
    
    def mkres(wp=wp5, tgt=tgt_r):
        def res(p):
            dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
            r = np.empty(len(wp)*2)
            for k,(w1,w2) in enumerate(wp):
                A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
                r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
            return r
        return res
    res_fn = mkres()
    
    # 3x bounds around true values
    factor = 3.0
    lb = np.array([d1/factor, d2/factor, d3/factor,
                   a1/factor, b1/factor, a2/factor, b2/factor])
    ub = np.array([d1*factor, d2*factor, d3*factor,
                   a1*factor, b1*factor, a2*factor, b2*factor])
    
    # Unbounded
    sols_u = []
    for trial in range(200):
        np.random.seed(trial)
        x0 = tp * (0.5 + np.random.rand(7))
        try:
            r = least_squares(res_fn, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-14:
                sols_u.append(np.abs(r.x))
        except: pass
    uniq_u = []
    for s in sols_u:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq_u):
            uniq_u.append(s)
    
    # Filter unbounded by 3x bounds
    in_bounds = []
    for sol in uniq_u:
        if all(lb[i] <= sol[i] <= ub[i] for i in range(7)):
            in_bounds.append(sol)
    
    # Bounded TRF
    sols_b = []
    for trial in range(200):
        np.random.seed(trial)
        x0 = lb + np.random.rand(7) * (ub - lb)
        try:
            r = least_squares(res_fn, x0, method='trf', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15, bounds=(lb, ub))
            if r.cost < 1e-14:
                sols_b.append(np.abs(r.x))
        except: pass
    uniq_b = []
    for s in sols_b:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq_b):
            uniq_b.append(s)
    
    nt_b = sum(1 for s in uniq_b if np.allclose(s, tp, rtol=0.01))
    st = "✓UNIQUE" if len(uniq_b)==1 and nt_b==1 else f"✗ {len(uniq_b)} sol ({nt_b} true)"
    
    print(f"  {nm}:")
    print(f"    Unbounded: {len(uniq_u)} distinct, {len(in_bounds)} within 3x bounds")
    print(f"    Bounded (3x, TRF): conv={len(sols_b)}, distinct={len(uniq_b)} → {st}")
    
    if len(uniq_b) <= 5:
        for i, sol in enumerate(uniq_b[:5]):
            err = np.abs(sol - tp) / tp * 100
            mark = "✓TRUE" if err.max() < 1.0 else f"err={err.max():.2f}%"
            print(f"      Sol {i+1}: d1={sol[0]*1e3:.3f}mm d2={sol[1]*1e3:.1f}mm "
                  f"d3={sol[2]*1e3:.0f}mm f1={1/sol[3]*1e3:.3f}mm "
                  f"f2={1/sol[5]*1e3:.1f}mm  {mark}")


# ═══════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════
print("\n\n" + "═" * 70)
print("SUMMARY: DO BOUNDS HELP?")
print("═" * 70)

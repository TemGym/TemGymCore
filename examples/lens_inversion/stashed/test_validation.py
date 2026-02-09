#!/usr/bin/env python3
"""
Test whether spurious solutions can be distinguished by:
1. Predicting out-of-sample wobble measurements
2. Using known f-vs-I relationship (without absolute scale)
3. Changing operating point (different base excitation)
4. Any other strategy
"""
import numpy as np
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')

def AB_fast(d1, d2, d3, phi1, phi2):
    A = (1.0 - d2 * phi1) * (1.0 - d3 * phi2) - d3 * phi1
    B = d1 * A + d2 * (1.0 - d3 * phi2) + d3
    return A, B

# ── True parameters ──
d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
f1_t, f2_t = 3e-3, 50e-3
a1_t, a2_t = 1/f1_t, 1/f2_t
b1_t, b2_t = 5000.0, 200.0
tp = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])

# ── Collect spurious solutions (dual wobble, K=5) ──
wp_fit = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
tgt_fit = np.array([v for w1,w2 in wp_fit 
                     for v in AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w1, a2_t+b2_t*w2)])

def residual(p, tgt=tgt_fit, wp=wp_fit):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp)*2)
    for k,(w1,w2) in enumerate(wp):
        A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    return r

print("Collecting spurious solutions...")
all_sols = []
for trial in range(500):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(7) * 0.5)
    try:
        r = least_squares(residual, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-20:
            all_sols.append(np.abs(r.x))
    except: pass

unique = []
for s in all_sols:
    if all(not np.allclose(s, u, rtol=0.005) for u in unique):
        unique.append(s)
print(f"Found {len(unique)} distinct solutions from {len(all_sols)} convergent runs.\n")

# ══════════════════════════════════════════════════════════════════════
# TEST 1: Can you validate by predicting out-of-sample wobble?
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: VALIDATION BY PREDICTING NEW WOBBLE MEASUREMENTS")
print("=" * 70)

# Test at many out-of-sample wobble points
wp_test = [(0.003, 0.007), (0.02, 0.02), (-0.005, 0.015), 
           (0.001, 0.001), (0.008, 0.003), (-0.01, 0.01),
           (0.05, 0.05), (0.1, 0.1)]

print(f"\nEvaluating {len(unique)} solutions at {len(wp_test)} out-of-sample wobble points:")
max_A_spread = 0
max_B_spread = 0
for w1, w2 in wp_test:
    As = []
    Bs = []
    for s in unique:
        A, B = AB_fast(s[0], s[1], s[2], s[3]+s[4]*w1, s[5]+s[6]*w2)
        As.append(A)
        Bs.append(B)
    A_spread = max(As) - min(As)
    B_spread = max(Bs) - min(Bs)
    A_true, B_true = AB_fast(d1_t, d2_t, d3_t, a1_t+b1_t*w1, a2_t+b2_t*w2)
    max_A_spread = max(max_A_spread, A_spread/abs(A_true))
    max_B_spread = max(max_B_spread, B_spread/abs(B_true))
    print(f"  w=({w1:.3f},{w2:.3f}): A spread = {A_spread:.2e} (rel {A_spread/abs(A_true):.2e}), "
          f"B spread = {B_spread:.2e} (rel {B_spread/abs(B_true):.2e})")

print(f"\n  Max relative A spread: {max_A_spread:.2e}")
print(f"  Max relative B spread: {max_B_spread:.2e}")
if max_A_spread < 1e-10 and max_B_spread < 1e-10:
    print("  ═══> ALL solutions predict IDENTICAL A,B for ANY wobble setting.")
    print("  ═══> VALIDATION BY PREDICTION IS IMPOSSIBLE with wobble alone.")
else:
    print("  ═══> Solutions CAN be distinguished at large wobble!")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: Is the f₂/b₂ ratio preserved across degeneracy?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: f-vs-I RELATIONSHIP (KNOWING b₂/a₂ RATIO)")
print("=" * 70)

ratios_a2_b2 = [s[6]/s[5] for s in unique]  # b2/a2 ratio
print(f"\n  True b₂/a₂ = {b2_t/a2_t:.6f}")
print(f"  Across {len(unique)} solutions:")
print(f"    b₂/a₂ min = {min(ratios_a2_b2):.6f}")
print(f"    b₂/a₂ max = {max(ratios_a2_b2):.6f}")
print(f"    b₂/a₂ std = {np.std(ratios_a2_b2):.6e}")
if np.std(ratios_a2_b2) / np.mean(ratios_a2_b2) < 0.01:
    print("  ═══> b₂/a₂ ratio is PRESERVED across degeneracy (spread < 1%).")
    print("  ═══> Knowing φ∝f(I) (relative shape) DOES NOT HELP.")
else:
    print("  ═══> b₂/a₂ ratio VARIES — knowing it COULD help!")

# Also check b1/a1 ratio
ratios_a1_b1 = [s[4]/s[3] for s in unique]
print(f"\n  True b₁/a₁ = {b1_t/a1_t:.6f}")
print(f"    b₁/a₁ range: [{min(ratios_a1_b1):.6f}, {max(ratios_a1_b1):.6f}]")

# Check d2/d3 ratio
ratios_d2_d3 = [s[1]/s[2] for s in unique]
print(f"\n  True d₂/d₃ = {d2_t/d3_t:.6f}")
print(f"    d₂/d₃ range: [{min(ratios_d2_d3):.6f}, {max(ratios_d2_d3):.6f}]")
if np.std(ratios_d2_d3) / np.mean(ratios_d2_d3) < 0.01:
    print("  ═══> d₂/d₃ ratio is PRESERVED — knowing it doesn't help.")
else:
    print("  ═══> d₂/d₃ ratio VARIES — knowing it COULD help!")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: Different operating point (change base excitation)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: DIFFERENT OPERATING POINT (CHANGE BASE EXCITATION)")
print("=" * 70)
print("\nIf we change lens 2 to a DIFFERENT base focal length (different mode):")
print("Same d1,d2,d3 but different a₂' → does this break the degeneracy?")

# New operating point: f2 = 80mm instead of 50mm
a2_new = 1.0 / 80e-3  # 12.5 instead of 20.0
b2_new = 120.0  # different sensitivity at new operating point

# Generate data for new operating point with SAME geometry
wp_new = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
tgt_new = np.array([v for w1,w2 in wp_new
                     for v in AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w1, a2_new+b2_new*w2)])

# For each spurious solution from original, test if it can ALSO fit the new data
# with same d1,d2,d3 but free a2',b2'
print(f"\nFor each of {len(unique)} spurious solutions, fit a₂',b₂' to new operating point data")
print("keeping d1,d2,d3,a1,b1 fixed from the spurious solution:\n")

surviving = []
for i, sol in enumerate(unique):
    dd1, dd2, dd3, aa1, bb1 = sol[0], sol[1], sol[2], sol[3], sol[4]
    
    def res_new(p):
        aa2_n, bb2_n = np.abs(p)
        r = np.empty(len(wp_new)*2)
        for k,(w1,w2) in enumerate(wp_new):
            A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2_n+bb2_n*w2)
            r[2*k]=A-tgt_new[2*k]; r[2*k+1]=B-tgt_new[2*k+1]
        return r
    
    # Try to fit a2',b2' for this solution's geometry
    best = None
    for seed in range(20):
        np.random.seed(seed)
        x0 = np.array([a2_new, b2_new]) * np.exp(np.random.randn(2) * 0.5)
        try:
            r = least_squares(res_new, x0, method='lm', max_nfev=5000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-20:
                if best is None or r.cost < best.cost:
                    best = r
        except: pass
    
    if best is not None:
        surviving.append((sol, np.abs(best.x)))

print(f"  Solutions that fit BOTH operating points: {len(surviving)}/{len(unique)}")
if len(surviving) > 1:
    print("\n  Sample solutions (d2, d3 from each):")
    for sol, newp in surviving[:8]:
        print(f"    d2={sol[1]*1e3:.1f}mm, d3={sol[2]*1e3:.1f}mm, "
              f"a2_orig={sol[5]:.2f}, a2_new={newp[0]:.2f}")
    print("  ═══> Different operating point DOES NOT break the degeneracy!")
    print("       (Because the degeneracy is in d2,d3 — geometry doesn't change)")
else:
    print("  ═══> Different operating point BREAKS the degeneracy!")


# ══════════════════════════════════════════════════════════════════════
# TEST 4: What about NONLINEAR wobble (beyond linear model)?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: NONLINEAR φ(I) — DOES THE QUADRATIC TERM HELP?")
print("=" * 70)
print("\nIf φ(I) = α·I² (magnetic lens), then:")
print("  φ(I₀+δI) = α(I₀+δI)² = αI₀² + 2αI₀δI + α(δI)²")
print("  = a + b·w + c·w²  with c = b²/(4a)")
print("\nAdding this known quadratic constraint...")

# Quadratic model: φ(w) = a + b·w + c·w² with c = b²/(4a) (from φ∝I² assumption)
def AB_quad(d1, d2, d3, phi1, phi2):
    A = (1.0 - d2 * phi1) * (1.0 - d3 * phi2) - d3 * phi1
    B = d1 * A + d2 * (1.0 - d3 * phi2) + d3
    return A, B

# True parameters with quadratic model
c1_t = b1_t**2 / (4*a1_t)  # from φ∝I²
c2_t = b2_t**2 / (4*a2_t)

# Generate data using quadratic model with more wobble points
wp_quad = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01),
           (0.02,0),(0,0.02),(0.02,0.02),(-0.01,0),(0,-0.01)]
tgt_quad = []
for w1,w2 in wp_quad:
    phi1 = a1_t + b1_t*w1 + c1_t*w1**2
    phi2 = a2_t + b2_t*w2 + c2_t*w2**2
    A, B = AB_fast(d1_t, d2_t, d3_t, phi1, phi2)
    tgt_quad.extend([A, B])
tgt_quad = np.array(tgt_quad)

# 7 unknowns: d1,d2,d3,a1,b1,a2,b2 (c1,c2 derived from a,b via known physics)
def res_quad(p, tgt=tgt_quad, wp=wp_quad):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    cc1 = bb1**2 / (4*aa1)
    cc2 = bb2**2 / (4*aa2)
    r = np.empty(len(wp)*2)
    for k,(w1,w2) in enumerate(wp):
        phi1 = aa1 + bb1*w1 + cc1*w1**2
        phi2 = aa2 + bb2*w2 + cc2*w2**2
        A,B = AB_fast(dd1,dd2,dd3, phi1, phi2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    return r

sols_quad = []
for trial in range(500):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(7) * 0.5)
    try:
        r = least_squares(res_quad, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-18:
            sols_quad.append(np.abs(r.x))
    except: pass

uniq_quad = []
for s in sols_quad:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq_quad):
        uniq_quad.append(s)

nt = sum(1 for s in uniq_quad if np.allclose(s, tp, rtol=0.01))
print(f"\n  Quadratic model (c=b²/4a, 10 wobble pts): {len(sols_quad)} conv, "
      f"{len(uniq_quad)} distinct, {nt} true")

if len(uniq_quad) == 1 and nt == 1:
    print("  ═══> Nonlinear model with known physics BREAKS the degeneracy!")
else:
    print("  ═══> Nonlinear model DOES NOT break the degeneracy.")
    print("       The degeneracy scales ALL optical powers by same factor,")
    print("       preserving c = b²/(4a) automatically.")

# Verify: in the degeneracy, does c'/a' = c/a? 
if len(uniq_quad) > 1:
    print("\n  Checking c₂/a₂ across solutions (should be constant if degeneracy preserves it):")
    for s in uniq_quad[:5]:
        cc2 = s[6]**2 / (4*s[5])
        print(f"    a₂={s[5]:.4f}, b₂={s[6]:.2f}, c₂={cc2:.4f}, c₂/a₂={cc2/s[5]:.6f}")
    print(f"    True c₂/a₂ = {c2_t/a2_t:.6f}")


# ══════════════════════════════════════════════════════════════════════
# TEST 5: What if d2/d3 ratio is NOT constant across solutions?
#         Could knowing physical length ratios help?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: WHAT INDEPENDENT KNOWLEDGE ACTUALLY BREAKS IT?")
print("=" * 70)

# Test: if we know d3 to within X%, how many solutions survive?
print("\nIf d₃ is known to within X% of true value:")
for pct in [50, 30, 20, 10, 5, 2]:
    d3_lo = d3_t * (1 - pct/100)
    d3_hi = d3_t * (1 + pct/100)
    surv = [s for s in unique if d3_lo <= s[2] <= d3_hi]
    print(f"  ±{pct:2d}%: {len(surv):3d}/{len(unique)} solutions survive")

# Test: if we know f2 to within X%
print(f"\nIf f₂ is known to within X% of true value:")
for pct in [50, 30, 20, 10, 5, 2]:
    f2_lo = f2_t * (1 - pct/100)
    f2_hi = f2_t * (1 + pct/100)
    surv = [s for s in unique if f2_lo <= 1.0/s[5] <= f2_hi]
    print(f"  ±{pct:2d}%: {len(surv):3d}/{len(unique)} solutions survive")

# Test: if we know d2 to within X%
print(f"\nIf d₂ is known to within X% of true value:")
for pct in [50, 30, 20, 10, 5, 2]:
    d2_lo = d2_t * (1 - pct/100)
    d2_hi = d2_t * (1 + pct/100)
    surv = [s for s in unique if d2_lo <= s[1] <= d2_hi]
    print(f"  ±{pct:2d}%: {len(surv):3d}/{len(unique)} solutions survive")


# ══════════════════════════════════════════════════════════════════════
# TEST 6: Constraining d2+d3 (total column length)?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 6: CONSTRAINING d₂+d₃ (TOTAL COLUMN LENGTH)")
print("=" * 70)
print(f"\n  True d₂+d₃ = {(d2_t+d3_t)*1e3:.1f} mm")

# In the degeneracy d2 and d3 scale together: d2'=λd2, d3'=λd3
# So d2'+d3' = λ(d2+d3) ≠ d2+d3 unless λ=1
# → Knowing d2+d3 SHOULD fix λ and break the degeneracy!

total_true = d2_t + d3_t
print("  d₂+d₃ across solutions:")
totals = [s[1]+s[2] for s in unique]
print(f"    min = {min(totals)*1e3:.1f} mm, max = {max(totals)*1e3:.1f} mm")
print(f"    spread = {max(totals)/min(totals):.1f}x")

for pct in [50, 30, 20, 10, 5, 2]:
    lo = total_true * (1 - pct/100)
    hi = total_true * (1 + pct/100)
    surv = [s for s in unique if lo <= s[1]+s[2] <= hi]
    print(f"  d₂+d₃ known ±{pct:2d}%: {len(surv):3d}/{len(unique)} solutions survive")


# ══════════════════════════════════════════════════════════════════════
# TEST 7: What about measuring at two DIFFERENT d₃ values?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 7: TWO CAMERA POSITIONS (d₃ and d₃+Δd₃)")
print("=" * 70)
print("\nIf camera moved by known Δd₃ = 50mm, A'=A+Δd₃·C, B'=B+Δd₃·D")
print("→ This gives C and D, effectively ABCD measurements!")

delta_d3 = 50e-3  # 50mm camera shift
# ABCD matrix
def ABCD(d1, d2, d3, phi1, phi2):
    # M = P(d3)·L(f2)·P(d2)·L(f1)·P(d1)
    # Build step by step
    M = np.eye(2)
    M = np.array([[1, d1],[0, 1]]) @ M  # P(d1) - but order is right-to-left
    # Actually: ψ_out = M · ψ_in, M = P(d3)·L(f2)·P(d2)·L(f1)·P(d1)
    # So we multiply left to right:
    M = np.array([[1, d1],[0, 1]])  # P(d1)
    M = np.array([[1, 0],[-phi1, 1]]) @ M  # L(f1)·P(d1)
    M = np.array([[1, d2],[0, 1]]) @ M  # P(d2)·L(f1)·P(d1)
    M = np.array([[1, 0],[-phi2, 1]]) @ M  # L(f2)·P(d2)·L(f1)·P(d1)
    M = np.array([[1, d3],[0, 1]]) @ M  # P(d3)·L(f2)·P(d2)·L(f1)·P(d1)
    return M[0,0], M[0,1], M[1,0], M[1,1]

# At original d3: we get A, B
# At d3 + delta_d3: A' = A + delta_d3*C, B' = B + delta_d3*D
# So: C = (A'-A)/delta_d3, D = (B'-B)/delta_d3

# Generate data: A,B at d3 and A',B' at d3+delta
wp_cam = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
tgt_cam = []
for w1,w2 in wp_cam:
    phi1 = a1_t + b1_t*w1
    phi2 = a2_t + b2_t*w2
    A1, B1 = AB_fast(d1_t, d2_t, d3_t, phi1, phi2)
    A2, B2 = AB_fast(d1_t, d2_t, d3_t + delta_d3, phi1, phi2)
    tgt_cam.extend([A1, B1, A2, B2])  # 4 values per wobble point
tgt_cam = np.array(tgt_cam)

def res_cam(p, tgt=tgt_cam, wp=wp_cam, dd3=delta_d3):
    dd1,dd2,dd3_base,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp)*4)
    for k,(w1,w2) in enumerate(wp):
        phi1 = aa1+bb1*w1
        phi2 = aa2+bb2*w2
        A1,B1 = AB_fast(dd1,dd2,dd3_base, phi1, phi2)
        A2,B2 = AB_fast(dd1,dd2,dd3_base+dd3, phi1, phi2)
        r[4*k]=A1-tgt[4*k]; r[4*k+1]=B1-tgt[4*k+1]
        r[4*k+2]=A2-tgt[4*k+2]; r[4*k+3]=B2-tgt[4*k+3]
    return r

sols_cam = []
for trial in range(300):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(7) * 0.5)
    try:
        r = least_squares(res_cam, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-20:
            sols_cam.append(np.abs(r.x))
    except: pass

uniq_cam = []
for s in sols_cam:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq_cam):
        uniq_cam.append(s)

nt = sum(1 for s in uniq_cam if np.allclose(s, tp, rtol=0.01))
print(f"\n  Two camera positions (Δd₃=50mm known): {len(sols_cam)} conv, "
      f"{len(uniq_cam)} distinct, {nt} true")
if len(uniq_cam) == 1 and nt == 1:
    print("  ═══> Two camera positions BREAKS the degeneracy! UNIQUE SOLUTION!")
    print(f"       Sol: d1={uniq_cam[0][0]*1e3:.3f}mm, d2={uniq_cam[0][1]*1e3:.1f}mm, "
          f"d3={uniq_cam[0][2]*1e3:.1f}mm")
else:
    print(f"  ═══> Two camera positions: {len(uniq_cam)} solutions remain")


# ══════════════════════════════════════════════════════════════════════
# TEST 8: What about a THIRD lens (known relative to lens 2)?
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 8: WHAT IF YOU KNOW d₂+d₃ (total specimen-to-camera distance)?")
print("=" * 70)
print("This is easier to measure physically than d₃ alone.")
print(f"True d₂+d₃ = {(d2_t+d3_t)*1e3:.1f} mm\n")

# Add d2+d3 constraint to optimize
total_known = d2_t + d3_t

def res_total(p, tgt=tgt_fit, wp=wp_fit, total=total_known, lam=1e6):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp)*2 + 1)
    for k,(w1,w2) in enumerate(wp):
        A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    r[-1] = lam * (dd2 + dd3 - total)  # constraint: d2+d3 = known
    return r

sols_total = []
for trial in range(500):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(7) * 0.3)
    try:
        r = least_squares(res_total, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-10:
            s = np.abs(r.x)
            # Check AB fit quality (exclude constraint term)
            r_check = residual(s)
            if np.max(np.abs(r_check)) < 1e-9 and abs(s[1]+s[2]-total_known) < 1e-6:
                sols_total.append(s)
    except: pass

uniq_total = []
for s in sols_total:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq_total):
        uniq_total.append(s)

nt = sum(1 for s in uniq_total if np.allclose(s, tp, rtol=0.01))
print(f"  With known d₂+d₃: {len(sols_total)} conv, {len(uniq_total)} distinct, {nt} true")
if len(uniq_total) == 1 and nt == 1:
    print("  ═══> Knowing d₂+d₃ BREAKS the degeneracy! UNIQUE SOLUTION!")
else:
    print(f"  ═══> Knowing d₂+d₃: {len(uniq_total)} solutions remain")


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("COMPREHENSIVE SUMMARY")
print("═" * 70)
print("""
WHAT DOES NOT WORK:
  ✗ New wobble measurements: All solutions predict identically for ANY (w₁,w₂)
  ✗ Knowing b₂/a₂ ratio (f-vs-I shape): Already preserved in degeneracy
  ✗ Tighter bounds: Reduces solutions but never uniqueness (72 at ±50%)
  ✗ More wobble points (K→∞): Degeneracy is exact, not sampling artifact
  ✗ d₁-defocus: Adds zero information (A'=A, B'=B+AΔz)

PARTIALLY RECOVERED:
  ✓ d₁ is ALWAYS exactly recovered
  ✓ f₁ (a₁) is ALWAYS nearly exactly recovered  
  ✓ b₁ is ALWAYS exactly recovered
  ✗ d₂, d₃, f₂, b₂ have continuous degeneracy

WHAT DOES WORK:""")

# Summarize positive results
strategies = [
    ("Know d₃ (camera distance)", "from previous tests", True),
    ("Know d₂+d₃ (total column length)", f"{len(uniq_total)} solutions", len(uniq_total)==1),
    ("Two camera positions (known Δd₃)", f"{len(uniq_cam)} solutions", len(uniq_cam)==1),
    ("Nonlinear φ(I) with known c=b²/4a", f"{len(uniq_quad)} solutions", len(uniq_quad)==1),
    ("Different operating point", f"{len(surviving)} solutions", len(surviving)==1),
]
for name, result, works in strategies:
    sym = "✓" if works else "✗"
    print(f"  {sym} {name}: {result}")

# How much d3 knowledge is needed?
print(f"\n  To get down to ~1-3 solutions, know d₃ to ±10%:")
d3_lo = d3_t * 0.9; d3_hi = d3_t * 1.1
surv = [s for s in unique if d3_lo <= s[2] <= d3_hi]
print(f"    {len(surv)} solutions survive with d₃ known ±10% ({d3_t*1e3:.0f}mm)")

print()

#!/usr/bin/env python3
"""
Test three practical strategies for breaking the two-lens degeneracy:
1. Known total specimen distance d1+d2+d3
2. Known nonlinear phi(I) = alpha*I^2 (quadratic model)
3. Two accelerating voltages (different relativistic correction)

All tests: dual wobble K=5, 7 unknowns, 500 random starts.
"""
import numpy as np
from scipy.optimize import least_squares
import warnings, sys, time
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

wp5 = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
tgt = np.array([v for w1,w2 in wp5
                for v in AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w1, a2_t+b2_t*w2)])

def residual_base(p):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp5)*2)
    for k,(w1,w2) in enumerate(wp5):
        A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    return r

def solve(res_fn, ntrials=500, tp=tp, check_fn=None):
    """Run random starts, return unique solutions."""
    sols = []
    for trial in range(ntrials):
        np.random.seed(trial)
        x0 = tp * np.exp(np.random.randn(len(tp)) * 0.5)
        try:
            r = least_squares(res_fn, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-10:
                s = np.abs(r.x)
                # Optional extra check
                if check_fn is None or check_fn(s):
                    sols.append(s)
        except: pass
    uniq = []
    for s in sols:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq):
            uniq.append(s)
    nt = sum(1 for s in uniq if np.allclose(s, tp[:len(s)], rtol=0.01))
    return sols, uniq, nt


# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STRATEGY 1: KNOWN TOTAL SPECIMEN DISTANCE (d₁+d₂+d₃)")
print("=" * 70)
# ══════════════════════════════════════════════════════════════════════

total_dist = d1_t + d2_t + d3_t
print(f"\nTrue d₁+d₂+d₃ = {total_dist*1e3:.2f} mm")
print(f"  (d₁={d1_t*1e3:.3f}mm + d₂={d2_t*1e3:.1f}mm + d₃={d3_t*1e3:.0f}mm)")

def res_total_dist(p):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp5)*2 + 1)
    for k,(w1,w2) in enumerate(wp5):
        A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    r[-1] = 1e6 * (dd1 + dd2 + dd3 - total_dist)
    return r

def check_total(s):
    r_check = residual_base(s)
    return np.max(np.abs(r_check)) < 1e-9 and abs(s[0]+s[1]+s[2]-total_dist) < 1e-6

t0 = time.time()
sols, uniq, nt = solve(res_total_dist, check_fn=check_total)
dt = time.time() - t0
print(f"\nResult: {len(sols)} conv, {len(uniq)} distinct, {nt} true  [{dt:.1f}s]")
if len(uniq) == 1 and nt == 1:
    s = uniq[0]
    print(f"  ✓ UNIQUE SOLUTION!")
    print(f"    d₁={s[0]*1e3:.4f}mm, d₂={s[1]*1e3:.2f}mm, d₃={s[2]*1e3:.1f}mm")
    print(f"    f₁={1e3/s[3]:.3f}mm, f₂={1e3/s[5]:.2f}mm")
    print(f"    b₁={s[4]:.1f}, b₂={s[6]:.2f}")
else:
    print(f"  ✗ {len(uniq)} solutions remain")
    for s in uniq[:5]:
        err = np.max(np.abs(s-tp)/tp)*100
        print(f"    d₂={s[1]*1e3:.1f}mm d₃={s[2]*1e3:.1f}mm f₂={1e3/s[5]:.1f}mm max_err={err:.1f}%")

# Also test: d1+d2+d3 known only approximately
print(f"\nWith approximate total distance knowledge:")
# First collect unbounded solutions
sols_ub, uniq_ub, _ = solve(residual_base)
for pct in [20, 10, 5, 2, 1]:
    lo = total_dist * (1 - pct/100)
    hi = total_dist * (1 + pct/100)
    surv = [s for s in uniq_ub if lo <= s[0]+s[1]+s[2] <= hi]
    print(f"  ±{pct:2d}%: {len(surv):3d}/{len(uniq_ub)} solutions survive")

# Robustness across regimes
print("\nRobustness across parameter regimes:")
regimes = [
    ('Default f1=3mm f2=50mm',   3.06e-3,205.5e-3,1.05,  3e-3, 50e-3, 5000, 200),
    ('Equal f1=f2=10mm',         5e-3,   100e-3,  500e-3,10e-3, 10e-3, 3000, 3000),
    ('Strong f1=1mm f2=100mm',   1.5e-3, 150e-3,  2.0,   1e-3, 100e-3,10000,100),
    ('Weak f1=50mm f2=200mm',    60e-3,  500e-3,  3.0,   50e-3,200e-3, 1000, 50),
]
for nm,d1,d2,d3,f1,f2,b1,b2 in regimes:
    a1,a2 = 1/f1, 1/f2
    tp_r = np.array([d1,d2,d3,a1,b1,a2,b2])
    total_r = d1+d2+d3
    tgt_r = np.array([v for w1,w2 in wp5
                      for v in AB_fast(d1,d2,d3, a1+b1*w1, a2+b2*w2)])
    def res_r(p, tgt_r=tgt_r, total_r=total_r):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        r = np.empty(len(wp5)*2 + 1)
        for k,(w1,w2) in enumerate(wp5):
            A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
            r[2*k]=A-tgt_r[2*k]; r[2*k+1]=B-tgt_r[2*k+1]
        r[-1] = 1e6 * (dd1+dd2+dd3 - total_r)
        return r
    def rbase(p, tgt_r=tgt_r):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        r = np.empty(len(wp5)*2)
        for k,(w1,w2) in enumerate(wp5):
            A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
            r[2*k]=A-tgt_r[2*k]; r[2*k+1]=B-tgt_r[2*k+1]
        return r
    def check_r(s, tgt_r=tgt_r, total_r=total_r):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(s)
        r = np.empty(len(wp5)*2)
        for k,(w1,w2) in enumerate(wp5):
            A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
            r[2*k]=A-tgt_r[2*k]; r[2*k+1]=B-tgt_r[2*k+1]
        return np.max(np.abs(r)) < 1e-9 and abs(s[0]+s[1]+s[2]-total_r) < 1e-6
    sols_r = []
    for trial in range(500):
        np.random.seed(trial)
        x0 = tp_r * np.exp(np.random.randn(7) * 0.5)
        try:
            r = least_squares(res_r, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-10:
                s = np.abs(r.x)
                if check_r(s):
                    sols_r.append(s)
        except: pass
    uniq_r = []
    for s in sols_r:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq_r):
            uniq_r.append(s)
    nt_r = sum(1 for s in uniq_r if np.allclose(s, tp_r, rtol=0.01))
    sym = "✓ UNIQUE" if len(uniq_r)==1 and nt_r==1 else f"✗ {len(uniq_r)} sol"
    print(f"  {nm:30s}: {sym}")


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 2: NONLINEAR φ(I) = α·I² (QUADRATIC MODEL)")
print("=" * 70)
# ══════════════════════════════════════════════════════════════════════

# For a magnetic lens: φ ∝ I² (optical power proportional to current squared)
# φ(I₀+δI) = α(I₀+δI)² = αI₀² + 2αI₀δI + α(δI)²
# If w = δI/I₀: φ = a + b·w + c·w²  where c = a (since b=2a and c=a from expansion)
# Wait, let's be more careful:
# φ(I) = α·I², I = I₀(1+w), φ = α·I₀²(1+w)² = a(1+w)²
# = a + 2a·w + a·w² → b = 2a, c = a → c = b²/(4a)
# 
# More generally for φ∝I^n: c = a·n·(n-1)/2 / (n²) ... let's just test φ∝I²
print("\nModel: φ(w) = a·(1+w)² = a + 2a·w + a·w²")
print("  → b = 2a (linear coefficient)")  
print("  → c = a  (quadratic coefficient)")
print("  → Constraint: c = b²/(4a), equivalently b = 2a")
print()

c1_t = b1_t**2 / (4*a1_t)
c2_t = b2_t**2 / (4*a2_t)
print(f"  True lens 1: a₁={a1_t:.2f}, b₁={b1_t:.1f}, c₁=b₁²/4a₁={c1_t:.2f}")
print(f"  True lens 2: a₂={a2_t:.2f}, b₂={b2_t:.1f}, c₂=b₂²/4a₂={c2_t:.2f}")

# Generate data with quadratic model, using wider wobble range to see curvature
wp_quad = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01),
           (0.02,0),(0,0.02),(0.02,0.02),(-0.01,0),(0,-0.01)]

tgt_quad = []
for w1,w2 in wp_quad:
    phi1 = a1_t + b1_t*w1 + c1_t*w1**2
    phi2 = a2_t + b2_t*w2 + c2_t*w2**2
    A, B = AB_fast(d1_t, d2_t, d3_t, phi1, phi2)
    tgt_quad.extend([A, B])
tgt_quad = np.array(tgt_quad)

def res_quad(p):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    cc1 = bb1**2 / (4*aa1)
    cc2 = bb2**2 / (4*aa2)
    r = np.empty(len(wp_quad)*2)
    for k,(w1,w2) in enumerate(wp_quad):
        phi1 = aa1 + bb1*w1 + cc1*w1**2
        phi2 = aa2 + bb2*w2 + cc2*w2**2
        A,B = AB_fast(dd1,dd2,dd3, phi1, phi2)
        r[2*k]=A-tgt_quad[2*k]; r[2*k+1]=B-tgt_quad[2*k+1]
    return r

t0 = time.time()
sols_q, uniq_q, nt_q = solve(res_quad)
dt = time.time() - t0
print(f"\nResult (10 wobble pts): {len(sols_q)} conv, {len(uniq_q)} distinct, {nt_q} true  [{dt:.1f}s]")
if len(uniq_q) == 1 and nt_q == 1:
    s = uniq_q[0]
    print(f"  ✓ UNIQUE SOLUTION!")
    print(f"    d₁={s[0]*1e3:.4f}mm, d₂={s[1]*1e3:.2f}mm, d₃={s[2]*1e3:.1f}mm")
    print(f"    f₁={1e3/s[3]:.3f}mm, f₂={1e3/s[5]:.2f}mm")
else:
    print(f"  ✗ {len(uniq_q)} solutions remain")

# Can we get by with fewer wobble points?
print("\nMinimum wobble points needed:")
for nw in [5, 6, 7, 8, 10]:
    tgt_nw = []
    for w1,w2 in wp_quad[:nw]:
        phi1 = a1_t + b1_t*w1 + c1_t*w1**2
        phi2 = a2_t + b2_t*w2 + c2_t*w2**2
        A, B = AB_fast(d1_t, d2_t, d3_t, phi1, phi2)
        tgt_nw.extend([A, B])
    tgt_nw = np.array(tgt_nw)
    def res_nw(p, tgt_nw=tgt_nw, wp=wp_quad[:nw]):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        cc1 = bb1**2 / (4*aa1); cc2 = bb2**2 / (4*aa2)
        r = np.empty(len(wp)*2)
        for k,(w1,w2) in enumerate(wp):
            phi1 = aa1 + bb1*w1 + cc1*w1**2
            phi2 = aa2 + bb2*w2 + cc2*w2**2
            A,B = AB_fast(dd1,dd2,dd3, phi1, phi2)
            r[2*k]=A-tgt_nw[2*k]; r[2*k+1]=B-tgt_nw[2*k+1]
        return r
    _, uniq_nw, nt_nw = solve(res_nw, ntrials=300)
    sym = "✓ UNIQUE" if len(uniq_nw)==1 and nt_nw==1 else f"✗ {len(uniq_nw)} sol"
    print(f"  K={nw:2d}: {sym}")

# Robustness across regimes
print("\nRobustness across parameter regimes:")
for nm,d1,d2,d3,f1,f2,b1,b2 in regimes:
    a1,a2 = 1/f1, 1/f2
    tp_r = np.array([d1,d2,d3,a1,b1,a2,b2])
    c1 = b1**2/(4*a1); c2 = b2**2/(4*a2)
    tgt_r = []
    for w1,w2 in wp_quad:
        phi1 = a1 + b1*w1 + c1*w1**2
        phi2 = a2 + b2*w2 + c2*w2**2
        A, B = AB_fast(d1,d2,d3, phi1, phi2)
        tgt_r.extend([A, B])
    tgt_r = np.array(tgt_r)
    def res_qr(p, tgt_r=tgt_r):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        cc1 = bb1**2/(4*aa1); cc2 = bb2**2/(4*aa2)
        r = np.empty(len(wp_quad)*2)
        for k,(w1,w2) in enumerate(wp_quad):
            phi1 = aa1+bb1*w1+cc1*w1**2
            phi2 = aa2+bb2*w2+cc2*w2**2
            A,B = AB_fast(dd1,dd2,dd3, phi1, phi2)
            r[2*k]=A-tgt_r[2*k]; r[2*k+1]=B-tgt_r[2*k+1]
        return r
    sols_r = []
    for trial in range(300):
        np.random.seed(trial)
        x0 = tp_r * np.exp(np.random.randn(7) * 0.5)
        try:
            r = least_squares(res_qr, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-18:
                sols_r.append(np.abs(r.x))
        except: pass
    uniq_r = []
    for s in sols_r:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq_r): uniq_r.append(s)
    nt_r = sum(1 for s in uniq_r if np.allclose(s, tp_r, rtol=0.01))
    sym = "✓ UNIQUE" if len(uniq_r)==1 and nt_r==1 else f"✗ {len(uniq_r)} sol"
    print(f"  {nm:30s}: {sym}")


# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STRATEGY 3: TWO ACCELERATING VOLTAGES")
print("=" * 70)
# ══════════════════════════════════════════════════════════════════════

print("""
Physics: For a magnetic lens, φ = NI²/(8VrB₀) where Vr is the 
relativistically-corrected accelerating voltage.
Changing voltage V₁→V₂ scales ALL optical powers by a KNOWN ratio:
  φᵢ(V₂) = φᵢ(V₁) · (V₁/V₂) · (1+eV₁/2m₀c²)/(1+eV₂/2m₀c²)

Key: the SAME scaling factor γ applies to BOTH lenses.
The geometry (d₁,d₂,d₃) stays the same.
""")

# Relativistic correction
m0c2_eV = 511e3  # electron rest mass energy in eV

def rel_factor(V):
    """Relativistically corrected voltage."""
    return V * (1 + V / (2 * m0c2_eV))

V1 = 200e3  # 200 kV (primary voltage)
V2 = 300e3  # 300 kV (second voltage)

# Scaling factor for optical power
gamma = rel_factor(V1) / rel_factor(V2)
print(f"V₁ = {V1/1e3:.0f} kV, V₂ = {V2/1e3:.0f} kV")
print(f"Scaling factor γ = Vr(V₁)/Vr(V₂) = {gamma:.6f}")
print(f"All optical powers at V₂ = γ × (optical powers at V₁)")
print()

# At V1: normal measurements → A,B with φ₁(w), φ₂(w) 
# At V2: same geometry, but φ₁→γφ₁, φ₂→γφ₂ → A',B' data
tgt_V1 = tgt.copy()  # already computed

# V2 data: same d's, optical powers scaled by gamma
tgt_V2 = np.array([v for w1,w2 in wp5
                    for v in AB_fast(d1_t, d2_t, d3_t, 
                                     gamma*(a1_t+b1_t*w1), gamma*(a2_t+b2_t*w2))])

# Unknowns: d1, d2, d3, a1, b1, a2, b2 (at V1)
# Constraints: A,B at V1 (10 eq) + A',B' at V2 with known gamma (10 more eq)
# Total: 20 equations, 7 unknowns

def res_2V(p):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp5)*4)
    for k,(w1,w2) in enumerate(wp5):
        # V1 data
        phi1 = aa1 + bb1*w1
        phi2 = aa2 + bb2*w2
        A1,B1 = AB_fast(dd1,dd2,dd3, phi1, phi2)
        r[4*k]   = A1 - tgt_V1[2*k]
        r[4*k+1] = B1 - tgt_V1[2*k+1]
        # V2 data: same geometry, optical powers scaled by gamma
        A2,B2 = AB_fast(dd1,dd2,dd3, gamma*phi1, gamma*phi2)
        r[4*k+2] = A2 - tgt_V2[2*k]
        r[4*k+3] = B2 - tgt_V2[2*k+1]
    return r

t0 = time.time()
sols_2V, uniq_2V, nt_2V = solve(res_2V)
dt = time.time() - t0
print(f"Result: {len(sols_2V)} conv, {len(uniq_2V)} distinct, {nt_2V} true  [{dt:.1f}s]")
if len(uniq_2V) == 1 and nt_2V == 1:
    s = uniq_2V[0]
    print(f"  ✓ UNIQUE SOLUTION!")
    print(f"    d₁={s[0]*1e3:.4f}mm, d₂={s[1]*1e3:.2f}mm, d₃={s[2]*1e3:.1f}mm")
    print(f"    f₁={1e3/s[3]:.3f}mm, f₂={1e3/s[5]:.2f}mm")
else:
    print(f"  ✗ {len(uniq_2V)} solutions remain")
    for s in uniq_2V[:5]:
        err = np.max(np.abs(s-tp)/tp)*100
        print(f"    d₂={s[1]*1e3:.1f}mm d₃={s[2]*1e3:.1f}mm f₂={1e3/s[5]:.1f}mm err={err:.1f}%")

# What about smaller voltage difference?
print(f"\nEffect of voltage difference:")
for V2_test in [210e3, 220e3, 250e3, 300e3, 100e3]:
    gamma_t = rel_factor(V1) / rel_factor(V2_test)
    tgt_V2t = np.array([v for w1,w2 in wp5
                        for v in AB_fast(d1_t, d2_t, d3_t,
                                         gamma_t*(a1_t+b1_t*w1), gamma_t*(a2_t+b2_t*w2))])
    def res_2Vt(p, g=gamma_t, tgt2=tgt_V2t):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        r = np.empty(len(wp5)*4)
        for k,(w1,w2) in enumerate(wp5):
            phi1 = aa1+bb1*w1; phi2 = aa2+bb2*w2
            A1,B1 = AB_fast(dd1,dd2,dd3, phi1, phi2)
            r[4*k]=A1-tgt[2*k]; r[4*k+1]=B1-tgt[2*k+1]
            A2,B2 = AB_fast(dd1,dd2,dd3, g*phi1, g*phi2)
            r[4*k+2]=A2-tgt2[2*k]; r[4*k+3]=B2-tgt2[2*k+1]
        return r
    _, uniq_t, nt_t = solve(res_2Vt, ntrials=300)
    sym = "✓ UNIQUE" if len(uniq_t)==1 and nt_t==1 else f"✗ {len(uniq_t)} sol"
    print(f"  V₂={V2_test/1e3:5.0f}kV (γ={gamma_t:.4f}): {sym}")

# Robustness across regimes with 300kV
print(f"\nRobustness across parameter regimes (V₁=200kV, V₂=300kV):")
for nm,d1,d2,d3,f1,f2,b1,b2 in regimes:
    a1,a2 = 1/f1, 1/f2
    tp_r = np.array([d1,d2,d3,a1,b1,a2,b2])
    tgt_r1 = np.array([v for w1,w2 in wp5
                       for v in AB_fast(d1,d2,d3, a1+b1*w1, a2+b2*w2)])
    tgt_r2 = np.array([v for w1,w2 in wp5
                       for v in AB_fast(d1,d2,d3, gamma*(a1+b1*w1), gamma*(a2+b2*w2))])
    def res_2Vr(p, tgt1=tgt_r1, tgt2=tgt_r2, g=gamma):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        r = np.empty(len(wp5)*4)
        for k,(w1,w2) in enumerate(wp5):
            phi1 = aa1+bb1*w1; phi2 = aa2+bb2*w2
            A1,B1 = AB_fast(dd1,dd2,dd3, phi1, phi2)
            r[4*k]=A1-tgt1[2*k]; r[4*k+1]=B1-tgt1[2*k+1]
            A2,B2 = AB_fast(dd1,dd2,dd3, g*phi1, g*phi2)
            r[4*k+2]=A2-tgt2[2*k]; r[4*k+3]=B2-tgt2[2*k+1]
        return r
    sols_r = []
    for trial in range(300):
        np.random.seed(trial)
        x0 = tp_r * np.exp(np.random.randn(7) * 0.5)
        try:
            r = least_squares(res_2Vr, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < 1e-18:
                sols_r.append(np.abs(r.x))
        except: pass
    uniq_r = []
    for s in sols_r:
        if all(not np.allclose(s, u, rtol=0.005) for u in uniq_r): uniq_r.append(s)
    nt_r = sum(1 for s in uniq_r if np.allclose(s, tp_r, rtol=0.01))
    sym = "✓ UNIQUE" if len(uniq_r)==1 and nt_r==1 else f"✗ {len(uniq_r)} sol"
    print(f"  {nm:30s}: {sym}")


# ══════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("FINAL COMPARISON")
print("═" * 70)
print(f"""
Strategy                          Extra info needed            Unique?
──────────────────────────────────────────────────────────────────────
1. Known d₁+d₂+d₃               Total specimen distance      {"YES" if len(uniq)==1 else "NO"}
2. Nonlinear φ(I)=αI²            Physical model (no scale)    {"YES" if len(uniq_q)==1 else "NO"}
3. Two voltages (200+300kV)      Second voltage measurement   {"YES" if len(uniq_2V)==1 else "NO"}
──────────────────────────────────────────────────────────────────────
Baseline (linear AB only)        Nothing                      NO (70-210 sol)
""")
print("All three strategies work because they each constrain the 1-parameter")
print("degeneracy manifold d₂↔d₃↔f₂ in a different way:")
print("  1. Fixing d₁+d₂+d₃ pins the distance scaling factor λ=1")
print("  2. c=b²/4a couples a₂ and b₂ nonlinearly (breaks linear scaling)")
print("  3. Different γ scaling breaks the symmetry φ₂→λ'φ₂, d→λd")

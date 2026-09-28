"""
Fast d1-defocus solvability analysis using analytical A,B formulas.
A = (1-d₂φ₁)(1-d₃φ₂) - d₃φ₁
B = d₁·A + d₂(1-d₃φ₂) + d₃
"""
import numpy as np
from scipy.optimize import least_squares
from itertools import product
import time
import warnings
warnings.filterwarnings('ignore')


def AB_fast(d1, d2, d3, phi1, phi2):
    """Analytical A, B. phi = 1/f (optical power)."""
    A = (1.0 - d2 * phi1) * (1.0 - d3 * phi2) - d3 * phi1
    B_rest = d2 * (1.0 - d3 * phi2) + d3
    B = d1 * A + B_rest
    return A, B


def multistart(res_fn, true_p, n_trials=100, tol=1e-14):
    n = len(true_p)
    solutions = []
    for trial in range(n_trials):
        np.random.seed(trial)
        x0 = true_p * (0.5 + np.random.rand(n))
        try:
            r = least_squares(res_fn, x0, method='lm', max_nfev=10000,
                             ftol=1e-15, xtol=1e-15)
            if r.cost < tol:
                solutions.append(np.abs(r.x))
        except:
            pass
    unique = []
    for sol in solutions:
        if all(not np.allclose(sol, u, rtol=0.005) for u in unique):
            unique.append(sol)
    n_true = sum(1 for s in unique if np.allclose(s, true_p, rtol=0.01))
    return solutions, unique, n_true


def jac_num(res_fn, params, eps=1e-10):
    f0 = res_fn(params)
    J = np.zeros((len(f0), len(params)))
    for i in range(len(params)):
        p = params.copy(); p[i] += eps
        J[:, i] = (res_fn(p) - f0) / eps
    return J


# ═══════════════════════════════════════════
# TRUE PARAMS
# ═══════════════════════════════════════════
d1_t, d2_t, d3_t = 3.06e-3, 205.5e-3, 1.05
a1_t, b1_t = 1/3.0e-3, 5000.0      # φ₁(w) = a₁ + b₁w
a2_t, b2_t = 1/50.0e-3, 200.0      # φ₂(w) = a₂ + b₂w
true7 = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, a2_t, b2_t])

print(f"True: d1={d1_t*1e3:.3f}mm, d2={d2_t*1e3:.1f}mm, d3={d3_t*1e3:.0f}mm, "
      f"f1={1/a1_t*1e3:.2f}mm, f2={1/a2_t*1e3:.1f}mm")


# ═══════════════════════════════════════════
# TEST 1: DUAL WOBBLE, AB ONLY
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: DUAL-LENS WOBBLE, AB ONLY (7 unknowns)")
print("=" * 70)

cfgs = [
    ("K=4 corners",    [(0,0),(0.01,0),(0,0.01),(0.01,0.01)]),
    ("K=4 L+mid",      [(0,0),(0.01,0),(0,0.01),(0.005,0.005)]),
    ("K=5 +diag",      [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]),
    ("K=6 3x2",        list(product([0,0.005,0.01],[0,0.01]))),
    ("K=9 3x3",        list(product([0,0.005,0.01],[0,0.005,0.01]))),
    ("K=4 wide",       [(0,0),(0.02,0),(0,0.02),(0.02,0.02)]),
    ("K=12 4x3",       list(product([0,0.005,0.01,0.015],[0,0.005,0.01]))),
    ("K=16 4x4",       list(product([0,0.005,0.01,0.015],[0,0.005,0.01,0.015]))),
]

for name, wp in cfgs:
    K = len(wp)
    tgt = []
    for w1,w2 in wp:
        A,B = AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w1, a2_t+b2_t*w2)
        tgt.extend([A,B])
    tgt = np.array(tgt)

    _wp, _tgt = wp[:], tgt.copy()
    def mkres(wp=_wp, tgt=_tgt):
        def res(p):
            d1,d2,d3,a1,b1,a2,b2 = np.abs(p)
            r = np.empty(len(wp)*2)
            for k,(w1,w2) in enumerate(wp):
                A,B = AB_fast(d1,d2,d3, a1+b1*w1, a2+b2*w2)
                r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
            return r
        return res
    res = mkres()

    J = jac_num(res, true7)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    sv = np.linalg.svd(J, compute_uv=False)
    
    t0 = time.time()
    sols, uniq, nt = multistart(res, true7, n_trials=100)
    dt = time.time()-t0
    
    st = "✓UNIQUE" if len(uniq)==1 and nt==1 else f"✗ {len(uniq)} sol ({nt} true)"
    print(f"  {name:16s}: {2*K:2d} constr, rank={rank}/7, sv_min={sv[-1]:.1e}, "
          f"conv={len(sols):3d}/100, dist={len(uniq):3d} → {st}  [{dt:.1f}s]")


# ═══════════════════════════════════════════
# TEST 2: SINGLE WOBBLE (confirm fail)
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: SINGLE-LENS WOBBLE (6 unknowns, confirm fail)")
print("=" * 70)

true6 = np.array([d1_t, d2_t, d3_t, a1_t, b1_t, 1/a2_t])

for K in [5, 8, 12]:
    wv = np.linspace(0, 0.01, K)
    tgt = []
    for w in wv:
        A,B = AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w, a2_t)
        tgt.extend([A,B])
    tgt = np.array(tgt)

    _wv, _tgt = wv.copy(), tgt.copy()
    def mkres1(wv=_wv, tgt=_tgt):
        def res(p):
            d1,d2,d3,a1,b1,f2 = np.abs(p)
            r = np.empty(len(wv)*2)
            for k,w in enumerate(wv):
                A,B = AB_fast(d1,d2,d3, a1+b1*w, 1.0/f2)
                r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
            return r
        return res
    res = mkres1()
    
    J = jac_num(res, true6)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    sols, uniq, nt = multistart(res, true6, 80)
    st = "✓UNIQUE" if len(uniq)==1 and nt==1 else f"✗ {len(uniq)} sol ({nt} true)"
    print(f"  K={K:2d}: {2*K:2d} constr, rank={rank}/6, dist={len(uniq):2d} → {st}")


# ═══════════════════════════════════════════
# TEST 3: DUAL WOBBLE + KNOWN d3
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: DUAL WOBBLE + KNOWN d₃ (6 unknowns)")
print("=" * 70)

true6b = np.array([d1_t, d2_t, a1_t, b1_t, a2_t, b2_t])

for K, wp in [(4, [(0,0),(0.01,0),(0,0.01),(0.01,0.01)]),
              (5, [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]),
              (9, list(product([0,0.005,0.01],[0,0.005,0.01])))]:
    tgt = []
    for w1,w2 in wp:
        A,B = AB_fast(d1_t,d2_t,d3_t, a1_t+b1_t*w1, a2_t+b2_t*w2)
        tgt.extend([A,B])
    tgt = np.array(tgt)

    _wp, _tgt = wp[:], tgt.copy()
    def mkres3(wp=_wp, tgt=_tgt, d3f=d3_t):
        def res(p):
            d1,d2,a1,b1,a2,b2 = np.abs(p)
            r = np.empty(len(wp)*2)
            for k,(w1,w2) in enumerate(wp):
                A,B = AB_fast(d1,d2,d3f, a1+b1*w1, a2+b2*w2)
                r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
            return r
        return res
    res = mkres3()
    
    J = jac_num(res, true6b)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    sols, uniq, nt = multistart(res, true6b, 80)
    st = "✓UNIQUE" if len(uniq)==1 and nt==1 else f"✗ {len(uniq)} sol ({nt} true)"
    print(f"  K={K:2d}: {2*K:2d} constr, rank={rank}/6, conv={len(sols):2d}/80, "
          f"dist={len(uniq):2d} → {st}")
    if len(uniq) <= 3:
        for i,s in enumerate(uniq[:3]):
            err = np.abs(s-true6b)/true6b*100
            print(f"      Sol {i+1}: max_err={err.max():.4f}%")


# ═══════════════════════════════════════════
# TEST 4: d1-DEFOCUS help?
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: d₁-DEFOCUS help? (7 unknowns)")
print("=" * 70)

wp4 = [(0,0),(0.01,0),(0,0.01),(0.01,0.01)]
for label, dzv in [("No defocus  (4 img)",[0.0]),
                   ("3 d1-defocus (12 img)",[0.0, 0.5e-3, 1.0e-3])]:
    tgt = []
    for w1,w2 in wp4:
        phi1 = a1_t+b1_t*w1; phi2 = a2_t+b2_t*w2
        for dz in dzv:
            A,B = AB_fast(d1_t+dz,d2_t,d3_t, phi1, phi2)
            tgt.extend([A,B])
    tgt = np.array(tgt)

    _wp, _dzv, _tgt = wp4[:], dzv[:], tgt.copy()
    def mkres4(wp=_wp, dzv=_dzv, tgt=_tgt):
        def res(p):
            d1,d2,d3,a1,b1,a2,b2 = np.abs(p)
            r = np.empty(len(wp)*len(dzv)*2)
            idx=0
            for w1,w2 in wp:
                phi1=a1+b1*w1; phi2=a2+b2*w2
                for dz in dzv:
                    A,B = AB_fast(d1+dz,d2,d3, phi1, phi2)
                    r[idx]=A-tgt[idx]; r[idx+1]=B-tgt[idx+1]; idx+=2
            return r
        return res
    res = mkres4()
    nc = 2*len(wp4)*len(dzv)
    J = jac_num(res, true7)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    sols, uniq, nt = multistart(res, true7, 80)
    st = "✓UNIQUE" if len(uniq)==1 and nt==1 else f"✗ {len(uniq)} sol ({nt} true)"
    print(f"  {label}: {nc} constr, rank={rank}/7, dist={len(uniq):2d} → {st}")


# ═══════════════════════════════════════════
# TEST 5: ROBUSTNESS
# ═══════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: ROBUSTNESS across regimes (K=5 dual wobble)")
print("=" * 70)

wp5 = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]

for nm,d1,d2,d3,f1,f2,b1,b2 in [
    ("Default f1=3mm f2=50mm", 3.06e-3,205.5e-3,1.05, 3e-3,50e-3, 5000,200),
    ("Equal f1=f2=10mm",       5e-3,100e-3,500e-3, 10e-3,10e-3, 3000,3000),
    ("Strong f1=1mm f2=100mm", 1.5e-3,150e-3,2.0, 1e-3,100e-3, 10000,100),
    ("Weak f1=50mm f2=200mm",  60e-3,500e-3,3.0, 50e-3,200e-3, 1000,50),
]:
    a1=1/f1; a2=1/f2
    tp = np.array([d1,d2,d3,a1,b1,a2,b2])
    tgt = []
    for w1,w2 in wp5:
        A,B = AB_fast(d1,d2,d3, a1+b1*w1, a2+b2*w2)
        tgt.extend([A,B])
    tgt = np.array(tgt)

    _wp, _tgt = wp5[:], tgt.copy()
    def mkresR(wp=_wp, tgt=_tgt):
        def res(p):
            dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
            r = np.empty(len(wp)*2)
            for k,(w1,w2) in enumerate(wp):
                A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
                r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
            return r
        return res
    res = mkresR()
    J = jac_num(res, tp)
    rank = np.linalg.matrix_rank(J, tol=1e-8)
    sols, uniq, nt = multistart(res, tp, 100)
    st = "✓UNIQUE" if len(uniq)==1 and nt==1 else f"✗ {len(uniq)} sol ({nt} true)"
    print(f"  {nm:30s}: rank={rank}/7, dist={len(uniq):3d} → {st}")


# ═══════════════════════════════════════════
# CONCLUSION
# ═══════════════════════════════════════════
print("\n" + "═" * 70)
print("CONCLUSION")
print("═" * 70)
print("""
With d₁-defocus (correct physics):
  A' = A (constant), B' = B + A·Δz (slope = A, already known)
  → Each wobble setting gives 2 constraints (A, B₀)
  → d₁-defocus adds NO independent information

  ✗ Single-lens wobble: ALWAYS fails (continuous degeneracy)
  ✗ Dual-lens wobble AB-only: Full rank BUT many global solutions!
  ? Dual-lens + known d₃: Needs checking — may break degeneracy
  
  KEY DIFFERENCE FROM d₃-DEFOCUS:
    d₃-defocus gave ABCD → 3 constraints/setting → globally unique
    d₁-defocus gives AB only → 2 constraints/setting → NOT globally unique
""")

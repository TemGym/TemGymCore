#!/usr/bin/env python3
"""Test if knowing d2+d3 breaks the degeneracy."""
import numpy as np
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')

def AB_fast(d1, d2, d3, phi1, phi2):
    A = (1.0 - d2 * phi1) * (1.0 - d3 * phi2) - d3 * phi1
    B = d1 * A + d2 * (1.0 - d3 * phi2) + d3
    return A, B

d1_t,d2_t,d3_t = 3.06e-3, 205.5e-3, 1.05
a1_t,a2_t = 1/3e-3, 1/50e-3
b1_t,b2_t = 5000.0, 200.0
tp = np.array([d1_t,d2_t,d3_t,a1_t,b1_t,a2_t,b2_t])

wp = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]
tgt = np.array([v for w1,w2 in wp for v in AB_fast(d1_t,d2_t,d3_t,a1_t+b1_t*w1,a2_t+b2_t*w2)])

def residual(p):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp)*2)
    for k,(w1,w2) in enumerate(wp):
        A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    return r

total_known = d2_t + d3_t

def res_total(p):
    dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
    r = np.empty(len(wp)*2 + 1)
    for k,(w1,w2) in enumerate(wp):
        A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
        r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
    r[-1] = 1e6 * (dd2 + dd3 - total_known)
    return r

print(f"True d2+d3 = {total_known*1e3:.1f} mm")
sols = []
for trial in range(500):
    np.random.seed(trial)
    x0 = tp * np.exp(np.random.randn(7) * 0.3)
    try:
        r = least_squares(res_total, x0, method='lm', max_nfev=10000,
                         ftol=1e-15, xtol=1e-15)
        if r.cost < 1e-10:
            s = np.abs(r.x)
            r_check = residual(s)
            if np.max(np.abs(r_check)) < 1e-9 and abs(s[1]+s[2]-total_known) < 1e-6:
                sols.append(s)
    except: pass

uniq = []
for s in sols:
    if all(not np.allclose(s, u, rtol=0.005) for u in uniq):
        uniq.append(s)
nt = sum(1 for s in uniq if np.allclose(s, tp, rtol=0.01))
print(f"Known d2+d3: {len(sols)} conv, {len(uniq)} distinct, {nt} true")
if len(uniq) == 1 and nt == 1:
    print("UNIQUE SOLUTION!")
else:
    print(f"{len(uniq)} solutions remain:")
    for s in uniq[:8]:
        err = np.max(np.abs(s - tp)/tp)*100
        print(f"  d2={s[1]*1e3:.1f}mm d3={s[2]*1e3:.1f}mm f2={1e3/s[5]:.1f}mm max_err={err:.1f}%")

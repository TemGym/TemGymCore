#!/usr/bin/env python3
"""Quick bounds test - writes results to file."""
import numpy as np
from scipy.optimize import least_squares
import warnings, sys
warnings.filterwarnings('ignore')

def AB_fast(d1, d2, d3, phi1, phi2):
    A = (1.0 - d2 * phi1) * (1.0 - d3 * phi2) - d3 * phi1
    B = d1 * A + d2 * (1.0 - d3 * phi2) + d3
    return A, B

wp5 = [(0,0),(0.01,0),(0,0.01),(0.005,0.005),(0.01,0.01)]

out = []
regimes = [
    ('Default', 3.06e-3,205.5e-3,1.05, 3e-3,50e-3, 5000,200),
    ('Equal',   5e-3,100e-3,500e-3, 10e-3,10e-3, 3000,3000),
    ('Strong',  1.5e-3,150e-3,2.0, 1e-3,100e-3, 10000,100),
    ('Weak',    60e-3,500e-3,3.0, 50e-3,200e-3, 1000,50),
]

for nm,d1,d2,d3,f1,f2,b1,b2 in regimes:
    a1=1/f1; a2=1/f2
    tp = np.array([d1,d2,d3,a1,b1,a2,b2])
    tgt = np.array([v for w1,w2 in wp5 for v in AB_fast(d1,d2,d3,a1+b1*w1,a2+b2*w2)])
    
    def res(p, tgt=tgt, wp=wp5):
        dd1,dd2,dd3,aa1,bb1,aa2,bb2 = np.abs(p)
        r = np.empty(len(wp)*2)
        for k,(w1,w2) in enumerate(wp):
            A,B = AB_fast(dd1,dd2,dd3, aa1+bb1*w1, aa2+bb2*w2)
            r[2*k]=A-tgt[2*k]; r[2*k+1]=B-tgt[2*k+1]
        return r
    
    lb = tp / 3.0; ub = tp * 3.0
    sols = []
    for t in range(200):
        np.random.seed(t)
        x0 = lb + np.random.rand(7) * (ub - lb)
        try:
            r = least_squares(res, x0, method='trf', max_nfev=5000,
                             ftol=1e-14, xtol=1e-14, bounds=(lb, ub))
            if r.cost < 1e-13: sols.append(np.abs(r.x))
        except: pass
    uniq = []
    for s in sols:
        if all(not np.allclose(s,u,rtol=0.005) for u in uniq): uniq.append(s)
    nt = sum(1 for s in uniq if np.allclose(s,tp,rtol=0.01))
    st = 'UNIQUE' if len(uniq)==1 and nt==1 else f'{len(uniq)} sol ({nt} true)'
    line = f'{nm:8s}: 3x bounds -> {len(sols):3d} conv, {len(uniq):3d} distinct -> {st}'
    out.append(line)
    sys.stdout.write(line + '\n')
    sys.stdout.flush()

with open('/home/dl277493/Documents/Code/TemGymCore/examples/lens_inversion/bounds_results.txt', 'w') as f:
    f.write('\n'.join(out))
print('\nDone! Results written.')

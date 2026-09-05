"""Retry the frozen image likelihood with descent-safe curvature steps.

The rough-geometry source initializer uses the enlarged training data. Its
optical coincidence with the generator is explicitly conditional; offset
geometry starts are retained and must succeed independently for that claim.
"""
from pathlib import Path
import argparse
import numpy as np

from neoarm_condenser_extension_recovery import run
from temgym_core.feasible_least_squares import svd_gauss_newton

NATIVE_SCALES=np.r_[np.tile([.02,.1,.05,.1],3),
    [.1,1.,.1,1.,.1,.1,.1,1.,.1],.01,.005]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-iterations',type=int,default=250)
    args=parser.parse_args()
    run(args.inference,args.output,max_iterations=args.max_iterations,
        solver=svd_gauss_newton,solver_options=dict(native_scales=NATIVE_SCALES),
        include_rough_data_start=True)

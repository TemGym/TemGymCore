# Lens Inversion Analysis

This folder contains the core analysis for inverting TEM lens parameters from intensity measurements.

## Active Files

### Theory & Documentation
- **lens_inversion.md** - Main technical document explaining the inverse problem, degeneracy analysis, and solution strategies using physical constraints (nonlinear φ∝I² model, image rotation, dual voltages)
- **SOLVABILITY_SUMMARY.md** - Summary of previous numerical validation work

### Notebooks
- **two_lenses_simplified.ipynb** - Starting point for 2-lens system fitting with wobble measurements
- **single_lens.ipynb** - Reference implementation for single-lens parameter fitting

## Archive

**stashed/** - Contains previous analysis files, test scripts, and alternative implementations that were used to validate the three solution strategies (known total distance, nonlinear model, two voltages). These files document the exploration phase and numerical validation but are not needed for the current 2-lens implementation.

## Next Steps

The goal is to implement a 2-lens fitting procedure that uses:
1. **Nonlinear optical power model**: φ(w) = φ₀(1+w)² enforcing the constraint c = b²/(4a)
2. **Image rotation measurements**: θ(w) = r(1+w) providing linear current dependence
3. **Wobble diversity**: K≥5 current wobble settings

This combination should uniquely determine all 2N+1 parameters (N+1 distances + N optical powers) for N=2 lenses using only one accelerating voltage.

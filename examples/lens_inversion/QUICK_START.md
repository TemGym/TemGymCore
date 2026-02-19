# Quick Start Guide: Enhanced Microscope Inversion

## What This Is

An enhanced Jupyter notebook that solves the **inverse problem** of determining microscope optical parameters (distances and focal lengths) from manufacturer's magnification calibration data.

## What's New

**CRITICAL FIX:** The notebook now correctly starts from the **sample plane** instead of the OL image plane, properly including the objective lens in the optical model.

## Installation

```bash
# Clone repository (if not already done)
git clone https://github.com/TemGym/TemGymCore.git
cd TemGymCore

# Install dependencies
pip install -e .
pip install jupyter scipy matplotlib

# Navigate to notebook
cd examples/lens_inversion
```

## Running the Analysis

```bash
# Start Jupyter
jupyter notebook dac_lens_inversion_enhanced.ipynb
```

Then:
1. **Run All Cells** (Cell → Run All)
2. Wait ~15 minutes
3. Scroll to bottom for final assessment

## What You'll See

The notebook will run **5 optimization methods** and tell you one of:

### ✓ Problem is SOLVABLE
```
✓ PROBLEM IS SOLVABLE
  Best cost: 2.3e-6
  Solution appears unique (max CV = 0.03)
```
**Meaning:** Success! All parameters can be determined.

### ⚠ Multiple Solutions
```
✓ PROBLEM IS SOLVABLE
  Best cost: 5.1e-5
⚠ MULTIPLE DISTINCT SOLUTIONS (CV = 0.34)
```
**Meaning:** Partial success. Need one additional measurement to make unique.

### ⚠ Partially Solvable
```
⚠ PROBLEM IS PARTIALLY SOLVABLE
  Best cost: 3.2e-3
```
**Meaning:** Approximate solution found. Model may need refinement.

### ✗ Not Solvable
```
✗ PROBLEM IS NOT SOLVABLE
  No optimizers converged.
```
**Meaning:** Fundamental issue. Need independent measurements or model revision.

## Understanding Results

### Key Parameters Determined

If successful, you'll get:

**Distances (mm):**
- `d_obj`: Sample to objective lens (1-10 mm)
- `d0-d4`: Inter-lens distances (10-300 mm)

**Focal Length Coefficients:**
- `Cf_OL, Cf_IL1, Cf_IL2, Cf_IL3, Cf_PL1`
- Used in: `f = 1/(Cf·DAC²)`

**Validation Metrics:**
- Magnification RMS error (should be < 1%)
- Focus RMS error (should be < 1e-3)

## Optical Path

The enhanced model propagates light through:

```
Sample → [d_obj] → OL → [d0] → IL1 → [d1] → IL2 → [d2] → IL3 → [d3] → PL1 → [d4] → Detector
```

This is the **correct** path that matches the physical microscope.

## Optimization Methods Used

1. **Levenberg-Marquardt** - Trust region with Jacobian
2. **Trust-Region Reflective** - Bounded optimization
3. **Multi-Start L-BFGS-B** - 50 random initializations
4. **Differential Evolution** - Global search
5. **Latin Hypercube Sampling** - Bayesian exploration

## Files in This Directory

- `dac_lens_inversion_enhanced.ipynb` - **Main notebook (run this)**
- `README_ENHANCED.md` - Detailed documentation
- `ANALYSIS_SUMMARY.md` - Comprehensive analysis guide
- `QUICK_START.md` - This file
- `dac_lens_inversion.ipynb` - Original (for reference only)

## Common Issues

### "No module named 'jax'"
```bash
pip install jax jaxlib
```

### "No module named 'sympy'"
```bash
pip install sympy
```

### Notebook takes too long
- Normal runtime: ~15 minutes
- You can reduce iterations in optimization cells
- Or run only one method (e.g., just Levenberg-Marquardt)

### Results look wrong
- Check that you're using Mode C data (most constrained)
- Verify DAC values are correct
- Ensure physical bounds make sense for your microscope

## Next Steps After Running

### If Unique Solution Found (CV < 0.1)
✓ Extract parameters from best solution
✓ Validate with round-trip test
✓ Use for microscope control/simulation

### If Multiple Solutions (CV > 0.1)
1. Identify which parameters are uncertain (high CV)
2. Measure one of them independently
3. Add as constraint and re-run
4. Should converge to unique solution

### If Partially Solvable
1. Check model assumptions (thin lens valid?)
2. Verify DAC calibration data
3. Consider aberration terms
4. Try other operating modes (B, D)

### If Not Solvable
1. Verify optical diagram is correct
2. Get independent measurements of 2-3 parameters
3. Check if additional optical elements exist
4. Consider thick lens or aberration models

## Getting Help

For questions:
1. Read `ANALYSIS_SUMMARY.md` for detailed scenarios
2. Read `README_ENHANCED.md` for theory and references
3. Open issue on GitHub: https://github.com/TemGym/TemGymCore/issues

## Citation

If you use this enhanced inversion method, please cite:
- TemGymCore: https://github.com/TemGym/TemGymCore
- Original DAC inversion concept: [Your lab/paper]
- Enhanced implementation: This PR

## Technical Summary

**Model:**
- 5 lenses (OL, IL1, IL2, IL3, PL1)
- 6 distances (d_obj, d0-d4)
- 11 unknowns total

**Constraints:**
- 14 magnifications (Mode C)
- 14 focus conditions (B=0)
- 28 equations total
- Overdetermined 2.5:1

**Convergence:**
- Tolerances: 1e-12 (ftol, xtol, gtol)
- Max iterations: 1000-5000
- Success criterion: cost < 1e-4

---

**Ready to start?** Open the notebook and **Run All Cells**!

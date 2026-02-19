# Enhanced Microscope Inversion Analysis

## Overview

This directory contains an **enhanced version** of the DAC-based lens column inversion notebook that addresses the key limitation of the original implementation: **starting from the sample plane instead of the OL image plane**.

## Key Changes

### 1. **Corrected Optical Path**

**Original (INCORRECT):**
```
OL image → d0 → IL1 → d1 → IL2 → d2 → IL3 → d3 → PL1 → d4 → detector
```

**Enhanced (CORRECT):**
```
Sample → d_obj → OL → d0 → IL1 → d1 → IL2 → d2 → IL3 → d3 → PL1 → d4 → detector
```

The enhanced model:
- Starts at the **sample plane** (physical specimen)
- Includes **distance d_obj** from sample to objective lens
- Explicitly models the **objective lens (OL)** with variable focal length
- Continues through the projection system (IL1, IL2, IL3, PL1)
- Ends at the detector plane

### 2. **Comprehensive Optimization Strategies**

The enhanced notebook implements **5 different optimization methods** to exhaustively explore the solution space:

1. **Levenberg-Marquardt (LM)**
   - Trust-region method with Jacobian
   - Good for well-posed problems
   - Unbounded optimization

2. **Trust-Region Reflective (TRF)**
   - Bounded optimization with physical constraints
   - Enforces positive distances and realistic focal lengths
   - More robust than LM for ill-conditioned problems

3. **Multi-Start L-BFGS-B**
   - 50-100 random initializations
   - Gradient-based with bounds
   - Explores multiple local minima

4. **Differential Evolution (DE)**
   - Global optimization using evolutionary algorithm
   - Population-based search
   - Good for multi-modal landscapes

5. **Latin Hypercube Sampling (LHS)**
   - Bayesian exploration of parameter space
   - Quasi-random sampling for efficient coverage
   - Identifies feasible regions

### 3. **Tolerance and Convergence Control**

All optimizers support ultra-low tolerances:
- `ftol = 1e-12` (function tolerance)
- `xtol = 1e-12` (parameter tolerance)  
- `gtol = 1e-12` (gradient tolerance)

### 4. **Solution Multiplicity Analysis**

The notebook includes comprehensive analysis to determine:
- **Is there a solution?** → Minimum cost achieved
- **Is it unique?** → Coefficient of variation across solutions
- **How many solutions exist?** → Clustering analysis
- **What constraints are needed?** → Recommendations for additional measurements

## Model Parameters

### Optical System
- **5 lenses:** OL, IL1, IL2, IL3, PL1
- **6 distances:** d_obj, d0, d1, d2, d3, d4
- **Total unknowns:** 11 parameters

### Physical Bounds
```python
# Distances (meters)
d_obj:  0.001 - 0.010  # Sample to OL: 1-10 mm
d0-d3:  0.010 - 0.300  # Inter-lens: 10-300 mm
d4:     0.050 - 0.500  # Final leg: 50-500 mm

# Focal length coefficients
Cf:     1e-9 - 1e-5    # Wide range for robustness
```

### Constraints
1. **Focus condition:** B = M[0,1] ≈ 0 (image in focus at detector)
2. **Magnification match:** M[0,0] = target magnification
3. **Rotation condition:** Σ K_i * DAC_i = 0 (zero net rotation)
4. **Physical realizability:** All distances and focal lengths positive

## Usage

### Running the Enhanced Notebook

```bash
# Install dependencies
pip install jupyter scipy matplotlib jax sympy

# Install TemGymCore
pip install -e .

# Launch Jupyter
jupyter notebook examples/lens_inversion/dac_lens_inversion_enhanced.ipynb
```

### Running Optimization

The notebook is organized into cells that:
1. Load DAC calibration data (Mode A, B, C, D)
2. Define the enhanced forward model
3. Run each optimization method sequentially
4. Compare and analyze all results
5. Generate comprehensive visualizations
6. Provide solvability assessment

**Expected runtime:**
- Levenberg-Marquardt: ~10 seconds
- Trust-Region: ~15 seconds
- Differential Evolution: ~5 minutes (500 iterations)
- Multi-Start L-BFGS-B: ~2 minutes (50 starts)
- Latin Hypercube: ~5 minutes (1000 samples)

**Total:** ~15 minutes for complete analysis

## Interpreting Results

### Success Criteria

A solution is considered **successful** if:
1. Optimizer converges (`success = True`)
2. Final cost < 1e-4
3. Magnification RMS error < 1%
4. Focus RMS error (B values) < 1e-3

### Solution Uniqueness

Solutions are considered **distinct** if:
- Coefficient of variation (CV) > 0.1 for any parameter
- Multiple local minima with similar cost

Solutions are **unique** if:
- CV < 0.1 for all parameters
- All optimizers converge to same solution

### Solvability Assessment

The notebook provides a **definitive statement** on solvability:

**✓ SOLVABLE:** 
- Multiple methods converge to low-cost solution
- Magnification and focus constraints satisfied
- May have unique or multiple solutions

**⚠ PARTIALLY SOLVABLE:**
- Some methods converge but not to convergence threshold
- Indicates model mismatch or insufficient constraints
- Recommendations provided for improvement

**✗ NOT SOLVABLE:**
- No method converges successfully
- Fundamental issue with model or data
- Requires additional measurements or model revision

## Key Insights

### Why Starting Point Matters

Starting at the **OL image plane** assumes the objective lens behavior is already known and creates a fixed magnification block. Starting at the **sample plane** allows the optimizer to:
1. Solve for OL focal length from OLf DAC values
2. Account for sample-to-OL distance
3. Better constrain the full optical path
4. Match the physical microscope geometry

### Objective Lens Role

The objective lens (OL) is critical because:
- It provides the primary magnification (typically 20-200×)
- Its focal length changes with OLf DAC (Mode A vs C vs D)
- Sample-to-OL distance determines field of view
- OL post-field image becomes input to projection system

### Mode C Analysis

Mode C (30kx - 600kx magnifications) is optimal because:
- OLf is constant (39178) → OL focal length fixed
- PL1 is constant (64000) → PL1 focal length fixed
- Only IL1, IL2, IL3 vary → 3 degrees of freedom
- 14 settings → 28 equations (mag + focus)
- Heavily overdetermined system (28 equations, 11 unknowns)

## Recommendations

### If Problem is NOT Solvable

To make the problem solvable, obtain:
1. **Direct measurement of one distance** (e.g., total column length)
2. **Measurement of one focal length** (e.g., calibrate IL1 at one DAC)
3. **Additional data** from other operating modes (A, B, D)
4. **Independent verification** of DAC→current→focal length relationship
5. **Thick lens model** if thin-lens approximation insufficient

### If Multiple Solutions Exist

To achieve uniqueness, add constraints:
1. Fix one distance from mechanical drawings
2. Fix one Cf coefficient from lens calibration
3. Use ratio constraints (e.g., d1/d2 from symmetry)
4. Add aberration measurements (chromatic, spherical)
5. Use multiple detector positions (not just focus)

## Files

- `dac_lens_inversion.ipynb` - Original notebook (starts at OL image)
- `dac_lens_inversion_enhanced.ipynb` - Enhanced notebook (starts at sample)
- `README_ENHANCED.md` - This file
- `n_lens_forward.ipynb` - General n-lens forward model
- `n_lens_inversion.ipynb` - General n-lens inverse problem

## References

### Theory
- **Gaussian optics:** Thin lens ABCD matrices
- **Transfer matrices:** Paraxial ray tracing
- **Inverse problems:** Nonlinear least squares

### Optimization
- Levenberg-Marquardt: Moré (1978) "The Levenberg-Marquardt algorithm"
- L-BFGS-B: Byrd et al. (1995) "A limited memory algorithm"
- Differential Evolution: Storn & Price (1997) "Differential evolution"
- Latin Hypercube: McKay et al. (1979) "A comparison of three methods"

## Contributing

To extend this work:
1. Add aberration terms (Cs, Cc) to lens matrices
2. Include aperture effects (vignetting, diffraction)
3. Model thick lenses for more realistic OL
4. Add chromatic effects for energy spread
5. Include magnetic field rotation (螺旋 effect)

## Contact

For questions about the enhanced implementation, please open an issue on the TemGymCore repository.

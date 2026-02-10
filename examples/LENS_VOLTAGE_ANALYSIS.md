# Multi-Lens System Analysis with Voltage Variation

## Overview

This notebook investigates whether changing the electron beam voltage (200 keV ↔ 210 keV) helps fit 4 and 5 lens systems in transmission electron microscopy (TEM).

## Files Created

- **`examples/n_lens_forward.ipynb`**: Interactive Jupyter notebook for multi-lens system analysis
- **`src/temgym_core/utils.py`**: Added `electron_wavelength()` function for voltage-to-wavelength conversion
- **`tests/test_electron_wavelength.py`**: Comprehensive tests for the wavelength calculation

## Key Findings

### Question
*"Would changing voltage from 200 keV to 210 keV make it possible to fit a 4 and 5 lens system?"*

### Answer
**NO** - Voltage variation is NOT necessary to fit 4 and 5 lens systems.

### Reasoning

#### Degrees of Freedom Analysis

| System | DOF (no voltage) | DOF (with voltage) | Constraints | Extra DOF |
|--------|------------------|-------------------|-------------|-----------|
| 2-Lens | 4 | 5 | ~4 | 0 → 1 |
| 3-Lens | 6 | 7 | ~4 | 2 → 3 |
| 4-Lens | **8** | **9** | ~4 | **4 → 5** |
| 5-Lens | **10** | **11** | ~4 | **6 → 7** |

**Key Points:**
1. **4-Lens System**: 8 DOF vs. 4 constraints → **4 extra DOF**
2. **5-Lens System**: 10 DOF vs. 4 constraints → **6 extra DOF**
3. Both systems are already **over-determined** without voltage variation
4. Voltage adds only **1 DOF** (equivalent to global focal length scaling)
5. When you have 4-6 extra DOF, adding 1 more provides **minimal benefit**

#### Wavelength Analysis

```
200 keV → λ = 2.3249 pm
210 keV → λ = 2.2531 pm
Relative change: 3.09%
```

In paraxial ray tracing:
- ABCD matrices are wavelength-independent
- Voltage affects lens excitation → changes focal lengths
- Voltage variation ≈ additional focal length tuning range

## When Voltage Variation WOULD Help

1. **Fixed lens currents**: If focal lengths are constrained (e.g., magnetic saturation)
2. **Aberration correction**: In non-paraxial models with wavelength-dependent chromatic aberration
3. **2-Lens systems**: With only 4 DOF, adding voltage (→ 5 DOF) provides meaningful flexibility
4. **Chromatic aberration**: When color correction is needed

## Practical Implications

✓ **Focus on optimizing focal lengths and distances first**
✓ Use standard voltage (e.g., 200 keV) for 4-5 lens systems
✓ Reserve voltage variation for fine-tuning or special cases
✓ Multi-lens systems provide ample flexibility WITHOUT voltage tuning

## Usage

### Running the Notebook

```bash
cd examples
jupyter notebook n_lens_forward.ipynb
```

### Using the Wavelength Function

```python
from temgym_core.utils import electron_wavelength

# Calculate wavelength at 200 keV
wavelength = electron_wavelength(200.0)
print(f"Wavelength: {wavelength*1e12:.4f} pm")  # Output: 2.3249 pm
```

### Testing

```bash
# Run all tests
pytest tests/test_electron_wavelength.py -v

# Run specific test
pytest tests/test_electron_wavelength.py::test_electron_wavelength_200kev -v
```

## Technical Details

### Degrees of Freedom (DOF) Calculation

For an N-lens system:
- **Focal lengths**: N parameters
- **Distances**: N+1 distances, but 1 constraint (total length) → N free parameters
- **Total DOF**: 2N

With voltage variation:
- **Total DOF**: 2N + 1

### Typical Constraints

1. Magnification in x-direction
2. Magnification in y-direction (usually equals x)
3. Imaging condition (B matrix ≈ 0 or specific value)
4. Focus condition (detector at image plane)

**Total**: ~4 constraints

### ABCD Matrix Structure

5×5 homogeneous matrix:
```
[A_xx  A_xy  B_xx  B_xy  offset_x  ]
[A_yx  A_yy  B_yx  B_yy  offset_y  ]
[C_xx  C_xy  D_xx  D_xy  slope_x   ]
[C_yx  C_yy  D_yx  D_yy  slope_y   ]
[0     0     0     0     1         ]
```

Where:
- **A**: Magnification block
- **B**: Imaging/focus block
- **C**: Angular magnification block
- **D**: Slope transformation block

## References

- Relativistic electron wavelength: λ = h / √(2m_e·e·V·(1 + eV/(2m_e·c²)))
- Physical constants from CODATA 2018
- Paraxial ray tracing using JAX for automatic differentiation

## Contributing

This analysis uses the TemGymCore differentiable ray tracing framework. For questions or contributions, please open an issue or pull request.

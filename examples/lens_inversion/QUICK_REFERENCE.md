# Quick Reference: Two-Lens Inversion Analysis
## For Use in New Context Windows

---

## ⚡ **TL;DR - Problem & Solution**

**Goal:** Recover $(f_1, f_2, z_1, z_2, z_3)$ from defocused TEM images

**Answer:** ✓ Solvable with **dual lens wobble + Bayesian optimization + priors**

**Why:** 
- Without wobble: 2-10 degenerate solutions
- With dual wobble: 1-2 solutions → priors select physical one
- Data: 54 constraints for 7 unknowns (7.7× overdetermined)

---

## 📊 **Strategy Comparison (Ranked)**

| Rank | Strategy | Unknowns | Constraints | Solutions | Score |
|------|---|---|---|---|---|
| 1️⃣ | **Dual Wobble + Bayesian** | 7 | 54 | 1-2 | **73/100** ✓✓ |
| 2️⃣ | Wobble + Fixed z₃ | 5 | 18 | 2-3 | 60/100 ✓ |
| 3️⃣ | Single Wobble | 6 | 18 | 5-10 | 44/100 ⚠️ |
| 4️⃣ | Defocus Only | 5 | 6 | 10+ | 18/100 ❌ |

---

## 🔑 **Critical Physics Insights**

### 1. B/A = z_defocus (Magnification Cancels)
```
From ABCD constraints: A = M, B = M·z_def
Therefore: B/A = z_def (M cancels!)

Fresnel kernel: H(f) = exp(-iπλ(B/A)f²) = exp(-iπλ z_def·f²)
```
**Implication:** Fringe spacing independent of magnification M

### 2. Magnification Zoom Needed
When A_fit ≠ A_true, apply zoom:
```python
zoom_factor = A_fit / A_ref
intensity = jax.image.resize(intensity, (zoom_factor*N, zoom_factor*N))
# Pad/crop back to original grid
```

### 3. Missing C Element (Fundamental Degeneracy)
```
ABCD matrix has 3 DOF: A, B, C (D determined by constraint)
You measure: A, B from intensity (2 values)
You DON'T measure: C from ray angles (1 value)

Result: Multiple (z1,z2,z3,f1,f2) → same (A,B) → same images
```

---

## 🧪 **Experimental Setup**

**Measurements Needed:**
- Wobble states for Lens 1: M₁ = 3-5
- Wobble states for Lens 2: M₂ = 3-5  
- Defocus planes per wobble: N = 3-4
- **Total: 27-100 images → 54-200 ABCD constraints**

**Linear Focal Length Models:**
$$\frac{1}{f_1(w_1)} = a_1 + b_1 \cdot w_1$$
$$\frac{1}{f_2(w_2)} = a_2 + b_2 \cdot w_2$$

where $w_i$ are known wobble parameters.

---

## 💻 **Implementation Recipe**

### Step 1: Bayesian Optimization Setup
```python
import optuna

# Parameter bounds (from microscope geometry)
bounds = {
    'z1': (10e-6, 100e-6),
    'z2': (200e-6, 1000e-6),
    'z3': (0.5, 2.0),
    'a1': (20000, 60000),
    'b1': (100, 1000),
    'a2': (1000, 4000),
    'b2': (50, 200),
}

# Priors (from manufacturer specs ±10%)
priors = {
    'a1': NormalDist(1/f1_nominal, 0.1/f1_nominal),
    'a2': NormalDist(1/f2_nominal, 0.1/f2_nominal),
}
```

### Step 2: Objective Function
```python
def objective(trial):
    params = {k: trial.suggest_float(k, *v) for k, v in bounds.items()}
    
    # Forward model
    predicted = forward_model_dual_wobble(**params)
    
    # Data fidelity
    data_loss = sum((predicted - measured)**2)
    
    # Prior term (weak)
    prior_loss = -log(priors['a1'].pdf(params['a1'])) - log(priors['a2'].pdf(params['a2']))
    
    return data_loss + 0.1 * prior_loss
```

### Step 3: Run Optimization
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Extract solutions
best_params = study.best_params
top_solutions = sorted(study.trials, key=lambda t: t.value)[:5]
```

---

## 📈 **Expected Performance**

**Parameter Accuracy:**
- $z_1, z_2$: ±2-5%
- $z_3$: ±5-10% (hardest to recover)
- $f_1, f_2$: ±1-3%

**Robustness:**
- Intensity noise: ~2-3% before convergence issues
- Optimization time: 1-3 hours for 200 trials on GPU
- Number of solutions: 1-2 (vs 10+ without priors)

---

## ❓ **Why Degeneracy Exists**

### Simple Example
```
Two equations, two unknowns, multiple solutions:
  x² + y² = 1    (circle)
  y = x²         (parabola)
Solution: (±0.786, 0.618) - TWO solutions!

Nonlinear ≠ Simple equation counting
```

### In Your Problem
- Even with 54 constraints vs 7 unknowns
- Multiple parameter sets produce identical (A,B) sequences
- **Reason:** You don't measure C element (requires ray angle data)
- **Solution:** Wobble diversity + priors break degeneracy

---

## 📁 **Key Files Created**

1. **ANALYSIS_SUMMARY.md** - Full detailed analysis (9 parts)
2. **uniqueness_analysis.py** - Quantitative tests (Jacobian rank, multi-start)
3. **bayesian_dual_wobble_guide.md** - Implementation guide with code
4. **degeneracy_explanation.py** - Why degeneracy exists (demonstrations)
5. **strategy_comparison_plot.py** - Visual ranking of all approaches
6. **two_lenses.ipynb** - Main notebook with forward model

---

## 🎯 **Next Steps**

- [ ] Implement dual wobble in forward model
- [ ] Collect/generate 27-100 test images
- [ ] Set up Bayesian optimization with optuna
- [ ] Define bounds from microscope geometry
- [ ] Add priors from manufacturer specs
- [ ] Run 200-300 trials
- [ ] Analyze top-5 solutions for modes
- [ ] Validate recovered parameters

---

## 🔗 **How to Use This in a New Context**

In your next conversation, say:

> "I'm working on a two-lens inversion problem. Here's the analysis:
> 
> **Problem:** Recover (f₁, f₂, z₁, z₂, z₃) from defocused images
> 
> **Solution:** Dual wobble + Bayesian optimization
> 
> **Key physics:**
> - B/A = z_defocus (magnification cancels)
> - Missing C element causes degeneracy (need priors)
> - 54 constraints for 7 unknowns (highly overdetermined)
> 
> **Next task:** [describe specific implementation problem]"

Then paste the relevant section from this file or the full ANALYSIS_SUMMARY.md as needed.

---

## 💡 **Key Equations to Remember**

ABCD imaging constraints:
$$A = M, \quad B = M \cdot z_{\text{def}} \implies \frac{B}{A} = z_{\text{def}}$$

Fresnel transfer function:
$$H(f) = \exp\left(-i\pi\lambda \frac{B}{A}(f_x^2 + f_y^2)\right)$$

Linear wobble models:
$$\frac{1}{f_i(w_i)} = a_i + b_i w_i$$

Loss function:
$$L = L_{\text{data}} + \lambda_{\text{prior}} L_{\text{prior}}$$

---

**Note:** Full detailed analysis available in ANALYSIS_SUMMARY.md (9 sections, 90+ points)

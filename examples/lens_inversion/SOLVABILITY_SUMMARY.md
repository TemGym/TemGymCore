# Two-Lens Inverse Problem: Solvability Analysis

## Problem Statement

Given a two-lens electron-optical column with transfer matrix

$$M = P(d_3) \cdot L(f_2) \cdot P(d_2) \cdot L(f_1) \cdot P(d_1)$$

we measure only the **A** and **B** matrix elements (image magnification and
position sensitivity) for multiple wobble settings $(w_1, w_2)$ that
perturb the lens excitations:

$$\phi_i(w) = a_i + b_i \cdot w_i$$

**Goal:** Recover the 7 unknowns $(d_1, d_2, d_3, a_1, b_1, a_2, b_2)$ from
A and B measurements alone — where $a_i = 1/f_i$ is the optical power and
$b_i$ is the wobble sensitivity of lens $i$.

## The Fundamental Degeneracy

### What goes wrong with A,B only (linear model)

The analytical formulas are:

$$A = (1 - d_2 \phi_1)(1 - d_3 \phi_2) - d_3 \phi_1$$

$$B = d_1 \cdot A + d_2(1 - d_3 \phi_2) + d_3$$

From these, A **does not depend on** $d_1$ at all, and B depends on $d_1$
only through $d_1 \cdot A$, which is already measured. This means:

- $d_1$ and $b_1$ are **always exactly recovered** from any measurement set
- $a_1$ ($\approx 1/f_1$) is **nearly exactly recovered**
- But $d_2$, $d_3$, $a_2$, and $b_2$ have a **continuous 1-parameter degeneracy**

### The degeneracy manifold

The spurious solutions form a 1D curve with perfect correlations:

| Parameter pair | Correlation in log-space |
|:---------------|:------------------------|
| $d_2 \leftrightarrow d_3$ | $r = +1.000$ (scale together) |
| $d_2 \leftrightarrow \phi_2$ | $r = -1.000$ (anti-correlated) |
| $d_3 \leftrightarrow \phi_2$ | $r = -1.000$ (anti-correlated) |
| $d_2 \leftrightarrow b_2$ | $r = -1.000$ (anti-correlated) |

Physically: you can **scale** $d_2$ and $d_3$ by a common factor $\lambda$
while simultaneously scaling $\phi_2$ and $b_2$ by a compensating factor,
and get exactly the same A,B values for all wobble settings.

The ratio $d_2/d_3$ is perfectly preserved (constant = 0.1957 across all
solutions), but the absolute scale is unconstrained.

### What does NOT help

| Strategy | Outcome |
|:---------|:--------|
| More wobble measurements ($K = 4 \to 16$) | 70–78 solutions, never unique |
| d₁-defocus series | Adds zero information ($A' = A$, $B' = B + A\Delta z$) |
| Changing operating point (same geometry) | All solutions survive (degeneracy is geometric) |
| Tighter parameter bounds (even ±50%) | Still 72 distinct solutions |
| Bayesian inference | Cannot break exact non-identifiability — posterior is a ridge |
| Predicting out-of-sample wobble | All solutions predict **identically** for any $(w_1, w_2)$ |

### What DOES help: three practical strategies

All three strategies below give a **globally unique solution**, verified
across 4 parameter regimes and 300–500 random starting points each.

---

## Strategy 1: Known Total Specimen Distance ($d_1 + d_2 + d_3$)

### Concept

The total physical distance from sample plane to detector is often known
from microscope design specifications or can be measured directly (e.g.,
with a ruler on the column). Since the degeneracy scales $d_2$ and $d_3$
together ($d_2' = \lambda d_2$, $d_3' = \lambda d_3$), fixing their sum
constrains $\lambda = 1$.

Note: since $d_1$ is always perfectly recovered from the data anyway,
knowing $d_1 + d_2 + d_3$ is equivalent to knowing $d_2 + d_3$.

### Implementation

Add one constraint to the least-squares problem:

$$r_{\text{extra}} = \lambda_{\text{pen}} \cdot \bigl(d_1 + d_2 + d_3 - L_{\text{total}}\bigr)$$

where $L_{\text{total}}$ is the known total distance and
$\lambda_{\text{pen}} \sim 10^6$ is a large penalty weight.

Use `scipy.optimize.least_squares` with `method='lm'` and the extra
constraint appended to the residual vector.

### Results

| Regime | Unique? |
|:-------|:--------|
| Default ($f_1$=3mm, $f_2$=50mm) | **YES** |
| Equal ($f_1$=$f_2$=10mm) | **YES** |
| Strong ($f_1$=1mm, $f_2$=100mm) | **YES** |
| Weak ($f_1$=50mm, $f_2$=200mm) | **YES** |

**Effect of approximate knowledge:**

| Precision of $d_1+d_2+d_3$ | Surviving solutions (of ~219) |
|:----------------------------|:------------------------------|
| ±20% | 94 |
| ±10% | 46 |
| ±5% | 25 |
| ±2% | 11 |
| ±1% | 5 |

### Practical considerations

- **Easiest strategy** — requires no extra measurements, just a ruler or
  CAD drawing of the column
- Exact knowledge gives uniqueness; approximate knowledge progressively
  reduces ambiguity
- The total distance $d_1 + d_2 + d_3$ is from the object (sample) plane
  through the two lenses to the detector/camera plane

---

## Strategy 2: Nonlinear Excitation Model ($\phi \propto I^2$)

### Concept

For magnetic lenses, the optical power is proportional to the square of
the excitation current:

$$\phi(I) = \alpha \cdot I^2$$

Expanding around $I_0$ with relative wobble $w = \delta I / I_0$:

$$\phi(I_0 + \delta I) = \alpha I_0^2 (1 + w)^2 = a(1 + 2w + w^2)$$

This gives $\phi(w) = a + b \cdot w + c \cdot w^2$ with the constraint:

$$c = \frac{b^2}{4a}$$

This nonlinear coupling between $a_2$ and $b_2$ breaks the linear
scaling degeneracy. Note that the proportionality constant $\alpha$ need
not be known — only the functional form $\phi \propto I^2$.

### Implementation

Fit to the model $\phi_i(w) = a_i + b_i w + (b_i^2/4a_i) w^2$ with 7
unknowns $(d_1, d_2, d_3, a_1, b_1, a_2, b_2)$. The quadratic coefficient
is derived, not free.

Use wobble amplitudes large enough to produce measurable curvature
(e.g., $w = 0.01$–$0.02$).

### Results

| Regime | Unique? |
|:-------|:--------|
| Default ($f_1$=3mm, $f_2$=50mm) | **YES** |
| Equal ($f_1$=$f_2$=10mm) | **YES** |
| Strong ($f_1$=1mm, $f_2$=100mm) | **YES** |
| Weak ($f_1$=50mm, $f_2$=200mm) | **YES** |

Works with as few as **K=5 wobble settings** (same as the linear model).

### Why it works

In the linear model, the degeneracy scales $(a_2, b_2) \to (\mu a_2, \mu b_2)$
for some factor $\mu$. The constraint $c = b^2/(4a)$ transforms as
$c' = (\mu b)^2/(4 \mu a) = \mu c$. But applying the same scaling to the
distances ($d_2, d_3 \to \lambda d_2, \lambda d_3$) requires different compensation
in $c$ than in $a$ and $b$, breaking the symmetry.

### Practical considerations

- **No extra measurements needed** beyond the standard wobble experiment
- Requires wobble amplitudes large enough for the quadratic term to be
  observable above noise
- The $\phi \propto I^2$ law is standard physics for round magnetic lenses
  (e.g., the Glaser bell model)
- Could be extended to other known nonlinearities (e.g., $\phi \propto I^n$)
- If the lens response deviates from pure $I^2$ at large wobble, higher-order
  terms or a lookup table from measured hysteresis curves could be used

---

## Strategy 3: Two Accelerating Voltages

### Concept

For magnetic lenses, the optical power depends inversely on the
relativistically-corrected accelerating voltage:

$$\phi \propto \frac{1}{V_r}, \quad V_r = V\left(1 + \frac{eV}{2 m_0 c^2}\right)$$

Changing the microscope voltage from $V_1$ to $V_2$ scales **all** optical
powers by a known ratio:

$$\gamma = \frac{V_r(V_1)}{V_r(V_2)}$$

The geometry ($d_1$, $d_2$, $d_3$) remains unchanged. This effectively
doubles the data: for each wobble setting, we measure A,B at two voltages,
giving 4 values instead of 2.

### Implementation

Fit simultaneously:
- At $V_1$: standard A,B data with optical powers $\phi_1, \phi_2$
- At $V_2$: A,B data with optical powers $\gamma\phi_1, \gamma\phi_2$
- Same distances $(d_1, d_2, d_3)$ for both
- 7 unknowns total (optical powers parameterised at $V_1$)

### Results

| Regime | Unique? |
|:-------|:--------|
| Default ($f_1$=3mm, $f_2$=50mm) | **YES** |
| Equal ($f_1$=$f_2$=10mm) | **YES** |
| Strong ($f_1$=1mm, $f_2$=100mm) | **YES** |
| Weak ($f_1$=50mm, $f_2$=200mm) | **YES** |

**Effect of voltage difference** (primary voltage 200 kV):

| Second voltage | $\gamma$ | Unique? |
|:---------------|:---------|:--------|
| 210 kV | 0.9447 | **YES** |
| 220 kV | 0.8945 | **YES** |
| 250 kV | 0.7686 | **YES** |
| 300 kV | 0.6162 | **YES** |
| 100 kV | 2.1783 | **YES** |

Even a modest 10 kV change (5%) is sufficient to break the degeneracy.

### Why it works

The degeneracy manifold scales $(\phi_2, b_2) \to (\mu\phi_2, \mu b_2)$
and $(d_2, d_3) \to (\lambda d_2, \lambda d_3)$. At voltage $V_1$, this
produces a family of solutions. At voltage $V_2$, the optical powers are
multiplied by $\gamma$, but the distances stay fixed. The compensating
$\lambda$ required differs at $V_2$ from $V_1$, so no single
$(\lambda, \mu)$ pair satisfies both voltage datasets simultaneously.

### Practical considerations

- Requires repeating the wobble experiment at a second accelerating
  voltage — doubles the experimental effort
- Even a small voltage change suffices ($\gamma \neq 1$ is all that matters)
- The relativistic correction is well-known and precise
- This strategy is completely independent of the nonlinear excitation model
- In principle, could combine with Strategy 2 for extra robustness

---

## Summary Comparison

| Strategy | Extra information | Extra measurements | Unique? | Robustness |
|:---------|:-----------------|:-------------------|:--------|:-----------|
| **1. Known $d_1+d_2+d_3$** | Total column length | None | YES | All 4 regimes |
| **2. Nonlinear $\phi(I)$** | Physics model ($\phi \propto I^2$) | None (wider wobble) | YES | All 4 regimes |
| **3. Two voltages** | Second HT measurement | Full repeat at $V_2$ | YES | All 4 regimes |

### Recommended approach

1. **Start with Strategy 2** (nonlinear model) — it requires no extra
   measurements and uses well-established magnetic lens physics. Simply use
   larger wobble amplitudes to make the quadratic term observable.

2. **Validate with Strategy 1** — if the total column length is known
   approximately, check that the solution's $d_1 + d_2 + d_3$ is
   consistent. This provides an independent cross-check.

3. **Use Strategy 3 as a fallback** — if the lens response is poorly
   characterised (e.g., hysteresis, electrostatic contributions), repeating
   at a different voltage provides model-independent degeneracy breaking.

### Key insight

The linear AB-only inverse problem has a fundamental 1-parameter
degeneracy: $d_2$, $d_3$, $f_2$, and $b_2$ can trade off continuously
while preserving all observables. Any single piece of information that
constrains the absolute scale of the $d_2$–$d_3$–$f_2$ relationship
breaks this degeneracy and yields a unique solution.

---

## Appendix: Parameters Always Recovered

Regardless of which strategy is used, these parameters are **always
correctly recovered** from AB data alone:

| Parameter | Description | Typical spread across spurious solutions |
|:----------|:------------|:-----------------------------------------|
| $d_1$ | Object distance | **Exact** (1.0×) |
| $a_1 = 1/f_1$ | Lens 1 optical power | Near-exact (1.0×) |
| $b_1$ | Lens 1 wobble sensitivity | **Exact** (1.0×) |

These are uncertain without additional information:

| Parameter | Description | Typical spread |
|:----------|:------------|:---------------|
| $d_2$ | Inter-lens distance | 5.6× |
| $d_3$ | Camera distance | 5.6× |
| $a_2 = 1/f_2$ | Lens 2 optical power | 20.5× |
| $b_2$ | Lens 2 wobble sensitivity | 31.8× |

## Appendix: Test Code

All numerical results were generated using
[test_strategies.py](test_strategies.py) with:
- `scipy.optimize.least_squares` (Levenberg–Marquardt)
- 300–500 random starting points per test
- Solutions grouped by 0.5% relative tolerance
- Verified across 4 parameter regimes (Default, Equal, Strong, Weak)

"""
Why Overdetermination Doesn't Guarantee Uniqueness in Nonlinear Problems

This demonstrates the fundamental difference between equation count and uniqueness.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

print("="*80)
print("WHY MORE EQUATIONS ≠ UNIQUE SOLUTION (for nonlinear problems)")
print("="*80)

# =============================================================================
# EXAMPLE 1: Simple Demonstration - Circle and Parabola Intersection
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: Two Equations, Two Unknowns - Multiple Solutions")
print("="*80)

print("""
Consider solving:
    x² + y² = 1      (equation 1: circle)
    y = x²           (equation 2: parabola)

This is 2 equations for 2 unknowns, yet it has MULTIPLE solutions!
""")

# Solve analytically
# Substitute y = x² into x² + y² = 1
# x² + x⁴ = 1
# Let u = x², then u² + u - 1 = 0
# u = (-1 ± √5) / 2

u_solutions = [(-1 + np.sqrt(5)) / 2, (-1 - np.sqrt(5)) / 2]
x_solutions = []
y_solutions = []

for u in u_solutions:
    if u >= 0:  # x² must be non-negative
        x_vals = [np.sqrt(u), -np.sqrt(u)]
        for x in x_vals:
            y = x**2
            # Check if on circle
            if abs(x**2 + y**2 - 1) < 1e-10:
                x_solutions.append(x)
                y_solutions.append(y)

print(f"Number of solutions: {len(x_solutions)}")
for i, (x, y) in enumerate(zip(x_solutions, y_solutions)):
    print(f"  Solution {i+1}: x = {x:+.4f}, y = {y:.4f}")
    print(f"    Check: x²+y² = {x**2 + y**2:.6f}, y-x² = {y - x**2:.6e}")

print("\nKey insight: Nonlinear equations can have multiple solutions")
print("             even when equation count = unknown count!")


# =============================================================================
# EXAMPLE 2: Your Lens Problem - Same (A,B) from Different Parameters
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: Lens Inversion - Why (A,B) Measurements Alone Are Insufficient")
print("="*80)

# Import the ABCD computation
import sys
sys.path.insert(0, '/home/dl277493/Documents/Code/TemGymCore/tests')
from transfer_matrices import full_abcd_2lens, propagation_matrix, lens_matrix

def compute_AB(z1, z2, z3, f1, f2):
    """Compute A, B from two-lens system"""
    M = full_abcd_2lens((z1, z2, z3), f1, f2, symbolic=False)
    return M[0, 0], M[0, 1]

# Configuration 1: "Ground truth"
z1_1 = 25e-6
z2_1 = 500e-6
z3_1 = 1.0
f1_1 = 25e-6
f2_1 = 500e-6

A1, B1 = compute_AB(z1_1, z2_1, z3_1, f1_1, f2_1)

print(f"Configuration 1 (ground truth):")
print(f"  z1={z1_1*1e3:.4f}mm, z2={z2_1*1e3:.4f}mm, z3={z3_1*1e3:.1f}mm")
print(f"  f1={f1_1*1e6:.2f}µm, f2={f2_1*1e6:.2f}µm")
print(f"  → A = {A1:.4f}, B = {B1:.6e} m")

# Try to find another configuration with same A, B but different parameters
# This demonstrates the degeneracy

print(f"\nSearching for degenerate configurations with identical (A, B)...\n")

def objective_match_AB(params):
    """Objective: match target (A, B) values"""
    z1, z2, z3, f1, f2 = params
    A, B = compute_AB(z1, z2, z3, f1, f2)
    return [(A - A1), (B - B1)]

# Try multiple random initializations
degenerate_solutions = []
n_trials = 50

for seed in range(n_trials):
    np.random.seed(seed)
    
    # Random initialization (perturb ground truth by ±50%)
    init = np.array([z1_1, z2_1, z3_1, f1_1, f2_1]) * (0.5 + np.random.rand(5))
    
    # Solve for (A,B) match
    bounds = (
        [5e-6, 100e-6, 0.3, 5e-6, 100e-6],   # Lower bounds
        [100e-6, 2e-3, 3.0, 100e-6, 2e-3]    # Upper bounds
    )
    
    result = least_squares(objective_match_AB, init, bounds=bounds, 
                          ftol=1e-12, xtol=1e-12, max_nfev=1000)
    
    # Check if converged to same (A,B)
    A_fit, B_fit = compute_AB(*result.x)
    error_A = abs(A_fit - A1)
    error_B = abs(B_fit - B1)
    
    if error_A < 1e-10 and error_B < 1e-10:
        # Check if it's actually different parameters
        param_diff = np.max(np.abs((result.x - np.array([z1_1, z2_1, z3_1, f1_1, f2_1])) 
                                   / np.array([z1_1, z2_1, z3_1, f1_1, f2_1])))
        
        if param_diff > 0.01:  # More than 1% different
            degenerate_solutions.append(result.x)

print(f"Found {len(degenerate_solutions)} distinct solutions with IDENTICAL (A, B)!")
print(f"Using only 1 defocus plane (2 measurements: A, B) for 5 unknowns\n")

# Show a few examples
for i, sol in enumerate(degenerate_solutions[:5]):
    z1, z2, z3, f1, f2 = sol
    A, B = compute_AB(z1, z2, z3, f1, f2)
    
    # Compute C element (not measured!)
    M = full_abcd_2lens((z1, z2, z3), f1, f2, symbolic=False)
    C = M[1, 0]
    
    diff_pct = np.abs((sol - np.array([z1_1, z2_1, z3_1, f1_1, f2_1])) 
                     / np.array([z1_1, z2_1, z3_1, f1_1, f2_1])) * 100
    
    print(f"Degenerate solution {i+1}:")
    print(f"  z1={z1*1e3:.4f}mm ({diff_pct[0]:+.1f}%), z2={z2*1e3:.4f}mm ({diff_pct[1]:+.1f}%), "
          f"z3={z3*1e3:.1f}mm ({diff_pct[2]:+.1f}%)")
    print(f"  f1={f1*1e6:.2f}µm ({diff_pct[3]:+.1f}%), f2={f2*1e6:.2f}µm ({diff_pct[4]:+.1f}%)")
    print(f"  → A = {A:.4f} (Δ={abs(A-A1):.2e}), B = {B:.6e} (Δ={abs(B-B1):.2e})")
    print(f"  → C = {C:.6e} (NOT MEASURED, different from truth!)\n")


# =============================================================================
# EXAMPLE 3: Why This Happens - The Missing Information
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: What's Missing? The C Element!")
print("="*80)

print("""
The ABCD matrix has 3 independent parameters (since det = AD - BC = 1):

    ⎡ A   B ⎤
    ⎣ C   D ⎦    where D = (1 + BC) / A

You measure:
• Intensity patterns → determines B/A (defocus) and |A| (magnification)
• That's only 2 values: (A, B)

You DON'T measure:
• C element (ray angle transformation from position)
• This requires measuring ray angles, not just positions/intensities

ANALOGY: It's like determining both height and weight from BMI alone.
         BMI = weight / height²
         Multiple (height, weight) pairs give the same BMI!
         
In your case: Multiple (z1, z2, z3, f1, f2) give the same (A, B)!
""")

# Compute C for ground truth
M_true = full_abcd_2lens((z1_1, z2_1, z3_1), f1_1, f2_1, symbolic=False)
C_true = M_true[1, 0]
D_true = M_true[1, 1]

print(f"Ground truth ABCD matrix:")
print(f"  A = {A1:.4f}      B = {B1:.6e} m    (MEASURED)")
print(f"  C = {C_true:.6e} m⁻¹   D = {D_true:.4f}      (NOT MEASURED)")
print(f"\nDegenerate solutions have:")
print(f"  • Same (A, B) ← This is what you measure")
print(f"  • Different C ← This is what you DON'T measure")
print(f"  • Different D = (1+BC)/A ← Determined by others")


# =============================================================================
# VISUALIZATION: Multiple Solutions in Parameter Space
# =============================================================================
print("\n" + "="*80)
print("VISUALIZATION: Solution Manifold in Parameter Space")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Circle-Parabola intersection (simple example)
ax1 = axes[0]
theta = np.linspace(0, 2*np.pi, 1000)
circle_x = np.cos(theta)
circle_y = np.sin(theta)
parabola_x = np.linspace(-1.5, 1.5, 1000)
parabola_y = parabola_x**2

ax1.plot(circle_x, circle_y, 'b-', linewidth=2, label='$x^2 + y^2 = 1$')
ax1.plot(parabola_x, parabola_y, 'r-', linewidth=2, label='$y = x^2$')
ax1.plot(x_solutions, y_solutions, 'go', markersize=12, label=f'{len(x_solutions)} solutions', zorder=5)
for i, (x, y) in enumerate(zip(x_solutions, y_solutions)):
    ax1.annotate(f'Sol {i+1}', (x, y), xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Simple Example: 2 Equations, 2 Unknowns\n→ Multiple Solutions', 
             fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.5, 1.5)

# Plot 2: Lens problem - projection of degeneracy
ax2 = axes[1]

if len(degenerate_solutions) > 0:
    all_solutions = [np.array([z1_1, z2_1, z3_1, f1_1, f2_1])] + degenerate_solutions
    
    # Extract parameter variations (normalize for plotting)
    z1_vals = [s[0]/z1_1 for s in all_solutions]
    z3_vals = [s[2]/z3_1 for s in all_solutions]
    
    ax2.scatter(z1_vals[0], z3_vals[0], c='green', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='Ground truth', zorder=5)
    ax2.scatter(z1_vals[1:], z3_vals[1:], c='red', s=100, alpha=0.7, 
               edgecolors='black', linewidths=1, label='Degenerate solutions')
    
    ax2.set_xlabel('z₁ / z₁_true', fontsize=12)
    ax2.set_ylabel('z₃ / z₃_true', fontsize=12)
    ax2.set_title(f'Lens Problem: {len(all_solutions)} Configurations\n'
                 f'→ ALL produce identical (A, B)', 
                 fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add annotation
    ax2.text(0.05, 0.95, f'{len(degenerate_solutions)} degenerate\nsolutions found',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('degeneracy_explanation.png', dpi=150, bbox_inches='tight')
print("\nSaved figure to: degeneracy_explanation.png")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: Why Equation Count Doesn't Guarantee Uniqueness")
print("="*80)

print("""
1. LINEAR SYSTEMS (simple):
   • More equations than unknowns + full rank → Unique solution ✓
   • Example: Linear regression, solving Ax = b

2. NONLINEAR SYSTEMS (your case):
   • More equations than unknowns + full rank ✗ NOT SUFFICIENT
   • Can have multiple discrete solutions
   • Reason: Observables (A, B) don't uniquely determine parameters
   
3. YOUR LENS PROBLEM:
   • You measure: Intensity patterns → (A, B) values
   • You DON'T measure: C element (ray angles)
   • Result: Multiple (z1, z2, z3, f1, f2) give same (A, B)
   • This is FUNDAMENTAL, not numerical error!

4. THE JACOBIAN PARADOX:
   • Full rank Jacobian → "Locally unique" (infinitesimal perturbations)
   • But can still have discrete global solutions (finite differences)
   • Analogy: Top of multiple hills all have zero gradient (local minima)

5. WHY WOBBLE + PRIORS HELP:
   • Wobble: Changes how degenerate solutions respond
   • Priors: Guide toward physically meaningful solution
   • Bounds: Eliminate unphysical parameter ranges
   • Together: Break most of the degeneracy
   
6. MATHEMATICAL CONDITION FOR UNIQUENESS:
   • Need: rank(Jacobian) = n AND global injectivity
   • Have: rank(Jacobian) = n BUT NOT globally injective
   • Fix: Add more observables (C element) OR constrain via priors

BOTTOM LINE: "More equations than unknowns" is necessary but NOT sufficient
             for nonlinear inverse problems. Need global injectivity too!
""")

plt.show()

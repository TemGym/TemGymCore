# Aberration demonstration: jets through native ray components

Open [aberration_demonstration.ipynb](aberration_demonstration.ipynb).
The notebook differentiates `component(ray)` / `run_to_end(ray, column)` at one
on-axis reference ray using `jax.experimental.jet`.

- `KrivanekLensCartesian`: first- through fifth-order slope jets recover all
  implemented Krivanek amplitudes and orientations. It preserves the original
  Krivanek ray convention while replacing polar arithmetic with polynomials.
- `SeidelLens`: the new stationary geometric-ray branch gives third-order
  position/slope jets that recover all nine joint Seidel coefficients.

The solves use output ray coordinates, not the wave-aberration function or
`phase_shift`. Multiple tangent seeds resolve the jet at the same zero primal.
Independent off-axis rays validate the reconstructed components afterward.
Executed coefficient tables, numerical checks and ray-aberration plots are saved.

Run all cells in the repository environment (`pip install -e ".[inversion]"`).
The notebook locates the source tree automatically and uses CPU float64.
Targeted component tests are in `tests/test_lens_ray_jets.py`.

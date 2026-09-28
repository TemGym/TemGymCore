# Condenser convergence control

`condenser_convergence_control.pdf` and `.svg` are the publication exports;
`.png` is the preview embedded in `../condenser_convergence_angle.ipynb`.
The JSON contains fitted condenser parameters, control limits, complete descent
histories, the focused-angle curve, and quadrature checks.

The figure shows projected gradient descent through the condenser recovered
from beam images. CL3 and the mini condenser select RMS convergence angles of
8, 12, 18, and 24 mrad while placing the beam waist at the specimen. Every run
starts from the same lens settings; CL1 and the objective current stay fixed.
The shaded range is 6.925–25.604 mrad for the selected CL3/mini control box at
200 kV with the 50 µm aperture. These are model predictions over illustrative
current limits, with the same geometric-aperture approximation as the image
calibration. Numerical target errors describe optimization precision within the
fitted model; experimental accuracy also depends on calibration uncertainty and
model validity.

The angle is the square root of the mean of the two transverse angular
variances of the transmitted beam. The focus condition minimizes the trace of
its spatial covariance in a local drift, placing that minimum at the specimen.
This RMS definition describes a distribution rather than the edge of a hard cone.

Reproduce from the repository root using the notebook's saved fit:

```bash
.venv/bin/python examples/microscope_models/convergence_angle/make_condenser_convergence_figure.py
```

Use `--experiment PATH` for another saved experiment with `fit.json` and
`experiment.json`, and `--output PATH` to choose the export directory. The standalone
notebook loads the saved fit and passes `microscope` directly. The chosen targets and
control box are declared near the top of `build_figure`.

The executed figure cell checks automatic gradients against central finite
differences, decreasing loss, current bounds, angle and waist tolerances, the
first crossover, and endpoint stability when aperture quadrature increases from
192 × 64 to 384 × 128. The saved run changes endpoint angles by at most
2.83 × 10⁻¹⁰ mrad under that refinement. Image generation and microscope fitting
are reused from the preceding calibration work.

## Ray evolution

The notebook also replays the accepted descent steps with the existing TemGym
`trace_ray_envelope`, `plot_ray_bundle`, and column-glyph helpers. A target selector,
Play button, iteration slider, Restart button, and Final setting button update
the full condenser, final lenses, and specimen zoom together. Run the controls
cell in a live notebook to interact with it.

- `condenser_ray_evolution.gif` animates every accepted step for 24 mrad and works
  without a running kernel.
- `condenser_ray_evolution.png` and `.pdf` compare the initial and final beams.

The source-to-specimen panels show a representative two-RMS meridional ray
bundle clipped at the aperture's nominal radius, as in the calibration browser.
The specimen zoom shows the one-RMS radius from the optimizer's smooth-aperture
moments; its short extension after the specimen is a local vacuum drift. Fixed
axis limits preserve the comparison across targets and steps. Axial and
transverse scales differ, so cone angles are visually exaggerated.

For the 24 mrad run, the initial angle is 16.338 mrad and the waist lies 5.906 µm
upstream of the specimen. Gradient descent changes CL3 and mini together to
increase the angular spread and place the waist at the specimen. The first
crossover stays fixed because CL1 and the source stay fixed.

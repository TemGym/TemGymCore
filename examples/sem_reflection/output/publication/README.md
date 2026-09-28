# Reflection calibration workflow figure

- `reflection_calibration_workflow.pdf`: vector artwork for manuscript placement.
- `reflection_calibration_workflow.svg`: editable text, arrows and microscope artwork.
- `reflection_calibration_workflow.png`: 600 dpi export.
- `reflection_calibration_workflow_preview.png`: screen preview.

Designed on a 180 × 113 mm canvas. Microscope diagrams use the saved initial and fitted
parameters, identical coordinate limits and the native traced beam envelopes. The
stack contains recorded synthetic diffraction images with a common crop and logarithmic
contrast, displayed in an oblique affine projection. The two comparison panels use the
same acquisition setting and preserve image aspect ratio.

The diagram illustrates the local Gaussian reflection calibration experiment, not a
real hardware acquisition. The initial model and measured data independently feed the
fit. Their arrows do not imply that the initial guess generated the recorded data.
Beam envelopes are enlarged, as in the notebook. SVG/PDF keep microscope geometry,
arrows and labels as vectors; diffraction images are embedded raster data.

Regenerate from the repository root:

```sh
.venv/bin/python examples/sem_reflection/make_calibration_workflow_figure.py
```

The script reads `output/calibration_gaussian/` and does not rerun acquisition or fitting.

## Explore / fit

`microscope_model_to_ptychography.pdf/.svg/.png` uses two vertically stacked panels on a 230 × 119.6 mm canvas:
(a) Explore and (b) Fit. The former shared-state/imaging row and its connecting arrows have been removed.

- **Explore:** a wider native vector microscope view with compact controls, beside its matching saved detector image.
- **Fit:** compact initial and recovered microscope views sit above identical recorded detector images (frame 52). Gold initial-model and teal fitted-model 2σ contours show the change in agreement, using the actual saved acquisition setting and shared image contrast and crop. The initial microscope is muted grey. The central recorded-pattern stack feeds the geometry/lens-strength fit; the working-distance captions and synthetic-calibration footer are omitted.

The compact layout enlarges the microscope views and detector image, with matched limits for the initial and fitted columns.

The first-panel microscope is redrawn with native plotting tools at a wider aspect,
using the saved notebook parameters and only its column-setup cells. The full image
render is not rerun. The larger detector view is redrawn from the unchanged
`output/copper/camera_view.npz` intensity, transposed into the notebook display
orientation, with native vector title, axis labels, ticks and order annotations. Static sliders use the original notebook
source and saved camera readouts. The detector image alongside matches that same saved
4° interactive state. The fitting route uses the separate 2° calibration pilot.


Regenerate with:

```sh
.venv/bin/python examples/sem_reflection/make_microscope_ptychography_figure.py
```

Add `--poster` to write `microscope_model_to_ptychography_poster_b1.*` instead: the same
layout on a 660 mm wide canvas (full width of a B1 portrait sheet with ~25 mm margins).
Fonts, strokes, markers, optical symbols, tick marks and point offsets scale by 660/230,
with an 18 pt minimum text size; PNG export is 300 dpi.

All figures retain editable vector annotations in SVG/PDF. The microscope drawings and controls are vectors; the saved detector intensity
and diffraction images remain embedded raster data.

All three microscope views use the same native `plot_reflection_column` renderer and
shared coil/yoke and red/blue deflector styling. In the fit comparison, the notebook's
sample-relative deflector positions are fixed and steering is zero: these identity
elements do not alter the calibration propagation and are not inferred parameters.

Both fit views use 40× transverse envelope display magnification upstream,
smoothly reduced to physical scale (1×) across the final objective-to-specimen
drift. Axial positions and the laboratory pose are unchanged. The Gaussian is
continued through the transverse specimen plane and polygons are clipped to the
physical surface, removing the end cap. There is no axial shear or horizontal
warping. Display widening does not change the optical model or detector predictions.

The displayed initial model is an illustrative bounded guess with a source at
−445 mm, condensers at −215 and −170 mm, and an objective at −58 mm. Its focal
lengths were selected to keep the calculated detector disagreement moderate despite
the clearly different geometry. Camera length is 136.37 mm versus the recovered
120.00 mm. The side view and detector contours use the same parameter vector.
Calibration was not rerun from this illustrative start; the recovered result is
the saved fit. The recorded-pattern stack is labelled “with perturbed lenses”:
its saved frames use the experiment's varied relative lens-power settings.

The right-hand parameter table reads ground truth from `scoring_truth.json` and
fit values/standard errors from `fit.json`. It reports all nine parameters with
±1 local standard error, computed from the fitted residual Jacobian covariance.
Positions and focal lengths are in mm; source RMS size is in nm. The canvas is
widened to 230 mm to retain readable diagrams and numerical columns.

Component labels spell out condenser lenses, deflectors, the objective lens,
electron source and specimen. Every parameter row carries its own unit. The explanatory footnotes below the
table are omitted; z denotes position relative to the specimen and f focal length.

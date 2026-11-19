---
name: temgym
description: How to model electron optics with TemGym in this repository. Use it before writing or editing any code that builds a column, traces rays or beams, fits a column to data, or uses the simulated microscope behind PyJEM.
---

# Modelling with TemGym

DRAFT for David to edit.

TemGym is the model. Code in `examples/` describes a microscope with TemGym components and
asks TemGym what the beam does. The code does not contain optics of its own.

## Read first

Read these before the first line of code, each time. Do not work from memory.

1. `ARCHITECTURE.md` and `docs/concepts.md`: the units, the ray vector, the sign conventions.
2. The source of each component that the code will use (`src/temgym_core/components/`).
   Read the class, its docstring and its `__call__`. Say which parameters it has and in
   which units.
3. `src/temgym_core/run.py`: the functions that trace a column.
4. One existing notebook that does a similar task, as the pattern to follow.

## Rules

1. **A microscope is a list of components**, in the order the beam meets them: lenses,
   deflectors, planes, a detector. Each physical element is one component.
2. **Use the component that TemGym has.** Look in `components/` before writing a function
   that bends, focuses, rotates or shifts a beam. The list at the bottom says which
   component models which element.
3. **TemGym traces the beam.** Use `run_to_end` for a ray and `column_matrix` or
   `position_and_angle_response` for the linear response of a column. Do not multiply
   transfer matrices by hand and do not write a lens formula in a notebook.
4. **Call TemGym directly. Do not wrap it.** A notebook cell builds the list of components
   and calls `run_to_end` on it, in plain lines that the reader can see. Do not write a
   function whose body is one TemGym call with the arguments renamed. Do not write a layer
   of helper functions between the notebook and TemGym. The first version of any
   calculation is the direct TemGym calls with no function at all.
5. **Gradients and fits use TemGym and JAX.** A derivative with respect to a component
   parameter comes from TemGym (`column_matrix`, `run_with_grads`, `solve_model`) or from
   `jax.grad` and `jax.jacfwd` through `run_to_end`. Do not use finite differences. A fit
   is a JAX optimisation of the component parameters.
6. **No scipy by default.** Do not import scipy to fit, solve, optimise or interpolate.
   The one permitted use is a test that compares a TemGym result with an independent
   calculation, and the code says that this is its purpose.
7. **Component parameters are physical quantities in SI units**, as the component's
   docstring states them. A conversion from a console value (a DAC value) to a physical
   quantity is one visible line next to the component, with its source.
8. **When TemGym lacks something, the feature goes into TemGym.** The steps:
   1. Search `src/temgym_core/` for the feature and for something close to it. Read that
      code.
   2. Tell David what is missing and ask before adding it. Propose where the feature goes
      and what its interface is.
   3. Add the feature to TemGym in the style of the code around it. Make it small, with no
      options that nobody uses, a numpydoc docstring with units, a test that fails without
      it, and an entry in `CHANGELOG.md`.
   4. Decide for the clarity of TemGym as a whole. If the feature fits an existing
      component, extend that component. Do not add a second way to do one thing.
   Do not write a local replacement in the notebook.
9. **Float64 always.** Enable `jax_enable_x64` at the top of the script or notebook.
10. **The simulated microscope is `temgym_core.pyjem_simulator`.** Code that will run at
    the real microscope talks to it through jemwrap's control functions (the `pyjem`
    skill), so the same code runs on both.

## Plots

1. **A diagram of a column and its rays comes from TemGym's legacy plot**:
   `plot_model(components, rays=..., plot_params=legacy_beam_plot_params(...))` from
   `temgym_core.plotting`. Do not draw a column with matplotlib calls of your own, and do
   not use `plot_model_plotly`, `plot_ray_bundle` or `plot_abcd_profile` unless David asks.
   Change the look through the arguments of `legacy_beam_plot_params`.
2. **Look at every figure before handing over.** Save the figure to a file and open the
   image. Check each item:
   - Every title, axis label, tick label, legend and component label is fully inside the
     figure.
   - No text lies on top of other text, on the data or on a component.
   - The text is large enough to read at the size the notebook shows.
   - Every axis has a label with its unit.
   Correct the figure and look again until every item holds.

## Which component for which element

| Element | Component | Module |
|---|---|---|
| Round magnetic lens with rotation and a field of finite length | `ElectromagneticLens` | `components/lenses.py` |
| Round magnetic lens of iron pole pieces, set by its gap and bore radius (Lenz's field; not thin, rays only) | `PolePieceLens` | `components/lenses.py` |
| Ideal thin lens | `Lens` | `components/lenses.py` |
| Deflector coil, one stage | `Deflector` | `components/deflection.py` |
| Deflector coil, two stages | `DoubleDeflector` | `components/deflection.py` |
| Image rotation alone | `Rotator` | `components/deflection.py` |
| Scan and descan coils | `Scanner`, `Descanner` | `components/deflection.py` |
| A plane where the beam is read | `Plane` | `components/planes.py` |
| Camera | `Detector` | `components/planes.py` |
| Aperture | `SigmoidAperture` | `components/actions.py` |
| Specimen | `components/specimens.py`, `components/interpolated.py` | |
| Electron source | `PointSource`, `ParallelBeam`, `gaussian_waist`, `sample_rays` | `source.py` |

## Before handing over

- List every TemGym class and function the code uses.
- List every piece of optics the code computes without TemGym, and say why. The list
  should be empty.
- List every function the code defines, with the number of places that call it. A function
  with one caller, or a function that only passes its arguments to TemGym, is removed.
- Run `grep -n scipy` on the changed files and report the output.
- For each figure, say that you opened the image and what you checked.
- Run the notebook top to bottom and report the command and its output.

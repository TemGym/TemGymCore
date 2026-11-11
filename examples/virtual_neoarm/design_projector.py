"""Design the virtual NEOARM's projection system and its lens table, and write virtual_neoarm.json.

Not fitted to any recording: a plausible five-lens column that does what a console's labels say.

    specimen -> objective -> IL1 -> IL2 -> IL3 -> PL1 -> camera

What is designed
----------------
* The lenses (TemGym ElectromagneticLens): the objective has a 1.9 mm focal length and never changes; IL1-IL3
  and PL1 have 0.080 ampere-turns per DAC unit (as measured on the real NEOARM from image rotation, PL1
  wound the other way), a strength chosen so each has a sensible focal length in the middle of its DAC range,
  and a small thickness.
* Lenses a micron or so off the axis, and a small hysteresis in each lens (see OFFSET_M and HYSTERESIS).
* The lens table. PL1 stays at 64000, as on the real console. At each row IL1-IL3 are solved so that
    - MAG: the specimen is in focus on the camera and magnified by the row's label;
    - DIFF: the camera shows the beam's angles (a shift at the specimen does not move the pattern) and the
      camera length is the row's label.
  Two conditions and three lenses leave one freedom; each row is the solution nearest the previous row, so
  the lenses change smoothly along the selector.

What is carried over from the real NEOARM (2026-11-10): how far one CLA1 and one CLA2 unit displace the beam
on the camera at 40k and 40 cm (which, with the labels, fixes the shift and tilt per unit), the camera, and
the console's objective and condenser DAC values.

Run from this folder:  python design_projector.py   (then python design_condenser.py: the condenser is
designed around this objective)
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)

from temgym_core.components import ElectromagneticLens, Plane                    # noqa: E402
from temgym_core.constants import effective_accelerating_potential              # noqa: E402
from temgym_core.run import position_and_angle_response                         # noqa: E402

HERE = Path(__file__).parent
VOLTAGE = 200e3
V_STAR = float(effective_accelerating_potential(VOLTAGE))
AMPERE_TURNS_PER_DAC = 0.080
LENSES = ("IL1", "IL2", "IL3", "PL1")

# The column (m from the specimen). Positions are not measured: a plausible layout.
OBJECTIVE_FOCAL_LENGTH, OBJECTIVE_Z = 1.9e-3, 1.9368e-3      # first image 0.10 m below the specimen, before IL1
LENS_Z = {"IL1": 0.15, "IL2": 0.22, "IL3": 0.30, "PL1": 0.42}
CAMERA_Z = 0.80
# Focal length (m) each lens has at a DAC value, which sets its strength Gc (f = V* / (Gc (NI)^2))
FOCAL_LENGTH_AT = {"IL1": (0.020, 32768), "IL2": (0.020, 32768), "IL3": (0.020, 32768), "PL1": (0.002, 64000)}
WINDING = {"IL1": -1, "IL2": -1, "IL3": -1, "PL1": +1}         # PL1 turns the image the other way
THICKNESS_PER_AMPERE_TURN = 1e-6                               # m: a 2-5 mm thick lens
PL1_DAC = 64000
# What a fit is not told about. Each lens sits a little off the axis (m, x and y), so the beam lands elsewhere on the
# camera at each row; and each lens's field lags behind its DAC value by up to 3.6 DAC units, depending on the values
# it had before (LensHysteresis: the lag of each backlash element, and how much of it the field shows).
OFFSET_M = {"IL1": (1.0e-6, -0.6e-6), "IL2": (-0.8e-6, 0.5e-6), "IL3": (0.5e-6, 0.9e-6), "PL1": (2.0e-6, -1.5e-6)}
HYSTERESIS = dict(widths_dac=[300.0, 3000.0], fractions=[0.002, 0.001])
LENSES_WITH_HYSTERESIS = ("IL1", "IL2", "IL3", "PL1", "CL1", "CL3")

MAGNIFICATIONS = [8000, 10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 80000, 100000, 120000,
                  150000, 200000, 250000, 300000, 400000, 500000, 600000]
CAMERA_LENGTHS_M = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00]
DAC_LOW, DAC_HIGH = 3000, 62000                                # IL1-IL3 stay inside this

# Carried over from the real NEOARM
KICK_PER_DAC = [[3.253059162429868e-10, 2.2492235538637627e-12, 0, 0], [0.0, -3.166900378670819e-10, 0, 0],
                [0, 0, -1.5599388756955268e-05, -1.828603741338481e-07],
                [0, 0, 1.0532854771847016e-07, 1.552622894448789e-05]]
CAMERA = dict(name="screenCamera", sensor_shape=[1916, 1916], sensor_pixel_m=2.75e-05, binnings=[1, 2, 4],
              psf_sigma_sensor_pixels=4.0, mirrored=False)
CAMERA_ROTATION_RAD = -1.0194177199619057
OBJECTIVE_DAC = {"OLc": 45818, "OLf": 38938}
CONDENSER_DAC = {"CL1": 31232, "CL2": 0, "CL3": 33866, "CM": 26624}

Gc = {n: V_STAR / (f * (AMPERE_TURNS_PER_DAC * dac) ** 2) for n, (f, dac) in FOCAL_LENGTH_AT.items()}
OBJECTIVE_GC = V_STAR / OBJECTIVE_FOCAL_LENGTH


def column(il):
    """TemGym components from the specimen to the camera, for IL1, IL2, IL3 at these DAC values."""
    dac = dict(zip(LENSES, (il[0], il[1], il[2], PL1_DAC)))
    return [ElectromagneticLens(z=OBJECTIVE_Z, turns=1.0, current=1.0, Gc=OBJECTIVE_GC),
            *(ElectromagneticLens(z=LENS_Z[n], turns=WINDING[n] * AMPERE_TURNS_PER_DAC, current=dac[n], Gc=Gc[n],
                                  Tc=THICKNESS_PER_AMPERE_TURN) for n in LENSES),
            Plane(z=CAMERA_Z)]


@jax.jit
def a_and_b(il):
    """A (camera position per specimen position) and B (per specimen angle, m/rad), real and imaginary parts."""
    a, b = position_and_angle_response(column(il), voltage=VOLTAGE)
    return jnp.array([a.real, a.imag, b.real, b.imag])


a_and_b_slope = jax.jit(jax.jacfwd(a_and_b))
FOCUS_TOLERANCE_M = 1e-8            # MAG: the plane in focus is this close to the specimen
SHIFT_TOLERANCE = 10.0              # DIFF: A below this (1 um at the specimen moves the pattern 0.1 px)
STAY_NEAR = 1e-3 / 20000            # weak pull towards the previous row


def solve_row(start, previous, *, magnification=None, camera_length_m=None):
    """IL1-IL3 for one row, nearest ``previous``. The focus condition is tightened in steps, so a start far from
    focus still finds it."""
    def misfit(il, loosen):
        ar, ai, br, bi = np.asarray(a_and_b(jnp.asarray(il)))
        if magnification:
            tolerance = magnification * FOCUS_TOLERANCE_M * loosen
            conditions = [np.hypot(ar, ai) / magnification - 1, br / tolerance, bi / tolerance]
        else:
            tolerance = SHIFT_TOLERANCE * loosen
            conditions = [np.hypot(br, bi) / camera_length_m - 1, ar / tolerance, ai / tolerance]
        return np.r_[conditions, STAY_NEAR * (np.asarray(il) - previous)]

    il = np.clip(start, DAC_LOW + 1, DAC_HIGH - 1)
    for loosen in (1e4, 1e2, 1.0):
        il = least_squares(misfit, il, args=(loosen,), bounds=(DAC_LOW, DAC_HIGH), x_scale=1000.0, xtol=1e-13,
                           ftol=1e-13, gtol=1e-13, max_nfev=300).x
        il = np.clip(il, DAC_LOW + 1, DAC_HIGH - 1)
    return il, np.abs(misfit(il, 1.0)[:3]).max()


def all_solutions(**target):
    """Settings of IL1-IL3 that meet a row's conditions, from many random starts."""
    rng, found = np.random.default_rng(0), []
    for _ in range(80):
        start = rng.uniform(DAC_LOW + 2000, DAC_HIGH - 2000, 3)
        il, worst = solve_row(start, start, **target)
        if worst < 1e-3:
            found.append(il)
    if not found:
        raise RuntimeError(f"no setting of IL1-IL3 gives {target}")
    return found


def solve_table(labels, start_label, key):
    """Every row of one table, going out from ``start_label`` row by row. The first row is the solution with the
    lenses nearest the middle of their range; each next row is the solution nearest the previous row (the selector
    jumps to another family of settings where the previous one runs out of DAC range, as real consoles do)."""
    rows = {start_label: min(all_solutions(**{key: start_label}), key=lambda il: np.abs(il - 32768).max())}
    at = labels.index(start_label)
    for order in (labels[at + 1:], labels[:at][::-1]):
        il = rows[start_label]
        for label in order:
            near, worst = solve_row(il, il, **{key: label})
            if worst > 1e-3:
                near = min(all_solutions(**{key: label}), key=lambda other: np.abs(other - il).max())
                print(f"   {key} {label}: the previous row's family ends; jumped by {np.abs(near - il).max():.0f} DAC units")
            rows[label] = il = near
    return {label: [int(round(v)) for v in rows[label]] for label in labels}


mag_rows = solve_table(MAGNIFICATIONS, 40000, "magnification")
diff_rows = solve_table(CAMERA_LENGTHS_M, 0.40, "camera_length_m")

print(f"{'row':>9} {'IL1':>6} {'IL2':>6} {'IL3':>6}   what the rounded DAC values give")
for label, il in mag_rows.items():
    ar, ai, br, bi = np.asarray(a_and_b(jnp.asarray(il, float)))
    a, b = np.hypot(ar, ai), np.hypot(br, bi)
    print(f"{label:8d}x {il[0]:6d} {il[1]:6d} {il[2]:6d}   magnification {a:9.0f}, out of focus by {b / a * 1e9:7.1f} nm, "
          f"image turned {np.degrees(np.arctan2(ai, ar)):+7.1f} deg")
for label, il in diff_rows.items():
    ar, ai, br, bi = np.asarray(a_and_b(jnp.asarray(il, float)))
    print(f"{label * 100:7.0f}cm {il[0]:6d} {il[1]:6d} {il[2]:6d}   camera length {np.hypot(br, bi):6.3f} m, "
          f"A {np.hypot(ar, ai):6.1f}, pattern turned {np.degrees(np.arctan2(bi, br)):+7.1f} deg")

virtual = dict(
    description="The virtual NEOARM at 200 kV, designed by design_projector.py (not fitted): a plausible objective, "
                "IL1-IL3 and PL1 (TemGym ElectromagneticLens) with a lens table solved so that every MAG row is in "
                "focus at its labelled magnification and every DIFF row has its labelled camera length. Carried over "
                "from the real NEOARM (2026-11-10): the CLA1/CLA2 response on the camera (kick_per_dac), the camera, "
                "the objective and condenser DAC values. The condenser is in condenser.json.",
    reference_voltage_v=VOLTAGE,
    lenses=dict(objective=dict(z_m=OBJECTIVE_Z, ampere_turns=1.0, Gc=OBJECTIVE_GC),
                **{n: dict(z_m=LENS_Z[n], ampere_turns_per_dac=WINDING[n] * AMPERE_TURNS_PER_DAC, Gc=Gc[n],
                           Tc=THICKNESS_PER_AMPERE_TURN, offset_m=list(OFFSET_M[n])) for n in LENSES}),
    camera_z_m=CAMERA_Z, camera_rotation_rad=CAMERA_ROTATION_RAD, camera=CAMERA,
    deflectors=dict(kick_per_dac=KICK_PER_DAC),
    objective_dac=OBJECTIVE_DAC, condenser_dac=CONDENSER_DAC,
    hysteresis={n: HYSTERESIS for n in LENSES_WITH_HYSTERESIS},
    lens_table=dict(MAG={str(m): dict(zip(LENSES, [*il, PL1_DAC])) for m, il in mag_rows.items()},
                    DIFF={str(length): dict(zip(LENSES, [*il, PL1_DAC])) for length, il in diff_rows.items()}),
)
(HERE / "virtual_neoarm.json").write_text(json.dumps(virtual, indent=1))
print("written", HERE / "virtual_neoarm.json")

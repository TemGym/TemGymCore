"""Fit the NEOARM condenser lenses CL1 and CL3 to two recordings of the real microscope.

1. The focus curve, recorded on 2026-11-10 (beam_per_magnification.zip). At each MAG row from
   8k to 500k, at spot 2, an automatic step set CL3. The beam on the SightX then had an rms size
   of 60 px along one axis.
2. The parallel settings (calimages.zip). At spots 0 to 4, an operator set CL3 so that the beam
   at the specimen was parallel. Each setting is good to about 500 DAC units.

Both recordings have CM at 26624, the console's alpha 1. The fit uses the projector lenses and the
objective of projector_lenses.json as they are. The free numbers are the gap and bore of CL1, CL3
and CM, the rms size of the source and its rms spread of angles. CM keeps one value at alpha 1, so
its gap and bore are those of a static component that nobody measured. The rows from MAG 8k to 15k
stay out of the fit as a test, because the projector is 7 % to 16 % low there. The script writes the
numbers to condenser.json, for neoarm_model.ipynb.

The data do not determine the size of the source. Its fitted value runs to zero, so the beam
size on the SightX comes from the spread of angles alone.

The fit follows the focus curve. It does not reach the parallel settings. The model's parallel beam
is 600 to 3000 CL3 units away from them, more at the larger spot sizes. Freeing the source position
or a factor on the condenser coils does not remove that error. The cause is not known. A run takes a
few minutes on a GPU.

Along the axis, lengths are in units of L, the distance from the specimen to the SightX. Across
the axis, positions are in metres, so a slope is in metres per L.
"""

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import Lens, Plane, PolePieceLens
from temgym_core.run import column_matrix, position_and_angle_response

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
# The table of beam_per_magnification.zip, copied out of the zip unchanged.
FOCUS_CURVE = HERE / "recordings/beam_per_magnification_2026-11-10.csv"
# CL1 and CL3 of the five parallel frames of calimages.zip, one line for each frame.
PARALLEL = HERE / "recordings/parallel_settings.csv"
CONSOLE = json.loads(
    (HERE.parent / "microscope_inversion/recordings/2026-11-17/console_tables.json").read_text()
)
DRAWING = json.loads((HERE.parent / "virtual_neoarm/neoarm_column_diagram.json").read_text())["z_m"]
PROJECTOR = json.loads((HERE / "projector_lenses.json").read_text())
VOLTAGE = 200e3  # the high tension of the NEOARM in both recordings
PROJECTOR_LENSES = ("IL1", "IL2", "IL3", "PL1")
L_DRAWING = DRAWING["Screen"]
MM = 1e-3 / L_DRAWING
Z = {name: z / L_DRAWING for name, z in DRAWING.items()}
# The SightX is 12 cm wide, with 1916 pixels. The recording read the frames at binning 4.
PIXEL_M = 0.12 / 1916 * 4
CM_ALPHA_1 = 26624  # CM in both recordings, the console's alpha 1

# No measurement gives the ampere-turns per DAC unit of the condenser lenses. They use the mean of
# the four projector lenses, which agree to 1 %.
CONDENSER_AMPERE_TURNS_PER_DAC = float(
    np.mean(np.abs(list(PROJECTOR["ampere_turns_per_dac"].values())))
)

# Allowed errors and priors. Each value is a choice. No measurement gives it.
# The beam sizes of the focus curve agree with the model to 5 %.
SIZE_ERROR = 0.05
# An operator's parallel setting is good to about 500 DAC units of CL3.
PARALLEL_ERROR_DAC = 500.0
# A condenser lens has a pole-piece gap and a bore radius of at least 1 mm.
SMALLEST_POLE_PIECE = 1 * MM
# The field of CL1 or CL3 (half the gap and two bore radii each side) stays within 80 % of the
# distance between the two lenses.
FIELD_CLEARANCE = 0.8 * (Z["CL2"] - Z["CL1"])

# The focus curve. One row, MAG 100k, has the flag "good" at False and stays out.
with open(FOCUS_CURVE, newline="") as table:
    curve = [r for r in csv.DictReader(table) if r["good"] == "True"]
console_mag = {r["mag"][0]: r for r in CONSOLE["rows"]["MAG"]}
curve_mag = np.array([int(r["magnification"]) for r in curve])
curve_dac = np.array(
    [
        [float(r["CL1"]), float(r["CL3"])] + [console_mag[m][n] for n in PROJECTOR_LENSES]
        for r, m in zip(curve, curve_mag)
    ]
)
curve_size_px = np.array([float(r["beam_rms_px"]) for r in curve])
kept_back = curve_mag <= 15000

# The parallel settings, one for each spot, with CL1 and CL3 as the console gave them.
with open(PARALLEL, newline="") as table:
    parallel = np.array(sorted([float(r["cl1"]), float(r["cl3"])] for r in csv.DictReader(table)))
print("parallel settings (CL1, CL3):", parallel.astype(int).tolist())


def numbers_from(u):
    """Map the fit's free numbers to the condenser's numbers, in L and in metres."""
    return dict(
        cl1_gap=jnp.exp(u[0]),
        cl1_bore=jnp.exp(u[1]),
        cl3_gap=jnp.exp(u[2]),
        cl3_bore=jnp.exp(u[3]),
        source_size=jnp.exp(u[4]),
        source_spread=jnp.exp(u[5]),
        cm_gap=jnp.exp(u[6]),
        cm_bore=jnp.exp(u[7]),
    )


def condenser(numbers, cl1, cl3):
    """Return the column from the source to the specimen for the DAC values ``cl1`` and ``cl3``."""
    relay = (Z["Corrector Bottom"] - Z["Corrector Top"]) / 4
    turns = CONDENSER_AMPERE_TURNS_PER_DAC
    return [
        PolePieceLens(
            z=Z["CL1"],
            turns=turns,
            current=cl1,
            gap=numbers["cl1_gap"],
            bore_radius=numbers["cl1_bore"],
        ),
        # On the drawing, CL3 sits where the label says CL2.
        PolePieceLens(
            z=Z["CL2"],
            turns=turns,
            current=cl3,
            gap=numbers["cl3_gap"],
            bore_radius=numbers["cl3_bore"],
        ),
        Lens(z=Z["Corrector Top"] + relay, focal_length=relay),
        Lens(z=Z["Corrector Bottom"] - relay, focal_length=relay),
        PolePieceLens(
            z=Z["CLMini"],
            turns=turns,
            current=CM_ALPHA_1,
            gap=numbers["cm_gap"],
            bore_radius=numbers["cm_bore"],
        ),
        Lens(z=-PROJECTOR["objective_z"], focal_length=PROJECTOR["objective_f"]),
        Plane(z=0.0),
    ]


def projector(dac):
    """Return the column from the specimen to the SightX for the projector DAC values ``dac``."""
    lenses = [
        PolePieceLens(
            z=PROJECTOR["lens_z"][name],
            turns=1.0,
            current=PROJECTOR["ampere_turns_per_dac"][name] * dac[i],
            gap=PROJECTOR["pole_gap"][name],
            bore_radius=PROJECTOR["bore_radius"][name],
        )
        for i, name in enumerate(PROJECTOR_LENSES)
    ]
    return [
        Lens(z=PROJECTOR["objective_z"], focal_length=PROJECTOR["objective_f"]),
        *lenses,
        Plane(z=1.0),
    ]


def beam_size_px(numbers, dac):
    """Return the rms beam size on the SightX along one axis, in pixels, for one row."""
    column = condenser(numbers, dac[0], dac[1]) + projector(dac[2:])
    a, b = position_and_angle_response(column, source_z=Z["Source"], voltage=VOLTAGE)
    return (
        jnp.sqrt(
            (jnp.abs(a) * numbers["source_size"]) ** 2
            + (jnp.abs(b) * numbers["source_spread"]) ** 2
        )
        / PIXEL_M
    )


def exit_angle(numbers, cl1, cl3):
    """Return the angle at the specimen per unit angle at the source, without the image rotation.

    A parallel beam has zero. Every ray from one point of the source then leaves the condenser
    parallel.
    """
    lenses = condenser(numbers, cl1, cl3)
    m = column_matrix(lenses, source_z=Z["Source"], voltage=VOLTAGE)
    rotation = sum(
        lens.rotation_angle(VOLTAGE) for lens in lenses if isinstance(lens, PolePieceLens)
    )
    return jnp.real((m[2, 2] + 1j * m[3, 2]) * jnp.exp(-1j * rotation))


def misfits(u):
    """Return every misfit, each in units of its allowed error."""
    numbers = numbers_from(u)
    sizes = jax.vmap(lambda dac: beam_size_px(numbers, dac))(jnp.asarray(curve_dac[~kept_back]))
    parts = [jnp.log(sizes / curve_size_px[~kept_back]) / SIZE_ERROR]
    # The distance in CL3 units from each parallel setting to the model's parallel beam.
    angle = jax.vmap(lambda cl1, cl3: exit_angle(numbers, cl1, cl3))(*parallel.T)
    slope = jax.vmap(jax.grad(lambda cl3, cl1: exit_angle(numbers, cl1, cl3)))(
        parallel[:, 1], parallel[:, 0]
    )
    parts.append(angle / slope / PARALLEL_ERROR_DAC)
    for gap, bore in (
        (numbers["cl1_gap"], numbers["cl1_bore"]),
        (numbers["cl3_gap"], numbers["cl3_bore"]),
        (numbers["cm_gap"], numbers["cm_bore"]),
    ):
        parts += [
            jax.nn.relu(SMALLEST_POLE_PIECE - gap)[None] / (0.1 * MM),
            jax.nn.relu(SMALLEST_POLE_PIECE - bore)[None] / (0.1 * MM),
            jax.nn.relu(gap / 2 + 2 * bore - FIELD_CLEARANCE / 2)[None] / (1 * MM),
        ]
    return jnp.concatenate(parts)


def levenberg_marquardt_step(state, _):
    """Solve (J^T J + lambda diag J^T J) step = -J^T r, and keep the step if the misfit drops."""
    u, damping, cost = state
    r, jacobian = misfits(u), jax.jacfwd(misfits)(u)
    h = jacobian.T @ jacobian
    trial = u + jnp.linalg.solve(h + damping * jnp.diag(jnp.diag(h) + 1e-9), -jacobian.T @ r)
    trial_cost = misfits(trial) @ misfits(trial)
    better = jnp.isfinite(trial_cost) & (trial_cost < cost)
    return (
        jnp.where(better, trial, u),
        jnp.where(better, damping / 3, damping * 5),
        jnp.where(better, trial_cost, cost),
    ), None


@jax.jit
@jax.vmap
def fit(u):
    """Run 300 Levenberg-Marquardt steps from the start ``u``."""
    (u, _, cost), _ = jax.lax.scan(
        levenberg_marquardt_step, (u, 1e-2, misfits(u) @ misfits(u)), None, length=300
    )
    return u, cost


# The fit starts from 256 random points. The gaps of CL1, CL3 and CM are 3 to 40 mm, and each bore
# is a fifth of the gap to twice the gap. The source size is 1 nm to 100 nm, and the spread is 1e-6
# to 1e-3 rad.
rng = np.random.default_rng(0)
starts = np.array(
    [
        np.r_[
            np.log(g1 := rng.uniform(3, 40) * MM),
            np.log(g1 * rng.uniform(0.2, 2.0)),
            np.log(g3 := rng.uniform(3, 40) * MM),
            np.log(g3 * rng.uniform(0.2, 2.0)),
            np.log(rng.uniform(1e-9, 1e-7)),
            np.log(rng.uniform(1e-6, 1e-3) * L_DRAWING),
            np.log(gm := rng.uniform(3, 40) * MM),
            np.log(gm * rng.uniform(0.2, 2.0)),
        ]
        for _ in range(256)
    ]
)
ends, costs = fit(jnp.asarray(starts))
costs = np.where(np.isfinite(costs), np.asarray(costs), np.inf)
order = np.argsort(costs)
best = ends[order[0]]
numbers = numbers_from(best)
r = np.asarray(misfits(best))
print(
    f"misfit (sum of squares of {r.size} weighted residuals) {costs[order[0]]:.1f}; "
    f"the five best starts: {np.round(costs[order[:5]], 1)}"
)
print("the five best starts, gap and bore of CL1, CL3 and CM (mm):")
for k in order[:5]:
    n = numbers_from(ends[k])
    print(
        "   ",
        np.round(
            [
                float(n[key]) / MM
                for key in ("cl1_gap", "cl1_bore", "cl3_gap", "cl3_bore", "cm_gap", "cm_bore")
            ],
            1,
        ),
    )
print(
    f"source: rms size {float(numbers['source_size']) * 1e9:.1f} nm, "
    f"rms spread {float(numbers['source_spread']) / L_DRAWING * 1e6:.1f} urad with the drawing's L"
)

sizes = np.asarray(jax.vmap(lambda dac: beam_size_px(numbers, dac))(jnp.asarray(curve_dac)))
print("focus curve, model / measured size (rows 8k to 15k kept back):")
for m, s, measured, held in zip(curve_mag, sizes, curve_size_px, kept_back):
    print(f"   MAG {m:>6}: {s / measured:.2f}{'  (kept back)' if held else ''}")
angle = jax.vmap(lambda cl1, cl3: exit_angle(numbers, cl1, cl3))(*parallel.T)
slope = jax.vmap(jax.grad(lambda cl3, cl1: exit_angle(numbers, cl1, cl3)))(
    parallel[:, 1], parallel[:, 0]
)
print(
    "parallel settings, distance to the model's parallel beam (CL3 units):",
    np.round(np.asarray(angle / slope)),
)

(HERE / "condenser.json").write_text(
    json.dumps(
        dict(
            source="fit_condenser.py: CL1 and CL3 fit to beam_per_magnification.zip "
            "(2026-11-10) and the "
            "parallel settings of calimages.zip, CM at 26624 (alpha 1)",
            units="lengths along the axis in L; source size in metres; spread in metres per L",
            misfit=float(costs[order[0]]),
            note="The data do not determine source_size; the fit drives it to zero.",
            ampere_turns_per_dac=CONDENSER_AMPERE_TURNS_PER_DAC,
            pole_gap=dict(
                CL1=float(numbers["cl1_gap"]),
                CL3=float(numbers["cl3_gap"]),
                CM=float(numbers["cm_gap"]),
            ),
            bore_radius=dict(
                CL1=float(numbers["cl1_bore"]),
                CL3=float(numbers["cl3_bore"]),
                CM=float(numbers["cm_bore"]),
            ),
            source_size=float(numbers["source_size"]),
            source_spread=float(numbers["source_spread"]),
        ),
        indent=1,
    )
)
print("wrote condenser.json")

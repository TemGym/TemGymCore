"""Fit the NEOARM projector lenses to the lens changes recorded on 2026-11-10.

The recording kicked the beam at two console rows, MAG 40k (CLA1) and DIFF 40 cm (CLA2). It
repeated the kicks with IL1, IL2, IL3 or PL1 moved up and down. Each lens change scales and
turns the kick on the SightX. The ratio to the kick at the console row does not depend on the
unknown deflector strength.

First the script finds the ampere-turns per DAC unit of each lens from how far each change
turns the image. Then it fits the objective post-field and four PolePieceLens components to
the 16 complex ratios and to the console table of 2026-11-17. In the table, every MAG row
keeps the image in focus on the SightX. The labels set only the mean scale. One number scales the
MAG magnifications and one scales the DIFF camera lengths. The shape of each curve comes from the
lenses. The script writes the numbers to projector_lenses.json, for neoarm_model.ipynb. A run
takes about two minutes on a GPU.

All lengths along the column are in units of L, the distance from the specimen to the SightX.
The ratios have no length in them. The priors pull the lens positions towards the column
drawing. The drawing's specimen-to-screen distance converts the millimetres of the priors and
start values to L. This assumes that the screen of the drawing is at the SightX.
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import Lens, Plane, PolePieceLens
from temgym_core.constants import compute_Rc_from_voltage
from temgym_core.run import position_and_angle_response

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
RECORDING = (
    HERE.parent / "microscope_inversion/recordings/2026-11-10/imaging-system-fitting-40k-40cm.zip"
)
DRAWING = json.loads((HERE.parent / "virtual_neoarm/neoarm_column_diagram.json").read_text())["z_m"]
VOLTAGE = 200e3
PROJECTOR = ("IL1", "IL2", "IL3", "PL1")

CONSOLE = json.loads(
    (HERE.parent / "microscope_inversion/recordings/2026-11-17/console_tables.json").read_text()
)
L_DRAWING = DRAWING["Screen"]
DRAWING_GAPS = np.diff([0, *(DRAWING[name] for name in PROJECTOR), L_DRAWING]) / L_DRAWING
MM = 1e-3 / L_DRAWING

# Allowed errors and priors. Each value is a choice. No measurement gives it.
# The x and y kicks of one setting agree to about 0.3 %. The fit allows 0.5 % in each ratio.
RATIO_ERROR = 0.005
# At the console row, a tilt at the specimen moves the MAG image as little as a 1 um defocus.
MAG_FOCUS_ERROR = 1e-3 * MM
# At the console row, a shift at the specimen moves the DIFF pattern a little. The movement is
# at most 1 % of the movement from a tilt of one shift per objective focal length.
DIFF_FOCUS_ERROR = 0.01
# Lens positions stay within about 30 % of the gaps on the drawing.
POSITION_ERROR = 0.3
# The objective focal length is about 2 mm, within a factor of 1.5.
OBJECTIVE_F, OBJECTIVE_F_FACTOR = 2 * MM, 1.5
# A projector lens has a pole-piece gap and a bore radius of at least 1 mm.
SMALLEST_POLE_PIECE = 1 * MM
# The field of a lens (half the gap and two bore radii each side) stays clear of its neighbours.
FIELD_CLEARANCE = 0.8
# In each MAG row of the table, a tilt at the specimen moves the image as little as a 5 um defocus.
TABLE_FOCUS_ERROR = 5 * MAG_FOCUS_ERROR
# The mean of the model's magnifications, and of its camera lengths, agrees with the labels to 2 %.
LABEL_SCALE_ERROR = 0.02

# The console rows from MAG 8k to 500k and the DIFF camera lengths from 8 cm to 2 m, with the
# objective of these modes. The script leaves out the X25k row, because its PL1 value of 65535 is
# the end of the DAC range. The DIFF labels are in metres, and the drawing's L converts them.
MAG_ROWS = [
    r for r in CONSOLE["rows"]["MAG"] if 8000 <= r["mag"][0] <= 500000 and r["mag"][2] != "X25k"
]
DIFF_ROWS = [
    r
    for r in CONSOLE["rows"]["DIFF"]
    if r["mag"][1] == "mm" and r["OLf"] > 0 and 80 <= r["mag"][0] <= 2000
]
MAG_DAC = np.array([[r[name] for name in PROJECTOR] for r in MAG_ROWS], float)
DIFF_DAC = np.array([[r[name] for name in PROJECTOR] for r in DIFF_ROWS], float)
MAG_LABEL = np.array([r["mag"][0] for r in MAG_ROWS], float)
DIFF_LABEL = np.array([r["mag"][0] / 1000 for r in DIFF_ROWS], float) / L_DRAWING

# Each block has five frames. They have no kick, then kicks of +x, -x, +y and -y.
with zipfile.ZipFile(RECORDING) as archive:
    table = next(name for name in archive.namelist() if name.endswith("/table.csv"))
    frames = list(csv.DictReader(io.StringIO(archive.read(table).decode())))
blocks = []
for i in range(len(frames) - 4):
    rows = frames[i : i + 5]
    roles = [r["role"] for r in rows]
    one_setting = len({(r["mode"], r["setting"]) for r in rows}) == 1
    usable = all(r["beam_usable"] == "True" for r in rows)
    if roles != ["not deflected", "deflected +", "deflected -", "deflected +", "deflected -"]:
        continue
    if not (one_setting and usable):
        continue
    coil = rows[1]["deflector"]
    beam = np.array([complex(float(r["beam_x_px"]), float(r["beam_y_px"])) for r in rows])
    span = np.array(
        [
            float(rows[1][coil + "_x"]) - float(rows[2][coil + "_x"]),
            float(rows[3][coil + "_y"]) - float(rows[4][coil + "_y"]),
        ]
    )
    blocks.append(
        dict(
            mode=rows[0]["mode"],
            setting=rows[0]["setting"],
            dac=np.array([float(rows[0][name]) for name in PROJECTOR]),
            kick=(beam[[1, 3]] - beam[[2, 4]]) / span,
        )
    )

# For each mode, keep the console row, the lens settings and the measured ratios of the x and y
# kicks.
modes = []
for mode in ("MAG 40k", "DIFF 40 cm"):
    group = [b for b in blocks if b["mode"] == mode]
    row = next(b for b in group if b["setting"] == f"{mode} row")
    changed = [b for b in group if ": " in b["setting"]]
    modes.append(
        dict(
            name=mode,
            row_dac=row["dac"],
            changed_dac=np.array([b["dac"] for b in changed]),
            ratio=np.array([b["kick"] / row["kick"] for b in changed]),
            labels=[b["setting"].split(": ")[1] for b in changed],
        )
    )

# A lens turns the image by Rc * NI. The turn of each kick per DAC unit of the lens that changed
# gives the ampere-turns per DAC unit. The script uses the median over the four changes of each
# lens. The measured turns are below 0.13 rad, so the complex logarithm below does not wrap at pi.
rc = float(compute_Rc_from_voltage(VOLTAGE))
turns_per_dac = {name: [] for name in PROJECTOR}
for mode in modes:
    for dac, ratio in zip(mode["changed_dac"], mode["ratio"]):
        i = int(np.argmax(np.abs(dac - mode["row_dac"])))
        turns_per_dac[PROJECTOR[i]].append(np.angle(ratio).mean() / (dac[i] - mode["row_dac"][i]))
AMPERE_TURNS_PER_DAC = np.array([np.median(turns_per_dac[name]) / rc for name in PROJECTOR])
print("ampere-turns per DAC unit:", dict(zip(PROJECTOR, np.round(AMPERE_TURNS_PER_DAC, 4))))


def numbers_from(u):
    """Map the fit's free numbers to the model's numbers, all in units of L."""
    gaps = jax.nn.softmax(jnp.log(DRAWING_GAPS) + jnp.r_[u[2:6], 0.0])
    objective_f = jnp.exp(u[0])
    return dict(
        objective_f=objective_f,
        objective_z=objective_f * (1 + jnp.exp(-u[1])),
        lens_z=jnp.cumsum(gaps)[:4],
        gaps=gaps,
        pole_gap=jnp.exp(u[6:10]),
        bore_radius=jnp.exp(u[10:14]),
    )


def projector(numbers, dac):
    """Return the column from the specimen to the SightX for the projector DAC values ``dac``."""
    # The coil current is in proportion to the DAC value, with zero current at DAC 0. The image
    # rotation gives only the slope of that line. Its zero is an assumption.
    excitation = AMPERE_TURNS_PER_DAC * dac
    lenses = [
        PolePieceLens(
            z=numbers["lens_z"][i],
            turns=1.0,
            current=excitation[i],
            gap=numbers["pole_gap"][i],
            bore_radius=numbers["bore_radius"][i],
        )
        for i in range(4)
    ]
    objective = Lens(z=numbers["objective_z"], focal_length=numbers["objective_f"])
    return [Plane(z=0.0), objective, *lenses, Plane(z=1.0)]


def misfits(u):
    """Return every misfit, each in units of its allowed error."""
    numbers = numbers_from(u)
    parts = []
    for use, mode in enumerate(modes):
        # MAG 40k kicks with CLA1, a shift at the specimen. DIFF 40 cm kicks with CLA2, a tilt.
        a_row, b_row = position_and_angle_response(
            projector(numbers, mode["row_dac"]), voltage=VOLTAGE
        )
        a, b = jax.vmap(
            lambda dac: position_and_angle_response(projector(numbers, dac), voltage=VOLTAGE)
        )(jnp.asarray(mode["changed_dac"]))
        model = (a / a_row) if use == 0 else (b / b_row)
        error = jnp.log(model[:, None]) - jnp.log(mode["ratio"])
        parts += [jnp.real(error).ravel() / RATIO_ERROR, jnp.imag(error).ravel() / RATIO_ERROR]
        # At the console row the image (MAG) or the pattern (DIFF) is in focus on the SightX.
        if use == 0:
            parts.append((jnp.abs(b_row / a_row) / MAG_FOCUS_ERROR)[None])
        else:
            parts.append((jnp.abs(a_row / b_row) * numbers["objective_f"] / DIFF_FOCUS_ERROR)[None])
    a, b = jax.vmap(
        lambda dac: position_and_angle_response(projector(numbers, dac), voltage=VOLTAGE)
    )(jnp.asarray(MAG_DAC))
    parts += [
        jnp.abs(b / a) / TABLE_FOCUS_ERROR,
        jnp.mean(jnp.log(jnp.abs(a) / MAG_LABEL))[None] / LABEL_SCALE_ERROR,
    ]
    a, b = jax.vmap(
        lambda dac: position_and_angle_response(projector(numbers, dac), voltage=VOLTAGE)
    )(jnp.asarray(DIFF_DAC))
    parts.append(jnp.mean(jnp.log(jnp.abs(b) / DIFF_LABEL))[None] / LABEL_SCALE_ERROR)
    neighbours = jnp.minimum(numbers["gaps"][:4], numbers["gaps"][1:5])
    field_extent = numbers["pole_gap"] / 2 + 2 * numbers["bore_radius"]
    parts += [
        jnp.log(numbers["gaps"] / DRAWING_GAPS) / POSITION_ERROR,
        jnp.log(numbers["objective_f"] / OBJECTIVE_F)[None] / np.log(OBJECTIVE_F_FACTOR),
        jax.nn.relu(SMALLEST_POLE_PIECE - numbers["pole_gap"]) / (0.1 * MM),
        jax.nn.relu(SMALLEST_POLE_PIECE - numbers["bore_radius"]) / (0.1 * MM),
        jax.nn.relu(field_extent - FIELD_CLEARANCE * neighbours) / (1 * MM),
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
    """Run 400 Levenberg-Marquardt steps from the start ``u``."""
    (u, _, cost), _ = jax.lax.scan(
        levenberg_marquardt_step, (u, 1e-2, misfits(u) @ misfits(u)), None, length=400
    )
    return u, cost


# The fit starts from 256 random points. The objective focal length is 1.5 to 3 mm, and the
# positions are near the drawing. The pole-piece gaps are 3 to 30 mm, and each bore is a fifth of
# the gap to twice the gap.
rng = np.random.default_rng(0)
starts = np.array(
    [
        np.r_[
            np.log(rng.uniform(1.5, 3.0) * MM),
            np.log(rng.uniform(20, 80)),
            rng.uniform(-0.2, 0.2, 4),
            np.log(gap := rng.uniform(3, 30, 4) * MM),
            np.log(gap * rng.uniform(0.2, 2.0, 4)),
        ]
        for _ in range(256)
    ]
)
ends, costs = fit(jnp.asarray(starts))
costs = np.where(np.isfinite(costs), np.asarray(costs), np.inf)
best = ends[int(np.argmin(costs))]
numbers = numbers_from(best)
r = np.asarray(misfits(best))
print(
    f"misfit (sum of squares of {r.size} weighted residuals, priors included) {costs.min():.1f}; "
    f"the five best starts: {np.round(np.sort(costs)[:5], 1)}"
)
for i, name in enumerate(PROJECTOR):
    print(
        f"{name}: z {float(numbers['lens_z'][i]) / MM:5.0f} mm "
        f"(drawing {DRAWING[name] * 1e3:3.0f}), "
        f"gap {float(numbers['pole_gap'][i]) / MM:5.1f} mm, "
        f"bore radius {float(numbers['bore_radius'][i]) / MM:5.1f} mm"
    )
print(
    f"objective: focal length {float(numbers['objective_f']) / MM:.2f} mm "
    f"at {float(numbers['objective_z']) / MM:.2f} mm"
)
start = 0
for mode in modes:
    n = 2 * len(mode["labels"])
    size, turn = r[start : start + n].reshape(-1, 2), r[start + n : start + 2 * n].reshape(-1, 2)
    print(f"{mode['name']}: ratio misses for the x and y kicks, % in size and mrad in turn:")
    for label, s, t in zip(mode["labels"], size, turn):
        print(
            f"   {label:10s} {np.round(s * RATIO_ERROR * 100, 2)} %   "
            f"{np.round(t * RATIO_ERROR * 1000, 2)} mrad"
        )
    start += 2 * n + 1

a, b = jax.vmap(lambda dac: position_and_angle_response(projector(numbers, dac), voltage=VOLTAGE))(
    MAG_DAC
)
print("MAG, model magnification / label:", np.round(np.asarray(jnp.abs(a)) / MAG_LABEL, 3))
a, b = jax.vmap(lambda dac: position_and_angle_response(projector(numbers, dac), voltage=VOLTAGE))(
    DIFF_DAC
)
print("DIFF, model camera length / label:", np.round(np.asarray(jnp.abs(b)) / DIFF_LABEL, 3))

(HERE / "projector_lenses.json").write_text(
    json.dumps(
        dict(
            source="fit_projector_lenses.py: PolePieceLens fit to "
            "imaging-system-fitting-40k-40cm.zip (2026-11-10) "
            "and to the mean scale and MAG focus of console_tables.json (2026-11-17)",
            units="lengths in L, the distance from the specimen to the SightX",
            misfit=float(costs.min()),
            misfit_definition=f"sum of squares of {r.size} residuals in units of their allowed "
            "errors, priors included",
            ampere_turns_per_dac=dict(zip(PROJECTOR, AMPERE_TURNS_PER_DAC.tolist())),
            lens_z=dict(zip(PROJECTOR, np.asarray(numbers["lens_z"]).tolist())),
            pole_gap=dict(zip(PROJECTOR, np.asarray(numbers["pole_gap"]).tolist())),
            bore_radius=dict(zip(PROJECTOR, np.asarray(numbers["bore_radius"]).tolist())),
            objective_f=float(numbers["objective_f"]),
            objective_z=float(numbers["objective_z"]),
        ),
        indent=1,
    )
)
print("wrote projector_lenses.json")

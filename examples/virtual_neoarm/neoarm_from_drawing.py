# %% [markdown]
# # The NEOARM from its drawing and its lens table
#
# A first TemGym model of the NEOARM's imaging system built from what we know rather than invented:
#
# * **Where the lenses are**: read off JEOL's column diagram (`neoarm_column_diagram.json`). Each lens may move up
#   to 60 mm from the drawing, because the drawing was read by eye and its scale rests on one guess.
# * **How strong the lenses are**: fitted so that every row of the real MAG lens table (8000x to 2,000,000x,
#   `neoarm_lens_table.json`) puts the specimen in focus on the screen at the row's label, and the one real DIFF row
#   we have (40 cm, recorded 2026-11-10) gives a 40 cm camera length.
# * **Ampere-turns per DAC unit**: 0.080 for IL1-PL1, measured from image rotation on 2026-11-10.
#
# Every lens is a TemGym `ElectromagneticLens`. This is provisional: the fit leaves the lens strengths poorly pinned
# down (see the last cell).
#
# Run from this folder:  python neoarm_from_drawing.py   (add --refit to redo the fit, about 10 minutes)

# %%
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.ticker
import numpy as np
from scipy.optimize import least_squares

jax.config.update("jax_enable_x64", True)
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402

from temgym_core.components import ElectromagneticLens, Plane                    # noqa: E402
from temgym_core.constants import effective_accelerating_potential              # noqa: E402
from temgym_core.plotting import legacy_beam_plot_params, plot_model            # noqa: E402
from temgym_core.ray import Ray                                                   # noqa: E402
from temgym_core.run import position_and_angle_response                         # noqa: E402

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
VOLTAGE = 200e3
V_STAR = float(effective_accelerating_potential(VOLTAGE))
LENSES = ("IL1", "IL2", "IL3", "PL1")
AMPERE_TURNS_PER_DAC = 0.080
WINDING = np.array([-1.0, -1.0, -1.0, 1.0])            # PL1 turns the image the other way (measured)
MOVE_FROM_DRAWING_M = 0.060

# %% [markdown]
# ## 1. What we know

# %%
drawing = json.loads((HERE / "neoarm_column_diagram.json").read_text())["z_m"]
DRAWING_Z = np.array([drawing[n] for n in LENSES] + [drawing["Screen"]])
raw = json.loads((HERE / "neoarm_lens_table.json").read_text())["raw_dac_hex"]
table = {int(m): {k: int(v, 16) for k, v in row.items()} for m, row in raw.items()}
MAGS = [m for m in sorted(table) if m >= 8000]          # below 8000 the objective changes too (low-mag mode)
MAG_DAC = np.array([[table[m][n] for n in LENSES] for m in MAGS], float)
DIFF_40CM_DAC = np.array([20771, 48739, 32605, 64000], float)   # the console's DIFF 40 cm row, 2026-11-10

# The measured magnification curve: beam shift on the camera per CLA1 unit at each MAG row (2026-11-10). Only its
# ratios mean anything (a CLA1 unit's size at the specimen is not known), so it is scaled to the 40k row.
recording = HERE.parent / "microscope_inversion/recordings/2026-11-10/beam_per_magnification.zip"
with zipfile.ZipFile(recording) as z:
    rows = list(csv.DictReader(io.StringIO(
        z.read("output/beam_per_magnification_neoarm/2026-11-10_164922/table.csv").decode())))
measured = {int(r["magnification"]): np.sqrt(abs(np.linalg.det(np.array(json.loads(r["px_per_cla1_unit"])))))
            for r in rows}
measured_relative = {m: v / measured[40000] * 40000 for m, v in measured.items()}
print("lens positions on the drawing (mm below the specimen):",
      {n: round(1e3 * z, 1) for n, z in zip(LENSES + ("screen",), DRAWING_Z)})

# %% [markdown]
# ## 2. The column
#
# Numbers of the model, all in one flat list `p`:
# objective focal length (log), objective magnification (log), shift of IL1, IL2, IL3, PL1 and the screen from the
# drawing (m), each lens's strength Gc (log), one overall scale between the console's labels and the true
# magnification on the screen, each lens's thickness per ampere-turn Tc (log).


# %%
def column(p, dac):
    """TemGym components from the specimen (z = 0) to the screen, at these IL1, IL2, IL3, PL1 DAC values."""
    f_objective, m_objective = jnp.exp(p[0]), jnp.exp(p[1])
    z = jnp.asarray(DRAWING_Z) + p[2:7]
    parts = [ElectromagneticLens(z=f_objective * (1 + 1 / m_objective), turns=1.0, current=1.0,
                                 Gc=V_STAR / f_objective)]
    for i in range(4):
        parts.append(ElectromagneticLens(z=z[i], turns=WINDING[i] * AMPERE_TURNS_PER_DAC, current=dac[i],
                                         Gc=jnp.exp(p[7 + i]), Tc=jnp.exp(p[12 + i])))
    return parts + [Plane(z=z[4])]


def magnification_and_camera_length(p, dac):
    a, b = position_and_angle_response(column(p, dac), voltage=VOLTAGE)
    return jnp.abs(a), jnp.abs(b)


def misfit(p):
    """Every MAG row: magnification at the label (3 %) and in focus (1 um at the specimen). DIFF 40 cm: camera
    length 0.40 m (5 %), and a shift at the specimen does not move the pattern. Lens shifts: about 30 mm."""
    out = []
    for label, dac in zip(MAGS, MAG_DAC):
        a, b = magnification_and_camera_length(p, jnp.asarray(dac))
        out += [(jnp.log(a) - jnp.log(p[11] * label)) / 0.03, (b / a) / 1e-6]
    a, b = magnification_and_camera_length(p, jnp.asarray(DIFF_40CM_DAC))
    out += [(jnp.log(b) - jnp.log(p[11] * 0.40)) / 0.05, a / 20.0]
    out += list(p[2:7] / 0.03)
    return jnp.stack(out)


# %% [markdown]
# ## 3. The fit (from many random starts; the result is kept in neoarm_from_drawing_fit.json)

# %%
FIT_FILE = HERE / "neoarm_from_drawing_fit.json"


def strength_for(focal_length, dac):
    return np.log(V_STAR / (focal_length * (AMPERE_TURNS_PER_DAC * dac) ** 2))


if "--refit" in sys.argv or not FIT_FILE.exists():
    residual, slope = jax.jit(misfit), jax.jit(jax.jacfwd(misfit))
    low = np.r_[np.log(1e-3), np.log(5), [-MOVE_FROM_DRAWING_M] * 5, [-60] * 4, 0.1, [np.log(1e-8)] * 4]
    high = np.r_[np.log(5e-3), np.log(200), [MOVE_FROM_DRAWING_M] * 5, [60] * 4, 10.0, [np.log(2e-5)] * 4]
    rng, best = np.random.default_rng(3), None
    for start in range(100):
        p0 = np.r_[np.log(2e-3 * rng.uniform(0.7, 1.4)), np.log(rng.uniform(20, 80)),
                   rng.uniform(-0.03, 0.03, 5), [strength_for(rng.uniform(0.005, 0.1), 40000) for _ in range(3)],
                   strength_for(rng.uniform(0.002, 0.05), 64000), np.exp(rng.uniform(-1.5, 1.5)),
                   np.log(rng.uniform(1e-7, 1e-5, 4))]
        p0 = np.clip(p0, low + 1e-9, high - 1e-9)
        try:
            result = least_squares(lambda q: np.asarray(residual(q)), p0, jac=lambda q: np.asarray(slope(q)),
                                   bounds=(low, high), max_nfev=400)
        except Exception:
            continue
        if np.isfinite(result.cost) and (best is None or result.cost < best.cost):
            best = result
            print(f"   start {start}: misfit {2 * result.cost:.0f}")
    FIT_FILE.write_text(json.dumps(dict(p=best.x.tolist(), chi2=float(2 * best.cost)), indent=1))
fit = json.loads(FIT_FILE.read_text())
P = np.array(fit["p"])

# %% [markdown]
# ## 4. What the fit says

# %%
z_fit = DRAWING_Z + P[2:7]
row_40k = MAG_DAC[MAGS.index(40000)]
print(f"objective: focal length {np.exp(P[0]) * 1e3:.2f} mm, magnification {np.exp(P[1]):.0f}x;  "
      f"true magnification on the screen = {P[11]:.2f} x the console's label")
print(f"{'':6s}{'drawing':>9s}{'model':>9s}{'moved':>8s}   at the 40k row")
for i, name in enumerate(LENSES + ("screen",)):
    line = f"{name:6s}{DRAWING_Z[i] * 1e3:8.0f}mm{z_fit[i] * 1e3:7.0f}mm{P[2 + i] * 1e3:+7.0f}mm"
    if i < 4:
        at = AMPERE_TURNS_PER_DAC * row_40k[i]
        f = V_STAR / (np.exp(P[7 + i]) * at ** 2)
        half_width = 0.0055 / np.exp(P[7 + i])          # bell-shaped field: Gc ~ 0.0055 / half-width (weak lens)
        line += (f"   focal length {f * 1e3:6.2f} mm, thickness {np.exp(P[12 + i]) * at * 1e3:5.1f} mm, "
                 f"field half-width ~{half_width * 1e3:4.1f} mm")
    print(line)
model_mag = np.array([float(magnification_and_camera_length(P, jnp.asarray(d))[0]) / P[11] for d in MAG_DAC])
defocus_um = np.array([float(np.divide(*magnification_and_camera_length(P, jnp.asarray(d))[::-1])) * 1e6
                       for d in MAG_DAC])
_, camera_length = magnification_and_camera_length(P, jnp.asarray(DIFF_40CM_DAC))
print(f"DIFF 40 cm row: camera length {float(camera_length) / P[11] * 100:.1f} cm (in label units)")

# %% [markdown]
# ## 5. The column, drawn by TemGym
#
# The same parallel beam, 1 um across, crosses the specimen in both panels. MAG 40k (left) makes an image of it on
# the screen, so the beam ends up wide; DIFF 40 cm (right) brings every ray that leaves the specimen at the same angle
# to one point, so the beam ends up as a spot (the direct beam of a diffraction pattern). Positions are in the frame
# that turns with the lenses.

# %%
LABELS = ["objective", *LENSES, "screen"]
PARAMS = legacy_beam_plot_params(figsize=(5.0, 9.0))
millimetres = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1e3:g}")


def parallel_beam(half_width=0.5e-6, angle=0.0, n=11):
    x = jnp.linspace(-half_width, half_width, n)
    return Ray(x=x, y=jnp.zeros(n), dx=jnp.full(n, angle), dy=jnp.zeros(n), z=jnp.zeros(n),
               pathlength=jnp.zeros(n), voltage=jnp.full(n, VOLTAGE))


def draw(ax, dac, title):
    plot_model(column(P, jnp.asarray(dac)), rays=parallel_beam(), ax=ax, plot_params=PARAMS,
               ray_coordinate="x_rot", component_labels=LABELS)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(millimetres)
    ax.yaxis.set_major_formatter(millimetres)
    ax.set_xlabel("distance from the axis (mm)")
    ax.set_ylabel("distance below the specimen (mm)")


fig, axes = plt.subplots(1, 2, figsize=(11, 9))
draw(axes[0], row_40k, "MAG 40k: an image of the beam")
draw(axes[1], DIFF_40CM_DAC, "DIFF 40 cm: the beam brought to a spot")
fig.tight_layout()
fig.savefig(FIGURES / "column_mag40k_diff40cm.png", dpi=130)

# %% [markdown]
# The same beam at four MAG rows: watch the intermediate lenses take over from one another.

# %%
SHOWN = [8000, 40000, 300000, 1200000]
fig, axes = plt.subplots(1, 4, figsize=(18, 9))
for ax, label in zip(axes, SHOWN):
    draw(ax, MAG_DAC[MAGS.index(label)], f"MAG {label:,}x")
fig.tight_layout()
fig.savefig(FIGURES / "column_four_magnifications.png", dpi=110)

# %% [markdown]
# ## 6. The curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
ax = axes[0, 0]
for i, name in enumerate(LENSES):
    ax.semilogx(MAGS, MAG_DAC[:, i], "o-", ms=3, label=name)
ax.set(title="The real lens table (console)", xlabel="magnification label", ylabel="DAC value")
ax.legend()

ax = axes[0, 1]
ax.loglog(MAGS, MAGS, "k--", lw=1, label="label")
ax.loglog(MAGS, model_mag, "o-", ms=4, label="model (in label units)")
ax.loglog(sorted(measured_relative), [measured_relative[m] for m in sorted(measured_relative)], "s", ms=4,
          mfc="none", label="measured 2026-11-10 (scaled to 40k)")
ax.set(title="Magnification", xlabel="magnification label", ylabel="magnification")
ax.legend()

ax = axes[1, 0]
ax.semilogx(MAGS, 100 * (model_mag / np.array(MAGS) - 1), "o-", ms=4, label="model")
ax.semilogx(sorted(measured_relative), [100 * (measured_relative[m] / m - 1) for m in sorted(measured_relative)],
            "s", ms=4, mfc="none", label="measured")
ax.axhline(0, color="k", lw=0.8)
ax.set(title="Off the label by", xlabel="magnification label", ylabel="%")
ax.legend()

ax = axes[1, 1]
for i, name in enumerate(LENSES):
    at = AMPERE_TURNS_PER_DAC * MAG_DAC[:, i]
    ax.loglog(MAGS, 1e3 * V_STAR / (np.exp(P[7 + i]) * at ** 2), "o-", ms=3, label=name)
ax.set(title="Model focal lengths along the table", xlabel="magnification label", ylabel="focal length (mm)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "magnification_curves.png", dpi=120)
print("figures in", FIGURES)

# %% [markdown]
# ## 7. How far to trust it
#
# * The fit puts every row in focus (to a few um at the specimen) but leaves the magnification 5-20 % off the
#   label on many rows, against a measured 3 %; the 800k row is far off (the console probably switches lens
#   arrangement there).
# * Different random starts reach similar misfits with very different lens strengths: the table alone does not
#   pin them down. Compare each lens's field half-width with the pole pieces on the drawing: a few mm is physical,
#   a fraction of a mm is not.
# * There is only one DIFF row, so there is no diffraction curve yet.

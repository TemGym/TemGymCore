"""Design the virtual NEOARM's TEM-mode condenser and write it to condenser.json.

Not fitted to any recording: a plausible condenser that works the way the NEOARM's should.

    source -> CL1 -> CL3 -> condenser aperture -> CLA1 -> CLA2 -> probe corrector -> CM -> objective front field -> specimen

How it works (TEM mode)
-----------------------
* CL1 (spot size) images the source to a first crossover. A stronger CL1 gives a smaller crossover closer
  to CL1, from which the beam spreads faster, so the aperture lets less of it through: a smaller spot and
  less current.
* CL3 (brightness) images the first crossover to a second one. The spot table changes CL3 with CL1 so the
  second crossover stays in the same plane when the spot size changes.
* CM and the objective's front field are fixed. The front field's back focal plane is (almost) the
  specimen, so:
    - focused on the specimen: the second crossover sits in CM's front focal plane. CM sends the beam on
      parallel and the front field focuses it on the specimen;
    - parallel at the specimen: the second crossover sits where CM images it to the front field's front
      focal plane. Every source point then gives a parallel bundle at the specimen.
  Turning brightness (CL3) moves the second crossover between and beyond these two planes.
* TEM mode: CM (30 mm focal length, 40 mm above the specimen) is chosen so that, with the 40 um hole, the
  focused beam converges by about 2 mrad and the parallel beam is about 7 um across, with brightness running
  about 12 000 DAC units from focused to parallel. A stronger CM gives smaller angles but a smaller parallel
  beam; a weaker one the reverse.
* The aperture sits between CL3 and its crossover. Its hole limits the spread of angles when focused and
  the illuminated area when parallel.
* The source stands for the gun: 10 nm rms (a cold field emitter's source, enlarged by the gun lens), its
  spread of angles chosen so the beam at the aperture is BEAM_AT_APERTURE_RMS (so the holes cut it), and its
  current from a cold field emitter's brightness. Brightness ties current, focused size and convergence
  together: with nA and mrad, the focused spot is about a nm. Real focused spots are larger through
  aberrations, which are not modelled.
* The probe corrector is only its round optics, a transfer that turns the beam over and changes nothing else
  (magnification -1). The design is done as if it were not there; then everything above it moves up by its
  length, CORRECTOR_LENGTH, which leaves the optics the same.

Which numbers are chosen and which come from the console
--------------------------------------------------------
Chosen (plausible, not measured): every plane, the corrector's length, the source, the holes, CM's strength, and the
first crossovers per spot.
From the NEOARM's console (2026-11-10, 200 kV, spot 3): CL1 31232 and CM 26624, the DAC values CL1's and CM's
strengths are set at; CL3 33400, where the console's beam crossed over at the specimen, which sets CL3's strength;
0.080 ampere-turns per DAC unit, measured for the projection lenses from image rotation and assumed for
the condenser lenses too. The objective's front field is the fitted objective, mirrored.

Run from this folder:  python design_condenser.py
"""

import json
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from temgym_core.components import ElectromagneticLens, Plane                    # noqa: E402
from temgym_core.constants import effective_accelerating_potential              # noqa: E402
from temgym_core.run import column_matrix                                       # noqa: E402

HERE = Path(__file__).parent
VOLTAGE = 200e3
AMPERE_TURNS_PER_DAC = 0.080

# Planes (m from the specimen) as if there were no corrector. Lens positions are not measured: a plausible layout.
SOURCE_Z, CL1_Z, CL3_Z, APERTURE_Z, CLA1_Z, CLA2_Z, CM_Z = -0.55, -0.45, -0.28, -0.25, -0.17, -0.14, -0.04
CM_FOCAL_LENGTH = 30e-3                                      # TEM mode
SOURCE_RMS_M, BEAM_AT_APERTURE_RMS = 10e-9, 40e-6
CORRECTOR_EXIT_Z, CORRECTOR_LENGTH = -0.10, 0.25             # the probe corrector ends 0.10 m above the specimen


def built_at(z):
    """Where a plane of the design really is: above the corrector, everything moves up by its length."""
    return z - CORRECTOR_LENGTH if z < CORRECTOR_EXIT_Z else z
BRIGHTNESS = 1e8 * VOLTAGE                                    # A / (m^2 sr): a cold field emitter, 1e8 A/(m^2 sr V)
HOLE_DIAMETERS_M = (150e-6, 70e-6, 40e-6, 10e-6)            # holes 1-4
APERTURE_M_PER_DAC, APERTURE_EDGE_M = 0.1e-6, 0.5e-6

# The console's spot 3 (spot-size index 2), CM and the CL3 that focuses on the specimen
SPOT_3, CL1_AT_SPOT_3, CM_DAC, CL3_FOCUSED = 2, 31232, 26624, 33400
CL3_AT_SPOT_3 = 33866                                         # the console's brightness at spot 3

# Spot sizes 1-8 (PyJEM indices 0-7). CL1 images the source to the first crossover, shrunk by SHRINK. Each spot shrinks
# it by SPOT_STEP more: the focused spot gets SPOT_STEP smaller, and the beam leaves the crossover 1 / SPOT_STEP faster,
# so a hole the beam overfills passes SPOT_STEP^2 of the current, half of it per spot.
SPOTS = range(8)
SHRINK_AT_SPOT_3, SPOT_STEP = 0.5, 2 ** -0.5
SHRINK = {spot: SHRINK_AT_SPOT_3 * SPOT_STEP ** (spot - SPOT_3) for spot in SPOTS}
FIRST_CROSSOVER_Z = {spot: CL1_Z + SHRINK[spot] * (CL1_Z - SOURCE_Z) for spot in SPOTS}

V_STAR = effective_accelerating_potential(VOLTAGE)


def focal(u, v):
    """Thin lens: the focal length imaging a plane u before it to v after it."""
    return 1 / (1 / u + 1 / v)


def image_distance(u, f):
    """Thin lens: where a plane u before it is imaged (negative: a virtual image before the lens)."""
    return 1 / (1 / f - 1 / u)


def object_distance(v, f):
    """Thin lens: the plane before it imaged to v after it (v negative: a virtual image before the lens)."""
    return 1 / (1 / f - 1 / v)


def Gc_for(f, dac):
    """ElectromagneticLens strength giving focal length f at this DAC value (f = V* / (Gc (NI)^2))."""
    return V_STAR / (f * (AMPERE_TURNS_PER_DAC * dac) ** 2)


def dac_for(f, Gc):
    return np.sqrt(V_STAR / (Gc * f)) / AMPERE_TURNS_PER_DAC


# The objective's front field: the fitted objective, mirrored to the source side
fitted = json.loads((HERE / "virtual_neoarm.json").read_text())["lenses"]["objective"]
front = ElectromagneticLens(z=-fitted["z_m"], turns=1.0, current=fitted["ampere_turns"], Gc=fitted["Gc"])
front_f = float(front.focal_length(VOLTAGE))
front_focal_plane = -fitted["z_m"] - front_f

# CL1: images the source to the first crossover at spot 3
Gc_CL1 = Gc_for(focal(CL1_Z - SOURCE_Z, FIRST_CROSSOVER_Z[SPOT_3] - CL1_Z), CL1_AT_SPOT_3)

# CM: strong (TEM mode). The beam is parallel when the second crossover is at PARALLEL_Z, the plane CM images to the
# front field's front focal plane.
f_CM = CM_FOCAL_LENGTH
Gc_CM = Gc_for(f_CM, CM_DAC)
PARALLEL_Z = CM_Z - object_distance(front_focal_plane - CM_Z, f_CM)
# Focused: CM images the second crossover to the plane the front field images onto the specimen. The
# specimen is almost at the front field's back focal plane, so that plane is far away and the second
# crossover is almost at CM's front focal plane.
front_object_z = -fitted["z_m"] - object_distance(fitted["z_m"], front_f)      # imaged onto the specimen
FOCUSED_Z = CM_Z - object_distance(front_object_z - CM_Z, f_CM)

# CL3: puts the second crossover at CM's front focal plane at CL3_FOCUSED (spot 3)
Gc_CL3 = Gc_for(focal(CL3_Z - FIRST_CROSSOVER_Z[SPOT_3], FOCUSED_Z - CL3_Z), CL3_FOCUSED)

# The spot table: CL1 for each first crossover; CL3 keeps the second crossover where spot 3's console
# brightness puts it
second_z = CL3_Z + image_distance(CL3_Z - FIRST_CROSSOVER_Z[SPOT_3],
                                  V_STAR / (Gc_CL3 * (AMPERE_TURNS_PER_DAC * CL3_AT_SPOT_3) ** 2))
spot_table = {}
for spot, first_z in FIRST_CROSSOVER_Z.items():
    cl1 = dac_for(focal(CL1_Z - SOURCE_Z, first_z - CL1_Z), Gc_CL1)
    cl3 = dac_for(focal(CL3_Z - first_z, second_z - CL3_Z), Gc_CL3)
    spot_table[spot] = {"CL1": int(round(cl1)), "CL3": int(round(cl3))}


def lens(z, dac, Gc):
    return ElectromagneticLens(z=z, turns=AMPERE_TURNS_PER_DAC, current=float(dac), Gc=Gc)


def matrix(cl1, cl3, end_z):
    """TemGym's ray matrix from the source to the plane end_z, as sizes: how far (and at what angle) a ray ends per
    unit of position and of angle at the source. The lenses turn the beam about the axis, so each entry is the
    length of the turned (x, y) pair; the optics are round, so that is all there is."""
    column = [lens(CL1_Z, cl1, Gc_CL1), lens(CL3_Z, cl3, Gc_CL3), lens(CM_Z, CM_DAC, Gc_CM), front]
    column = [part for part in column if part.z < end_z] + [Plane(z=end_z)]
    m = np.asarray(column_matrix(column, source_z=SOURCE_Z, voltage=VOLTAGE))
    return np.array([[np.hypot(m[0, 0], m[1, 0]), np.hypot(m[0, 2], m[1, 2])],
                     [np.hypot(m[2, 0], m[3, 0]), np.hypot(m[2, 2], m[3, 2])]])


def rms(m):
    """Rms size and spread of angles after the ray matrix m (no aperture)."""
    c = m @ np.diag([SOURCE_RMS_M ** 2, SOURCE_RMS_ANGLE_RAD ** 2]) @ m.T
    return np.sqrt(c[0, 0]), np.sqrt(c[1, 1])


def at_specimen(cl1, cl3):
    """Rms size and spread of angles at the specimen (no aperture)."""
    return rms(matrix(cl1, cl3, 0.0))


def cl3_for(spot, second_crossover_z):
    """CL3 putting the second crossover at this plane, at this spot."""
    return dac_for(focal(CL3_Z - FIRST_CROSSOVER_Z[spot], second_crossover_z - CL3_Z), Gc_CL3)


cl3_parallel = cl3_for(SPOT_3, PARALLEL_Z)

# The source: its spread of angles from the beam at the aperture, its current from the brightness (rms: current =
# brightness x pi^2 x size^2 x spread^2)
to_aperture = matrix(CL1_AT_SPOT_3, CL3_FOCUSED, APERTURE_Z)
SOURCE_RMS_ANGLE_RAD = np.sqrt(BEAM_AT_APERTURE_RMS ** 2 - (to_aperture[0, 0] * SOURCE_RMS_M) ** 2) / abs(to_aperture[0, 1])
SOURCE_CURRENT_A = BRIGHTNESS * np.pi ** 2 * SOURCE_RMS_M ** 2 * SOURCE_RMS_ANGLE_RAD ** 2
print(f"source: rms size {SOURCE_RMS_M * 1e9:.0f} nm, rms spread of angles {SOURCE_RMS_ANGLE_RAD * 1e3:.4f} mrad, "
      f"current {SOURCE_CURRENT_A * 1e9:.1f} nA")
print(f"objective front field: f = {front_f * 1e3:.3f} mm, front focal plane at {front_focal_plane * 1e3:.3f} mm")
print(f"second crossover: focused at {FOCUSED_Z:.4f} m, parallel at {PARALLEL_Z:.4f} m, console (spot 3) at {second_z:.4f} m")
for name, cl3 in [("focused", CL3_FOCUSED), ("console", CL3_AT_SPOT_3), ("parallel", cl3_parallel)]:
    size, spread = at_specimen(CL1_AT_SPOT_3, cl3)
    print(f"  {name:9s} CL3 {cl3:7.0f}: rms size {size * 1e9:8.1f} nm, rms spread of angles {spread * 1e3:7.3f} mrad")
print("spot table (index: CL1, CL3):", spot_table)

condenser = {
    "description": "The virtual NEOARM's TEM-mode condenser, designed by design_condenser.py (not fitted). "
                   "Planes in m from the specimen; lenses are TemGym ElectromagneticLens.",
    "reference_voltage_v": VOLTAGE,
    "source": {"z_m": built_at(SOURCE_Z), "rms_size_m": SOURCE_RMS_M, "rms_angle_rad": SOURCE_RMS_ANGLE_RAD,
               "current_a": SOURCE_CURRENT_A},
    "lenses": {name: {"z_m": built_at(z), "ampere_turns_per_dac": AMPERE_TURNS_PER_DAC, "Gc": Gc, "Tc": 0.0}
               for name, z, Gc in [("CL1", CL1_Z, Gc_CL1), ("CL3", CL3_Z, Gc_CL3), ("CM", CM_Z, Gc_CM)]},
    "aperture": {"z_m": built_at(APERTURE_Z), "hole_diameters_m": list(HOLE_DIAMETERS_M), "m_per_dac": APERTURE_M_PER_DAC,
                 "edge_m": APERTURE_EDGE_M},
    "deflectors_z_m": [built_at(CLA1_Z), built_at(CLA2_Z)],
    "corrector": {"entrance_z_m": CORRECTOR_EXIT_Z - CORRECTOR_LENGTH, "exit_z_m": CORRECTOR_EXIT_Z},
    "spot_table": spot_table,
    "cm_dac": CM_DAC,
    "design": {"as_if_no_corrector": "planes below are as if there were no corrector",
               "first_crossover_z_m": FIRST_CROSSOVER_Z, "second_crossover_z_m": {
        "focused": FOCUSED_Z, "parallel": PARALLEL_Z, "console_spot_3": second_z},
        "CL3_parallel_at_spot_3": cl3_parallel},
}
(HERE / "condenser.json").write_text(json.dumps(condenser, indent=1))
print("written", HERE / "condenser.json")


# ---------------------------------------------------------------------------------------------------------------------
# The design curves: per spot, the DAC values, the current through each hole, the focused spot, and what each hole
# gives at the specimen: the convergence semi-angle when focused and the illuminated diameter when parallel. Rays from
# the middle of the source pass the aperture at N[0, 1] x their angle at the source, and reach the specimen at
# M[0, 1] x that angle with angle M[1, 1] x it, so a hole of radius R gives R M[1, 1] / N[0, 1] (focused) and
# 2 R M[0, 1] / N[0, 1] (parallel).
# ---------------------------------------------------------------------------------------------------------------------
curves = {"spot": [], "CL1": [], "CL3": [], "CL3_focused": [], "CL3_parallel": [], "current_a": [], "focused_rms_m": [],
          "convergence_rad": [], "illuminated_diameter_m": []}
for spot in SPOTS:
    cl1 = spot_table[spot]["CL1"]
    focused, parallel = cl3_for(spot, FOCUSED_Z), cl3_for(spot, PARALLEL_Z)
    at_aperture = matrix(cl1, spot_table[spot]["CL3"], APERTURE_Z)      # CL3 is before the aperture: any CL3 will do
    beam_at_aperture = rms(at_aperture)[0]
    radii = np.array(HOLE_DIAMETERS_M) / 2
    to_focus, to_parallel = matrix(cl1, focused, 0.0), matrix(cl1, parallel, 0.0)
    curves["spot"].append(spot + 1)
    curves["CL1"].append(cl1)
    curves["CL3"].append(spot_table[spot]["CL3"])
    curves["CL3_focused"].append(focused)
    curves["CL3_parallel"].append(parallel)
    curves["current_a"].append(SOURCE_CURRENT_A * (1 - np.exp(-radii ** 2 / (2 * beam_at_aperture ** 2))))
    curves["focused_rms_m"].append(rms(to_focus)[0])
    curves["convergence_rad"].append(radii * abs(to_focus[1, 1] / at_aperture[0, 1]))
    curves["illuminated_diameter_m"].append(2 * radii * abs(to_parallel[0, 1] / at_aperture[0, 1]))
curves = {k: np.array(v) for k, v in curves.items()}

holes = [f"{d * 1e6:.0f} um" for d in HOLE_DIAMETERS_M]
print(f"\n{'spot':>4} {'CL1':>6} {'CL3':>6} {'CL3 focus':>9} {'CL3 par.':>9} {'focused':>9}   "
      + "  ".join(f"{'current ' + h:>14}" for h in holes))
for i in range(len(SPOTS)):
    print(f"{curves['spot'][i]:4d} {curves['CL1'][i]:6d} {curves['CL3'][i]:6d} {curves['CL3_focused'][i]:9.0f} "
          f"{curves['CL3_parallel'][i]:9.0f} {curves['focused_rms_m'][i] * 1e9:7.2f}nm   "
          + "  ".join(f"{c * 1e9:12.3f}nA" for c in curves["current_a"][i]))
print("\nper hole (the same at every spot when the beam overfills it):")
for h, a, d in zip(holes, curves["convergence_rad"][SPOT_3], curves["illuminated_diameter_m"][SPOT_3]):
    print(f"   {h:>6}: convergence semi-angle {a * 1e3:6.2f} mrad when focused; illuminated diameter {d * 1e6:6.2f} um when parallel")

# ---------------------------------------------------------------------------------------------------------------------
# The figure: condenser_design.png
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib                                                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                     # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6E6E6E"
HOLE_STYLE = [(BLUE, "o"), (ORANGE, "s"), (GREEN, "^"), (RED, "D")]
spots = curves["spot"]
figure = plt.figure(figsize=(7.6, 5.0), constrained_layout=True)
grid = figure.add_gridspec(2, 6)
axes = [figure.add_subplot(grid[0, 0:2]), figure.add_subplot(grid[0, 2:4]), figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 0:3]), figure.add_subplot(grid[1, 3:6])]

ax = axes[0]
for key, colour, marker, style, label in [("CL1", BLUE, "o", "-", "CL1"), ("CL3", ORANGE, "s", "-", "CL3 (tracks CL1)"),
                                          ("CL3_focused", GREY, None, "--", "CL3 focused"),
                                          ("CL3_parallel", GREY, None, ":", "CL3 parallel")]:
    ax.plot(spots, curves[key], style, color=colour, marker=marker, ms=4, lw=1.5, label=label)
ax.set(xlabel="spot size", ylabel="DAC value", title="(a) lens values per spot", xticks=spots)
ax.legend(frameon=False, fontsize=6.5)

ax = axes[1]
for i, (h, (colour, marker)) in enumerate(zip(holes, HOLE_STYLE)):
    ax.semilogy(spots, curves["current_a"][:, i] * 1e9, "-", color=colour, marker=marker, ms=4, lw=1.5)
    ax.annotate(h, (spots[-1], curves["current_a"][-1, i] * 1e9), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=6.5, color="black")
ax.set(xlabel="spot size", ylabel="current at the specimen (nA)", title="(b) current through each hole",
       xlim=(spots[0] - 0.3, spots[-1] + 2.2), xticks=spots)

ax = axes[2]
ax.semilogy(spots, SOURCE_RMS_M * np.array([SHRINK[s] for s in SPOTS]) * 1e9, "-s", color=ORANGE, ms=4, lw=1.5,
            label="crossover, CL1 to CL3")
ax.semilogy(spots, curves["focused_rms_m"] * 1e9, "-o", color=BLUE, ms=4, lw=1.5, label="focused on specimen")
ax.legend(frameon=False, fontsize=6.5, loc="lower left")
ax.set(xlabel="spot size", ylabel="rms size, no aberrations (nm)", title="(c) spot size", xticks=spots)

for ax, key, scale, unit, title in [(axes[3], "convergence_rad", 1e3, "mrad", "(d) convergence semi-angle, focused"),
                                    (axes[4], "illuminated_diameter_m", 1e6, "um", "(e) illuminated diameter, parallel")]:
    values = curves[key][SPOT_3] * scale
    ax.bar(range(len(holes)), values, 0.6, color=[c for c, _ in HOLE_STYLE])
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3g}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(holes)), holes)
    ax.set(xlabel="condenser aperture hole", ylabel=unit, title=title + " (any spot)")
figure.savefig(HERE / "condenser_design.png")
print("written", HERE / "condenser_design.png")

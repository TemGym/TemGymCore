"""A simulated NEOARM with the model of neoarm_model.ipynb behind PyJEM's names.

The hidden optics are the column of notebook 1. They hold the condenser of condenser.json and the
projector of projector_lenses.json, as PolePieceLens components. The console's own tables of
2026-11-17 drive it, so selecting a magnification or a spot size sets the lenses as the real console
does. The spot table is that of alpha 1, the alpha of the condenser fit. The SightX is the camera.

A script calls install_simulated_neoarm() once, before it imports PyJEM. After that, `import PyJEM`
finds this microscope, so the rest of the script is the same as at the real microscope:

    from simulated_neoarm import install_simulated_neoarm
    install_simulated_neoarm()
    from PyJEM import TEM3, detector, sightx

Run this file to poll every value that jemwrap reads, once, and print jemwrap's state table.

The hidden microscope is not notebook 1. HIDDEN below changes notebook 1's numbers by about as
much as notebook 1 may miss the real NEOARM. It moves the lenses and the corrector and changes
their coils. It changes the gaps and bores of the lenses and puts their centres off the axis. It
moves the objective and changes its focus. It gives the deflectors their own positions, strengths
and directions. Some parts have another form than the model's. The lenses lag behind their DAC
values (hysteresis), and the objective's post-field is shorter than its pre-field. The corrector is
a transfer block with a magnification of -1 and no lenses (requirement M2, 2026-11-20). A fit
that starts from notebook 1, as at the real NEOARM, need not find these numbers.
It must predict the beam for settings that it never saw. Notebook 2 does not read HIDDEN.

Limits of this simulated NEOARM:
- The condenser follows the spot table of alpha 1, with its spot alignment SpotA. The alpha
  selector accepts a new value but changes nothing.
- simulated_neoarm() sets the aligned deflector values and the objective after it makes the
  microscope. SimulatedMicroscope.reset() puts CLA1 and CLA2 back to neutral, SpotA to the spot
  table's value, and the selector to the first MAG row.

Along the axis, lengths are in units of L, the distance from the specimen to the SightX. Across
the axis, positions are in metres, so a slope is in metres per L.
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from temgym_core.components import (
    ABCDTransfer,
    CrossGrating,
    PolycrystallineFilm,
    Deflector,
    Lens,
    Plane,
    PolePieceLens,
    Rotator,
    Scanner,
    SigmoidAperture,
)
from temgym_core.constants import magnetic_strength_ratio
from temgym_core.pyjem_simulator import CameraModel, LensHysteresis, SimulatedMicroscope, install
from temgym_core.ray import Ray
from temgym_core.run import run_to_end

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
CONSOLE = json.loads(
    (HERE.parent / "microscope_inversion/recordings/2026-11-17/console_tables.json").read_text()
)
PROJECTOR = json.loads((HERE / "projector_lenses.json").read_text())
CONDENSER = json.loads((HERE / "condenser.json").read_text())
DRAWING = json.loads((HERE.parent / "virtual_neoarm/neoarm_column_diagram.json").read_text())["z_m"]
Z = {name: z / DRAWING["Screen"] for name, z in DRAWING.items()}

# The deflector values of the console's MAG 40k row are the aligned state of CLA1 and CLA2. The
# aligned state of SpotA is its value at spot 2 and alpha 1, the start of the simulated NEOARM. A
# deflector kicks the beam only when its value moves away from its aligned state. The positions of
# SpotA, CLA1 and CLA2 are the start values of notebook 1.
ALIGNED = next(r for r in CONSOLE["rows"]["MAG"] if r["mag"][2] == "X40k")
SPOT_2 = next(r for r in CONSOLE["spot_alpha"] if r["spot"] == 2 and r["alpha"] == 1)
ALIGNED_DEFLECTORS = {"SpotA": SPOT_2["SpotA"], "CLA1": ALIGNED["CLA1"], "CLA2": ALIGNED["CLA2"]}
DEFLECTOR_Z = {"SpotA": -0.66, "CLA1": -0.5786, "CLA2": -0.5623}

# The differences of the hidden microscope from notebook 1. Each number is a choice. A lens moves
# by up to 7 mm along the axis. Its gap and bore change by up to 8 %, and its coil by up to 2 %.
# The centres of CL1 and CL3 are a few um off the axis. Each deflector kicks with about 1e-7 rad per
# DAC unit. This is a third of notebook 1's start value. The directions turn, with some crosstalk
# between x and y. Some parts also have another form than the model's (the harder test of
# 2026-11-20). The lenses lag behind their DAC values, and the objective's two fields differ. The
# corrector has no lenses, because the principal says there are no hidden lenses (M2).
HIDDEN = dict(
    lens_z_shift={"IL1": 0.006, "IL2": -0.008, "IL3": 0.010, "PL1": -0.005},
    gap_factor={"CL1": 1.05, "CL3": 0.95, "IL1": 1.08, "IL2": 0.93, "IL3": 1.06, "PL1": 0.94},
    bore_factor={"CL1": 0.96, "CL3": 1.05, "IL1": 0.94, "IL2": 1.07, "IL3": 0.95, "PL1": 1.08},
    ampere_turns_factor={"IL1": 1.01, "IL2": 0.99, "IL3": 1.01, "PL1": 0.99},
    centre_m={
        "CL1": (3e-6, -2e-6),
        "CL3": (-2e-6, 1e-6),
        "IL1": (20e-6, 10e-6),
        "IL2": (-15e-6, 25e-6),
        "IL3": (10e-6, -20e-6),
        "PL1": (-20e-6, -10e-6),
    },
    # The iron of each lens that a script changes lags behind its DAC value. After a rise, the
    # field is that of a value up to 150 units lower. After a fall, the field is that of a value
    # up to 150 units higher. Notebook 2 fits one lag per lens. The widths and fractions are
    # choices, about 0.5 % of a typical value.
    hysteresis=LensHysteresis(widths_dac=(100.0, 400.0), fractions=(0.3, 0.3)),
    # Positions along the axis (L) of the condenser lenses and the corrector, and the coils of the
    # condenser lenses. These numbers equal notebook 1's in the model's start, or are values that
    # notebook 2 holds fixed.
    # The condenser's positions and strengths below come from a least-squares match to the
    # console's spot table of alpha 1, held near the drawing (2026-11-22). These are
    # condenser_z_shift, condenser_lens_factor, and SpotA's place and strength. Each spot focuses
    # 300 units below its console CL3. So the console's CL3, approached from below (150 units of
    # lag), lies just above focus, as record_lens_changes.py expects. Let SpotA have the console's
    # value, and CL1 and CL3 come from below. A change of spot then moves the beam on the specimen
    # by at most 34 nm. Each spot focuses within 0.4 nm per urad of source angle.
    condenser_z_shift={"CL1": 0.004 - 0.000496, "CL3": -0.005 + 0.000630, "CM": 0.003 + 0.002681},
    corrector_z_shift=(0.003, -0.002),
    # The corrector's transfer. "free space" is a corrector that is off, as on the NEOARM. In TEM
    # mode, its software shows every current at zero, including those of the transfer lenses (the
    # principal, 2026-11-22, Q13). "minus one" images its top onto its bottom with a magnification
    # of -1. A dict gives any other A, B and D.
    corrector="free space",
    # The console's CL3 of each spot is near the condenser focus, as on the real NEOARM. Take the
    # moved condenser lenses and the -1 corrector. Then 1.017 times the coils put the focus at the
    # console's CL3 at MAG 100k (a trace with frames). With the corrector off, the focus needs
    # 0.9162 of the -1 corrector's coils (the smallest beam at the specimen, 2026-11-22). The
    # factor is therefore 1.017 x 0.9162.
    condenser_turns_factor=1.017 * 0.9162,
    # Each condenser lens's own share of that strength. Separate lenses need not have the same
    # ampere-turns per DAC unit. With one strength for all three, the spot table did not match. The
    # mini lens CM comes out at 0.38 of the drawing's strength.
    condenser_lens_factor={"CL1": 1.0617, "CL3": 1.0276, "CM": 0.3833},
    cm_bore_factor=0.95,
    # Radians per DAC unit. The kick (x, y) is this matrix times the change of (x, y). The
    # strength of CLA1 and CLA2 follows from the ranges of a beam shift and tilt (COIL_RANGES).
    kick_per_dac={
        "SpotA": 1.0431e-7 * DRAWING["Screen"] * np.array([[0.95, 0.25], [-0.22, 1.05]]),
        # These are the shapes of CLA1 and CLA2. COIL_RANGES below scales them after the column
        # exists.
        "CLA1": np.array([[0.98, -0.20], [0.17, 1.02]]),
        "CLA2": np.array([[1.03, 0.12], [-0.15, 0.97]]),
    },
    # The projector deflector PLA (diffraction shift), midway between IL3 and PL1, in rad per DAC
    # unit about 32768. No source gives its place or strength. There 1.2e-6 rad per unit moves a
    # DIFF pattern about 0.015 px per unit, about two SightX widths over half the range.
    pla_kick_per_dac=1.2e-6 * np.array([[1.02, -0.10], [0.08, 0.97]]),
    source_spread_factor=1.1,
    # The objective sits 5 % further from the specimen (2.20 mm against 2.10 mm), and its focal
    # length is 4 % longer. Notebook 2 fits both, from notebook 1's values. The specimen sits only
    # 0.03 mm outside the focal length in notebook 1, and 0.05 mm here. A small change of the focal
    # length alone moves the specimen's image a long way. A trace without frames tried a focal
    # length 2 % shorter, at notebook 1's distance. After some lens changes, CLA2 then hardly
    # moved the beam in DIFF. These values do not have this problem.
    objective_factor=1.04,
    objective_z_factor=1.05,
    # The objective's post-field has this times the focal length of its pre-field. 2 % shorter
    # moved the camera lengths of the DIFF rows by 20 %, far from their labels. 0.5 % moves them by
    # 4 %.
    objective_post_factor=0.995,
    # The fine focus OLf changes the objective's excitation by this fraction per DAC unit. The
    # strength of a magnetic lens goes as the square of its excitation, so the focal length goes as
    # its inverse square. 4e-7 per unit is about 1.7 nm of focus per unit.
    objective_excitation_per_dac=4e-7,
    # CM's gap is 10 % larger. SpotA and CLA1 sit about 3 mm lower and CLA2 2 mm higher. The SightX
    # sits off the axis by (1.5, -1.0) mm, about (24, -16) px, and its turn is 20 mrad.
    cm_gap_factor=1.10,
    deflector_z_shift={"SpotA": 0.004 - 0.0000838, "CLA1": 0.004, "CLA2": -0.003},
    # CLA1 (beam shift) and CLA2 (beam tilt) are two commands to the same two pairs of coils. The
    # upper pair sits at notebook 1's CLA1 and the lower 0.7 mm below notebook 1's CLA2. The
    # two pairs share each command's kick. Its balance sets the share of the upper pair,
    # and with it the pivot point, but not the kick. At the console's ShifBal and TiltBal the
    # pivots are nearly right (CONSOLE_SHARE, below). The share changes by balance_per_dac per DAC
    # unit of ShifBal or TiltBal. The scale is a choice, since no source gives JEOL's. One unit
    # moves a pivot by 3.5 um.
    cla_coils_z=(DEFLECTOR_Z["CLA1"], DEFLECTOR_Z["CLA2"] + 0.001),
    balance_per_dac=2e-4,
    # How far the console's ShifBal and TiltBal are from the pivot points (DAC units). Choices.
    pivot_error_dac={"ShifBal": (-300, 200), "TiltBal": (600, -400)},
    sightx_centre_m=(1.5e-3, -1.0e-3),
    sightx_turn_rad=0.020,
    # The condenser aperture, at the drawing's CLAperture (stage 9b, 2026-11-22). Its holes, its
    # step per Apt3 DAC unit about 2048 and its soft edge are those of the virtual NEOARM
    # (examples/virtual_neoarm/condenser.json). No source gives the real NEOARM's values. Its
    # centre sits off the axis by condenser_aperture_offset_m at the Apt3 value 2048. This offset
    # gives the aperture centring of the alignment guide something to do. These are choices, made
    # before any fit with the aperture. The simulated beam is about 10 um rms at the aperture
    # (2026-11-22). So the 150 um, 70 um and 40 um holes hardly touch it. The recordings and
    # notebook 2's fit give the same results as with hole 1 (a check found 0.03 px). The 10 um
    # hole limits the beam, as a condenser aperture should. Its position then sets the beam's
    # axis. The aperture-centring rehearsal selects it (install_simulated_neoarm(condenser_hole=4)).
    # Notebook 2's model has no aperture.
    condenser_aperture_z=Z["CLAperture"],
    condenser_aperture_holes_m=(150e-6, 70e-6, 40e-6, 10e-6),
    condenser_aperture_m_per_dac=1e-7,
    condenser_aperture_edge_m=0.5e-6,
    condenser_aperture_offset_m=(5e-6, -3e-6),
)

# The SightX has 1916 pixels of 62.6 um, 12 cm in all. An offset of 100 counts keeps every count
# above zero. A reader that takes the frame as unsigned, as jemwrap does, then gets the same
# numbers.
SIGHTX = CameraModel(
    name="screenCamera",
    sensor_shape=(1916, 1916),
    sensor_pixel_m=62.6e-6,
    binnings=(1, 2, 4),
    data_bits=16,
    offset_adu=100.0,
)


def condenser_aperture(console):
    """The condenser aperture of the console's hole and Apt3 position, or nothing for hole 0."""
    hole, x_dac, y_dac = console.get("CondenserAperture", (0, 2048, 2048))
    if int(hole) == 0:
        return []
    offset = HIDDEN["condenser_aperture_offset_m"]
    step = HIDDEN["condenser_aperture_m_per_dac"]
    return [
        SigmoidAperture(
            z=HIDDEN["condenser_aperture_z"],
            radius=HIDDEN["condenser_aperture_holes_m"][int(hole) - 1] / 2,
            edge_width=HIDDEN["condenser_aperture_edge_m"],
            x0=(float(x_dac) - 2048) * step + offset[0],
            y0=(float(y_dac) - 2048) * step + offset[1],
        )
    ]


def corrector_transfer(length):
    """Return the A, B and D of the corrector's transfer over ``length`` (HIDDEN["corrector"])."""
    form = HIDDEN["corrector"]
    if form == "minus one":
        return dict(A=-np.eye(2), B=np.zeros((2, 2)), D=-np.eye(2))
    if form == "free space":
        return dict(A=np.eye(2), B=length * np.eye(2), D=np.eye(2))
    # Any other form is a dict of its own A, B and D.
    return {key: np.asarray(value, float) for key, value in form.items()}


# The share of each condenser deflector's kick in the upper coils at the console's balances. The
# code below sets it after the column exists.
CONSOLE_SHARE = None


def whole_column(console, voltage_v, aligning_kick=None):
    """Return the column from the source to the SightX, and the beam, for these DAC values.

    ``aligning_kick`` holds the CLA1 and CLA2 kicks (x, y, x, y) of the operator's alignment. By
    default it is ALIGNING_KICK.
    """
    if aligning_kick is None:
        aligning_kick = ALIGNING_KICK
    corrector_top = Z["Corrector Top"] + HIDDEN["corrector_z_shift"][0]
    corrector_bottom = Z["Corrector Bottom"] + HIDDEN["corrector_z_shift"][1]
    condenser_turns = CONDENSER["ampere_turns_per_dac"] * HIDDEN["condenser_turns_factor"]
    condenser_z = {
        name: Z[label] + HIDDEN["condenser_z_shift"][name]
        for name, label in (("CL1", "CL1"), ("CL3", "CL2"), ("CM", "CLMini"))
    }
    gap, bore, centre = HIDDEN["gap_factor"], HIDDEN["bore_factor"], HIDDEN["centre_m"]
    excitation = 1 + HIDDEN["objective_excitation_per_dac"] * (
        float(console["OLf"]) - float(ALIGNED["OLf"])
    )
    # A magnetic lens's focal length grows with the beam's momentum squared. The objective follows
    # the high tension here, as the pole-piece lenses do by themselves.
    momentum_squared = 1 / magnetic_strength_ratio(voltage_v, 200e3) ** 2
    objective_f = (
        PROJECTOR["objective_f"] * HIDDEN["objective_factor"] / excitation**2 * momentum_squared
    )
    objective_z = PROJECTOR["objective_z"] * HIDDEN["objective_z_factor"]
    spot_kick = HIDDEN["kick_per_dac"]["SpotA"] @ (
        np.asarray(console["SpotA"], float) - np.asarray(ALIGNED_DEFLECTORS["SpotA"], float)
    )
    spot_z = DEFLECTOR_Z["SpotA"] + HIDDEN["deflector_z_shift"]["SpotA"]
    deflectors = [Deflector(z=spot_z, def_x=spot_kick[0], def_y=spot_kick[1])]
    upper, lower = HIDDEN["cla_coils_z"]
    upper_kick, lower_kick = np.zeros(2), np.zeros(2)
    for name, balance_name in (("CLA1", "ShifBal"), ("CLA2", "TiltBal")):
        change = HIDDEN["kick_per_dac"][name] @ (
            np.asarray(console[name], float) - np.asarray(ALIGNED_DEFLECTORS[name], float)
        )
        aligning = aligning_kick[2 * ("CLA1", "CLA2").index(name) :][:2]
        # The upper pair gives a share s of a kick k, and the lower pair gives the rest, (1 - s) k.
        # Together they act as one kick k at the plane s (lower - upper) above the lower pair.
        # The balance sets the share
        # of the change from the console's value, and with it that change's pivot point, but not
        # its kick. The operator's aligning kick keeps the console's share. It stands for the
        # alignment of the hidden column's lens offsets, which a balance does not move. If the
        # balance shared it too, a tilt balance would throw the aligned beam far off the axis.
        if CONSOLE_SHARE is None:  # CONSOLE_SHARE is None only while the code below finds it.
            single_z = DEFLECTOR_Z[name] + HIDDEN["deflector_z_shift"][name]
            console_share = (lower - single_z) / (lower - upper)
        else:
            console_share = CONSOLE_SHARE[name]
        share = console_share + HIDDEN["balance_per_dac"] * (
            np.asarray(console[balance_name], float) - np.asarray(ALIGNED[balance_name], float)
        )
        upper_kick += share * change + console_share * aligning
        lower_kick += (1 - share) * change + (1 - console_share) * aligning
    deflectors += [
        Deflector(z=upper, def_x=upper_kick[0], def_y=upper_kick[1]),
        Deflector(z=lower, def_x=lower_kick[0], def_y=lower_kick[1]),
    ]
    projector = [
        PolePieceLens(
            z=PROJECTOR["lens_z"][name] + HIDDEN["lens_z_shift"][name],
            turns=1.0,
            current=PROJECTOR["ampere_turns_per_dac"][name]
            * HIDDEN["ampere_turns_factor"][name]
            * console[name],
            gap=PROJECTOR["pole_gap"][name] * gap[name],
            bore_radius=PROJECTOR["bore_radius"][name] * bore[name],
            x0=centre[name][0],
            y0=centre[name][1],
        )
        for name in ("IL1", "IL2", "IL3", "PL1")
    ]
    pla_z = 0.5 * (
        PROJECTOR["lens_z"]["IL3"]
        + HIDDEN["lens_z_shift"]["IL3"]
        + PROJECTOR["lens_z"]["PL1"]
        + HIDDEN["lens_z_shift"]["PL1"]
    )
    pla_kick = HIDDEN["pla_kick_per_dac"] @ (np.asarray(console["PLA"], float) - 32768)
    projector.append(Deflector(z=pla_z, def_x=pla_kick[0], def_y=pla_kick[1]))
    column = [
        PolePieceLens(
            z=condenser_z["CL1"],
            turns=condenser_turns * HIDDEN["condenser_lens_factor"]["CL1"],
            current=console["CL1"],
            gap=CONDENSER["pole_gap"]["CL1"] * gap["CL1"],
            bore_radius=CONDENSER["bore_radius"]["CL1"] * bore["CL1"],
            x0=centre["CL1"][0],
            y0=centre["CL1"][1],
        ),
        # On the drawing, CL3 sits where the label says CL2.
        PolePieceLens(
            z=condenser_z["CL3"],
            turns=condenser_turns * HIDDEN["condenser_lens_factor"]["CL3"],
            current=console["CL3"],
            gap=CONDENSER["pole_gap"]["CL3"] * gap["CL3"],
            bore_radius=CONDENSER["bore_radius"]["CL3"] * bore["CL3"],
            x0=centre["CL3"][0],
            y0=centre["CL3"][1],
        ),
        *deflectors,
        *condenser_aperture(console),
        # The corrector is a transfer from its top to its bottom (HIDDEN["corrector"]).
        ABCDTransfer(
            **corrector_transfer(corrector_bottom - corrector_top),
            z=corrector_top,
            length=corrector_bottom - corrector_top,
        ),
        PolePieceLens(
            z=condenser_z["CM"],
            turns=condenser_turns * HIDDEN["condenser_lens_factor"]["CM"],
            current=console["CM"],
            gap=CONDENSER["pole_gap"]["CM"] * HIDDEN["cm_gap_factor"],
            bore_radius=CONDENSER["bore_radius"]["CM"] * HIDDEN["cm_bore_factor"],
        ),
        Lens(z=-objective_z, focal_length=objective_f),
        Lens(z=objective_z, focal_length=objective_f * HIDDEN["objective_post_factor"]),
        *projector,
        # The SightX off the axis and turned. The sort below keeps these two before the plane.
        Scanner(
            z=1.0,
            scan_pos_x=-HIDDEN["sightx_centre_m"][0],
            scan_pos_y=-HIDDEN["sightx_centre_m"][1],
        ),
        Rotator(z=1.0, angle=float(np.degrees(HIDDEN["sightx_turn_rad"]))),
        Plane(z=1.0),
    ]
    column = sorted(column, key=lambda component: float(component.z))
    size = CONDENSER["source_size"]
    spread = CONDENSER["source_spread"] * HIDDEN["source_spread_factor"]
    return column, Z["Source"], np.zeros(4), np.diag([size**2, size**2, spread**2, spread**2])


# The operator's alignment. The lenses of the hidden condenser are off the axis. So with the
# console's aligned values at MAG 40k and spot 2, the beam leaves the axis. An operator moves
# CLA1 and CLA2 until the beam is on the axis at the specimen and parallel to it. Here these two
# deflectors get the extra kick that does that. It is the same at every row.
ALIGNED_STATE = ALIGNED | {name: SPOT_2[name] for name in ("CL1", "CL3", "CM", "SpotA")}
# A lens brought to its value from 2000 units below acts as if its value were lower by this lag.
# The lag is the sum of the fractions times the widths of the hidden hysteresis. The operator
# aligns the beam below with CL1 and CL3 in that state, the state every script leaves them in.
FROM_BELOW_LAG_DAC = sum(
    fraction * width
    for fraction, width in zip(HIDDEN["hysteresis"].fractions, HIDDEN["hysteresis"].widths_dac)
)
ALIGNED_FROM_BELOW = ALIGNED_STATE | {
    name: ALIGNED_STATE[name] - FROM_BELOW_LAG_DAC for name in ("CL1", "CL3")
}


def beam_at_specimen(aligning_kick):
    """Return x, y, dx and dy of the central ray at the specimen, in the aligned state."""
    column, source_z, _, _ = whole_column(ALIGNED_FROM_BELOW, 200e3, aligning_kick)
    above_specimen = [part for part in column if float(part.z) < 0] + [Plane(z=0.0)]
    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=source_z, pathlength=0.0, voltage=200e3)
    out = run_to_end(ray, above_specimen)
    return jnp.stack([out.x, out.y, out.dx, out.dy])


# A first aligning kick, while the code below finds the specimen's height and the console's
# pivots. Only the column below the coils matters for those.
ALIGNING_KICK = np.zeros(4)


def specimen_to_camera(position_and_slope):
    """Return where a ray leaving the plane z = 0 with x, y, dx, dy lands on the SightX, in the
    aligned state at MAG 40k."""
    column, _, _, _ = whole_column(ALIGNED_STATE, 200e3)
    below_specimen = [part for part in column if float(part.z) > 0]
    x, y, dx, dy = position_and_slope
    out = run_to_end(
        Ray(x=x, y=y, dx=dx, dy=dy, z=0.0, pathlength=0.0, voltage=200e3), below_specimen
    )
    return jnp.stack([out.x, out.y])


# The operator's focus. The hidden objective is stronger than the drawing's. So the plane that the
# projector images onto the SightX lies below z = 0, by more than OLf can move it. An operator
# brings the specimen there with stage Z. A ray through that plane at height 0 crosses z = 0 at
# x = -h dx. It lands on the SightX at A x + B dx. This point does not depend on dx when B = h A.
# So h is the least-squares solution of B = h A. It is the same, within 0.1 um, at every row from
# MAG 30k up (2026-11-22).
_to_camera = jax.jacfwd(specimen_to_camera)(jnp.zeros(4))
_A, _B = _to_camera[:, :2], _to_camera[:, 2:]
SPECIMEN_Z = float(jnp.sum(_A * _B) / jnp.sum(_A * _A))


def coils_to_specimen():
    """Return the transfer (A, B, C, D), along x, from the lower condenser coils to the specimen."""
    _, lower = HIDDEN["cla_coils_z"]
    column, _, _, _ = whole_column(ALIGNED_STATE, 200e3)
    between = [
        part
        for part in column
        if lower < float(part.z) < SPECIMEN_Z and not isinstance(part, Deflector)
    ]

    def at_specimen(position_and_slope):
        x, dx = position_and_slope
        ray = Ray(x=x, y=0.0, dx=dx, dy=0.0, z=lower, pathlength=0.0, voltage=200e3)
        out = run_to_end(ray, [*between, Plane(z=SPECIMEN_Z)])
        return jnp.stack([out.x, out.dx])

    (a, b), (c, d) = np.asarray(jax.jacfwd(at_specimen)(jnp.zeros(2)))
    return a, b, c, d


# The console's pivot points are nearly right, as after a recent alignment loaded on the console.
# The source is the principal, 2026-11-22 ("near-aligned start"). A kick k at the plane L above the
# lower coils shifts the beam at the specimen by (A L + B) k. It tilts the beam there by
# (C L + D) k. Beam tilt pivots
# on the specimen at L = -B / A. Beam shift does not tilt at L = -D / C. The share of the upper
# coils that puts a kick there is L / (lower - upper). The console's balances miss these by
# HIDDEN["pivot_error_dac"], for the pivot-point scripts to find.
_upper, _lower = HIDDEN["cla_coils_z"]
_a, _b, _c, _d = coils_to_specimen()
CONSOLE_SHARE = {
    name: right / (_lower - _upper)
    - HIDDEN["balance_per_dac"] * np.asarray(HIDDEN["pivot_error_dac"][balance], float)
    for name, balance, right in (("CLA1", "ShifBal", -_d / _c), ("CLA2", "TiltBal", -_b / _a))
}

# The strength of CLA1 and CLA2. Over half their range (32768 units) they give a pure beam shift of
# 50 um and a pure beam tilt of 38 mrad. The shift is a choice, because no source gives it. It
# brings the beam onto the SightX at X2000, a field about 60 um wide. The beam-tilt coils of a TEM
# reach precession angles of 0 to 3 degrees. On a JEOL 2100F, 1 degree (17.5 mrad) is a common
# working value (en.wikipedia.org/wiki/Precession_electron_diffraction). At its pivot a kick k
# shifts the beam by -k / C (pure shift) or tilts it by k / A (pure tilt).
COIL_RANGES = {"CLA1": 50e-6 * abs(_c), "CLA2": 38e-3 * abs(_a)}
for _name, _range in COIL_RANGES.items():
    HIDDEN["kick_per_dac"][_name] = _range / 32768 * HIDDEN["kick_per_dac"][_name]

# The beam at the specimen is linear in the kicks, so one Newton step solves it exactly.
_no_kick = jnp.zeros(4)
ALIGNING_KICK = np.asarray(
    -jnp.linalg.solve(jax.jacfwd(beam_at_specimen)(_no_kick), beam_at_specimen(_no_kick))
)


# A cross-grating replica, after the standard 2160 lines/mm (a 463 nm period). Its holes, its bars'
# transmission, its soft edges and its turn are choices, made to look like the SightX images of
# a grating (2026-11-22). The specimen sits at the operator's focus, SPECIMEN_Z.
CROSS_GRATING = CrossGrating(
    z=SPECIMEN_Z,
    period=463e-9,
    hole_fraction=0.6,
    bar_transmission=0.05,
    edge_width=20e-9,
    rotation=0.3,
)

# A gold film at the operator's focus gives the DIFF ruler its 111, 200 and 220 rings. It is
# TemGym's PolycrystallineFilm, with 5 nm grains, and 30 % of them diffract.
GOLD_FILM = PolycrystallineFilm.gold(z=SPECIMEN_Z)


# How far the HT wobbler moves the high tension each way. Nobody knows the NEOARM's value. This
# value is a choice. With it, the cross grating's image swings 7 px at MAG X30k with the aligned
# CLA2. Then 1000 units of CLA2 change the swing by about 12 px (2026-11-22).
HT_WOBBLER_V = 100.0


def simulated_neoarm(
    ray_sample_power=16,
    condenser_hole=1,
    specimen=None,
    wave_beamlets=None,
):
    """Return the simulated NEOARM at MAG 40k, spot 2 and alpha 1, with aligned deflectors.

    ``condenser_hole`` is the condenser aperture hole selected at the start (1 to 4). ``specimen``
    is a TemGym component at the specimen plane (such as CROSS_GRATING), or None for no specimen.
    With a specimen, ``wave_beamlets`` (beamlets across) images it with one coherent Gaussian
    beam. None images it with rays, as its shadow.
    """
    scope = SimulatedMicroscope.from_console(
        CONSOLE,
        whole_column,
        alpha=1,
        camera=SIGHTX,
        hysteresis={
            name: HIDDEN["hysteresis"] for name in ("CL1", "CL3", "IL1", "IL2", "IL3", "PL1")
        },
        magnification=40000,
        spot=2,
        ray_sample_power=ray_sample_power,
    )
    # The simulator starts every deflector at 32768. The real console row has the aligned values.
    # SpotA already has the value of spot 2 from the spot table.
    scope.deflectors.update(
        {
            name: list(ALIGNED[name])
            for name in scope.deflectors
            if name in ALIGNED and name != "SpotA"
        }
    )
    # The operator's choice of condenser aperture hole (see HIDDEN): 1 is 150 um, 4 is 10 um.
    scope.apertures[1]["size"] = condenser_hole
    scope.specimen = specimen
    scope.wave_beamlets = wave_beamlets
    scope.ht_wobbler_v = HT_WOBBLER_V
    # The operator's last change brought every lens with hysteresis to its value from below, as
    # every script does.
    for name, law in scope.electronics.hysteresis.items():
        dac = scope.lens_dac(name)
        scope.lens_memory[name] = (dac, law.move(law.at_rest(dac - 2000), dac))
    # from_console takes the objective of the first MAG row (MAG 2000). The rows from MAG 8k up,
    # which this model uses, have the objective of the MAG 40k row.
    scope.electronics = dataclasses.replace(
        scope.electronics, objective_dac=(ALIGNED["OLc"], ALIGNED["OLf"])
    )
    return scope


# The entered install() context and its microscope. Python closes a context that nobody holds at
# once, and closing it removes the simulated PyJEM again.
_INSTALLED = []


def install_simulated_neoarm(
    ray_sample_power=16,
    condenser_hole=1,
    specimen=None,
    wave_beamlets=None,
):
    """Make `import PyJEM` find a new simulated NEOARM until the program ends, and return it.

    ``condenser_hole`` is the condenser aperture hole that the operator selects (1 to 4).
    ``specimen`` and ``wave_beamlets`` are as in simulated_neoarm.

    If a simulated NEOARM exists already, the function changes nothing and returns that one. A
    notebook (`alignment.ipynb`) runs the scripts one after another. They all work on the
    microscope that the notebook installed, and each starts where the last one left it.
    """
    if _INSTALLED:
        return _INSTALLED[0][1]
    scope = simulated_neoarm(ray_sample_power, condenser_hole, specimen, wave_beamlets)
    # install() is a context manager. Entering it without leaving keeps the simulated PyJEM for the
    # whole run, so a script needs no `with` block around its steps.
    context = install(scope)
    context.__enter__()
    _INSTALLED.append((context, scope))
    return scope


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent.parent / "jemwrap/src"))
    from jemwrap import ENDPOINTS, MicroscopeMonitor, PyJEMAdapter, render_state_table

    with install(simulated_neoarm()):
        from PyJEM import TEM3, detector, sightx

        adapter = PyJEMAdapter(
            connect=False, tem3_module=TEM3, detector_module=detector, sightx_module=sightx
        )
        monitor = MicroscopeMonitor(adapter=adapter)
        # One poll of every value. The monitor normally spreads these polls over time.
        for endpoint in ENDPOINTS:
            monitor._poll_endpoint(endpoint, raise_errors=False)
        print(render_state_table(monitor.snapshot()))

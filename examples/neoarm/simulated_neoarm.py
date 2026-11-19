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

Limits of this simulated NEOARM:
- The deflector strength is the start value of notebook 1. No fit gives it yet. A CLA1 kick moves
  the beam about three times as far as on the real NEOARM.
- The condenser follows the spot table of alpha 1. The alpha selector accepts a new value but
  changes nothing.
- simulated_neoarm() sets the aligned deflector values and the objective after it makes the
  microscope. SimulatedMicroscope.reset() puts them back to neutral and to the first MAG row.

Along the axis, lengths are in units of L, the distance from the specimen to the SightX. Across
the axis, positions are in metres, so a slope is in metres per L.
"""

import dataclasses
import json
import sys
from pathlib import Path

import jax
import numpy as np

from temgym_core.components import Deflector, Lens, Plane, PolePieceLens
from temgym_core.pyjem_simulator import CameraModel, SimulatedMicroscope, install

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
CONSOLE = json.loads(
    (HERE.parent / "microscope_inversion/recordings/2026-11-17/console_tables.json").read_text()
)
PROJECTOR = json.loads((HERE / "projector_lenses.json").read_text())
CONDENSER = json.loads((HERE / "condenser.json").read_text())
DRAWING = json.loads((HERE.parent / "virtual_neoarm/neoarm_column_diagram.json").read_text())["z_m"]
Z = {name: z / DRAWING["Screen"] for name, z in DRAWING.items()}

# The deflector values of the console's MAG 40k row are the aligned state. A deflector kicks the
# beam only when its value moves away from them. The strength, 3e-7 rad per DAC unit, and the
# positions of SpotA, CLA1 and CLA2 are the start values of notebook 1.
ALIGNED = next(r for r in CONSOLE["rows"]["MAG"] if r["mag"][2] == "X40k")
KICK_PER_DAC = 3e-7 * DRAWING["Screen"]  # metres per L per DAC unit
DEFLECTOR_Z = {"SpotA": -0.66, "CLA1": -0.5786, "CLA2": -0.5623}

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


def whole_column(console, voltage_v):
    """Return the column from the source to the SightX, and the beam, for these DAC values."""
    relay = (Z["Corrector Bottom"] - Z["Corrector Top"]) / 4
    condenser_turns = CONDENSER["ampere_turns_per_dac"]
    deflectors = [
        Deflector(
            z=DEFLECTOR_Z[name],
            def_x=KICK_PER_DAC * (console[name][0] - ALIGNED[name][0]),
            def_y=KICK_PER_DAC * (console[name][1] - ALIGNED[name][1]),
        )
        for name in ("SpotA", "CLA1", "CLA2")
    ]
    projector = [
        PolePieceLens(
            z=PROJECTOR["lens_z"][name],
            turns=1.0,
            current=PROJECTOR["ampere_turns_per_dac"][name] * console[name],
            gap=PROJECTOR["pole_gap"][name],
            bore_radius=PROJECTOR["bore_radius"][name],
        )
        for name in ("IL1", "IL2", "IL3", "PL1")
    ]
    column = [
        PolePieceLens(
            z=Z["CL1"],
            turns=condenser_turns,
            current=console["CL1"],
            gap=CONDENSER["pole_gap"]["CL1"],
            bore_radius=CONDENSER["bore_radius"]["CL1"],
        ),
        # On the drawing, CL3 sits where the label says CL2.
        PolePieceLens(
            z=Z["CL2"],
            turns=condenser_turns,
            current=console["CL3"],
            gap=CONDENSER["pole_gap"]["CL3"],
            bore_radius=CONDENSER["bore_radius"]["CL3"],
        ),
        *deflectors,
        Lens(z=Z["Corrector Top"] + relay, focal_length=relay),
        Lens(z=Z["Corrector Bottom"] - relay, focal_length=relay),
        PolePieceLens(
            z=Z["CLMini"],
            turns=condenser_turns,
            current=console["CM"],
            gap=CONDENSER["pole_gap"]["CM"],
            bore_radius=CONDENSER["bore_radius"]["CM"],
        ),
        Lens(z=-PROJECTOR["objective_z"], focal_length=PROJECTOR["objective_f"]),
        Lens(z=PROJECTOR["objective_z"], focal_length=PROJECTOR["objective_f"]),
        *projector,
        Plane(z=1.0),
    ]
    column = sorted(column, key=lambda component: float(component.z))
    size, spread = CONDENSER["source_size"], CONDENSER["source_spread"]
    return column, Z["Source"], np.zeros(4), np.diag([size**2, size**2, spread**2, spread**2])


def simulated_neoarm(ray_sample_power=16):
    """Return the simulated NEOARM at MAG 40k, spot 2 and alpha 1, with aligned deflectors."""
    scope = SimulatedMicroscope.from_console(
        CONSOLE,
        whole_column,
        alpha=1,
        camera=SIGHTX,
        magnification=40000,
        spot=2,
        ray_sample_power=ray_sample_power,
    )
    # The simulator starts every deflector at 32768. The real console row has the aligned values.
    scope.deflectors.update(
        {name: list(ALIGNED[name]) for name in scope.deflectors if name in ALIGNED}
    )
    # from_console takes the objective of the first MAG row (MAG 2000). The rows from MAG 8k up,
    # which this model uses, have the objective of the MAG 40k row.
    scope.electronics = dataclasses.replace(
        scope.electronics, objective_dac=(ALIGNED["OLc"], ALIGNED["OLf"])
    )
    return scope


# The entered install() contexts. Python closes a context that nobody holds at once, and closing
# it removes the simulated PyJEM again.
_INSTALLED = []


def install_simulated_neoarm(ray_sample_power=16):
    """Make `import PyJEM` find a new simulated NEOARM until the program ends."""
    # install() is a context manager. Entering it without leaving keeps the simulated PyJEM for the
    # whole run, so a script needs no `with` block around its steps.
    context = install(simulated_neoarm(ray_sample_power))
    context.__enter__()
    _INSTALLED.append(context)


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

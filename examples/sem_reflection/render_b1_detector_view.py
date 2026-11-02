"""Saved live-view states for the ERC B1 figure: Cu(001) 2×1, 70 mm camera.

The surface is a missing-row 2×1 reconstruction of Cu(001): one Cu atom per
surface cell doubled along the <110> beam row, so half-order Bragg beams appear
between the integer orders on the Laue arc. It is set by doubling the
notebook's lattice constant; the notebook's own (0, m) order row and Cu
scattering factors are used unchanged.

Runs the setup, column, lattice and preview/validation cells of
sem_reflection.ipynb headlessly and writes camera_view.{npz,json} in the
notebook's own format: one state per incidence to
output/copper_b1/incidence_<deg>/, the first also copied to output/copper_b1/.
The camera follows the specular beam, as in the notebook. The notebook is not
modified.

The image uses the notebook's medium path at 192 packets across, checked
against 128 across. The million-packet path is not used: at these settings it
returned a near-empty image (total ~1e-15), which needs separate investigation.
"""
from pathlib import Path
import contextlib
import io
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "sem_reflection.ipynb"
OUT = HERE / "output/copper_b1"
CAMERA_LENGTH = .07
MATERIAL = "Cu(001) 2x1 missing-row reconstruction"
LATTICE = 2 * .3615e-9   # doubled period along the beam row
INCIDENCES = (3., 4.5)
TOLERANCE = 1e-3


def notebook_state(overrides):
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    env = {}
    setup = "".join(cells[1]["source"]).replace("RUN_MILLION = True", "RUN_MILLION = False")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(setup, str(NOTEBOOK), "exec"), env)
        env["P"].update(overrides)
        env["P"]["sample_z"] = env["P"]["objective_z"] + env["OBJECTIVE_TO_SAMPLE"]
        exec(compile("".join(cells[3]["source"]), str(NOTEBOOK), "exec"), env)
        exec(compile("".join(cells[6]["source"]), str(NOTEBOOK), "exec"), env)
        exec(compile("".join(cells[8]["source"]), str(NOTEBOOK), "exec"), env)
    return env


def save(incidence, out):
    overrides = dict(incidence_deg=incidence, camera_length=CAMERA_LENGTH,
                     lattice_constant=LATTICE, material=MATERIAL)
    env = notebook_state(overrides)

    def render(n):
        beams = env["make_reflected"](n)[0]
        return np.asarray(env["gaussian_image"](beams, env["detector"], samples_per_pixel=4,
                                                method="spatial", tile_beams=32))

    image, check = render(192), render(128)
    relative = float(np.linalg.norm(check - image) / np.linalg.norm(image))
    assert relative < TOLERANCE, relative
    frame = env["screen_frame"]
    centre = np.asarray(frame.origin_xyz)
    pose = dict(z_mm=float(centre[2] * 1000), height_mm=float(centre[0] * 1000),
                distance_m=float(env["P"]["camera_length"]), centre_world_xyz_m=centre.tolist())
    parameters = {k: list(v) if isinstance(v, tuple) else v for k, v in env["P"].items()}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "camera_view.npz", intensity=image, camera_pose=json.dumps(pose),
                        parameters=json.dumps(parameters), quality="Medium render, 192 across")
    (out / "camera_view.json").write_text(json.dumps(dict(
        parameters=parameters, camera_pose=pose, quality="Medium render, 192 across",
        convergence=dict(n128_to_n192=relative), source=str(NOTEBOOK.name),
        surface=MATERIAL), indent=2) + "\n")
    print(f"saved {out}; n128 vs n192 relative difference {relative:.2e}", flush=True)


def main():
    for incidence in INCIDENCES:
        save(incidence, OUT / f"incidence_{incidence:g}")
    # The top-level state is the first incidence, copied rather than re-rendered.
    for name in ("camera_view.npz", "camera_view.json"):
        shutil.copyfile(OUT / f"incidence_{INCIDENCES[0]:g}" / name, OUT / name)


if __name__ == "__main__":
    main()

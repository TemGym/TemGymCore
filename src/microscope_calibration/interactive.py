import json
from pathlib import Path
import numpy as np
import panel as pn
import libertem.api as lt
from libertem_ui.figure import ApertureFigure
from libertem_ui.windows.com import CoMImagingWindow
from libertem_ui.windows.imaging import VirtualDetectorWindow, FrameImagingWindow
from microscope_calibration.udf import ShiftedSumUDF
from microscope_calibration.model import ModelParameters


def interactive_window(ctx: lt.Context, ds: lt.DataSet, model_params: ModelParameters):
    semi_conv_slider = pn.widgets.FloatSlider(
        name="Semiconv (mrad)",
        value=model_params["semi_conv"] * 1000,
        start=0.1,
        end=100,
        step=0.1,
    )
    defocus_slider = pn.widgets.FloatSlider(
        name="Defocus (m)",
        value=model_params["defocus"],
        start=-0.1,
        end=0.1,
        step=0.01,
    )
    camera_length_slider = pn.widgets.FloatSlider(
        name="Camera Length (m)",
        value=model_params["camera_length"],
        start=0.1,
        end=3,
        step=0.1,
    )
    flip_y_bool = pn.widgets.Checkbox(
        name="Flip-Y",
        value=model_params["flip_y"],
    )
    scan_rotation_slider = pn.widgets.FloatSlider(
        name="Scan rotation (deg.)",
        value=model_params["scan_rotation"],
        start=-180,
        end=180,
        step=1.0,
    )
    scan_step_input = pn.widgets.FloatInput(
        name="Scan step (mm)",
        value=min(model_params["scan_step"]) * 1000,
        start=0.1,
        end=0.3,
        step=0.01,
    )
    det_px_size_input = pn.widgets.FloatInput(
        name="Det px. size (mm)",
        value=min(model_params["det_px_size"]) * 1000,
        start=0.1,
        end=0.3,
        step=0.01,
    )
    descan_error = model_params["descan_error"]

    vi_window = VirtualDetectorWindow.using(ctx, ds)
    frame_window = FrameImagingWindow.linked_to(vi_window)
    com_window = CoMImagingWindow.linked_to(vi_window)

    result_fig = ApertureFigure.new(np.zeros(ds.shape.nav, dtype=np.float32))

    def get_model_parameters():
        return {
            "semi_conv": semi_conv_slider.value / 1000,
            "defocus": defocus_slider.value,  # Distance from the crossover to the sample
            "camera_length": camera_length_slider.value,  # distance from crossover to the detector
            "scan_step": (scan_step_input.value / 1000,) * 2,  # YX!
            "det_px_size": (det_px_size_input.value / 1000,) * 2,  # YX!
            "scan_rotation": scan_rotation_slider.value,
            "descan_error": descan_error,
            "flip_y": flip_y_bool.value,
        }

    px_shifts = model_params.get("px_shifts", None)
    if px_shifts is not None:
        px_shifts = ShiftedSumUDF.aux_data(
            px_shifts.astype(int), kind="nav", dtype=int, extra_shape=(2,)
        )

    def run_analysis(*e):
        try:
            run_btn.disabled = True
            udf = ShiftedSumUDF(
                model_parameters=get_model_parameters(), shifts=px_shifts
            )
            roi = np.random.choice([False] * 1 + [True] * 1, size=ds.shape.nav).astype(
                bool
            )
            for results in ctx.run_udf_iter(ds, udf, progress=False, roi=roi):
                shifted_sum = results.buffers[0]["shifted_sum"].data
                result_fig.update(shifted_sum)
        finally:
            run_btn.disabled = False

    run_btn = pn.widgets.Button(name="Run", button_type="success")
    run_btn.on_click(run_analysis)
    result_fig._toolbar.append(run_btn)

    shifted_sum_window = pn.Row(
        pn.Column(
            semi_conv_slider,
            defocus_slider,
            camera_length_slider,
            scan_rotation_slider,
            flip_y_bool,
            scan_step_input,
            det_px_size_input,
        ),
        pn.layout.Tabs(
            ("Frame Imaging", frame_window.layout()),
            ("Virtual Imaging", vi_window.layout()),
            ("CoM", com_window.layout()),
            # ("SpotSim", result_fig.layout),
            ("ShiftedSum", result_fig.layout),
        ),
    )
    return shifted_sum_window


if __name__ == "__main__":
    import jax

    jax.config.update("jax_platform_name", "cpu")

    rootdir = Path(__file__).parent
    ctx = lt.Context.make_with("inline")  # no parallelisation, good for debugging

    params_dict = json.load(open(rootdir / "params.json"))
    semi_conv = params_dict["semi_conv"]
    defocus = params_dict["defocus"]
    camera_length = params_dict["camera_length"]
    scan_step = params_dict["scan_step"]  # YX
    det_px_size = params_dict["det_px_size"]  # YX
    scan_rotation = params_dict["scan_rotation"]
    descan_error = params_dict["descan_error"]
    flip_y = params_dict["flip_y"]

    ds_path = rootdir / "fourdstem_array.npy"
    ds = ctx.load("npy", ds_path, num_partitions=4)

    model_parameters = ModelParameters(**{
        "semi_conv": semi_conv,
        "defocus": defocus,  # Distance from the crossover to the sample
        "camera_length": camera_length,  # distance from crossover to the detector
        "scan_step": scan_step,  # YX!
        "det_px_size": det_px_size,  # YX!
        "scan_rotation": scan_rotation,
        "descan_error": descan_error,
        "flip_y": False,
    })

    interactive_window(ctx, ds, model_parameters).show()

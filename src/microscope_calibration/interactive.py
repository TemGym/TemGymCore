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
        value=model_params["semi_conv"] * 1e3,
        start=0.001,
        end=10,
        step=0.1,
    )
    defocus_slider = pn.widgets.FloatSlider(
        name="Defocus (mm)",
        value=model_params["defocus"] * 1e3,
        start=-2,
        end=2,
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
        name="Scan step (um)",
        value=min(model_params["scan_step"]) * 1e6,
        start=0.01,
        end=10,
        step=0.01,
    )
    det_px_size_input = pn.widgets.FloatInput(
        name="Det px. size (um)",
        value=min(model_params["det_px_size"]) * 1e6,
        start=0.1,
        end=100,
        step=0.1,
    )
    descan_error = model_params["descan_error"]

    vi_window = VirtualDetectorWindow.using(ctx, ds)
    frame_window = FrameImagingWindow.linked_to(vi_window)
    com_window = CoMImagingWindow.linked_to(vi_window)

    result_fig = ApertureFigure.new(np.zeros(ds.shape.nav, dtype=np.float32))

    def get_model_parameters():
        return ModelParameters(
            semi_conv=float(semi_conv_slider.value) / 1e3,
            defocus=float(defocus_slider.value) / 1e3,
            camera_length=float(camera_length_slider.value),
            scan_step=(float(scan_step_input.value) / 1e6,) * 2,
            det_px_size=(float(det_px_size_input.value) / 1e6,) * 2,
            scan_rotation=float(scan_rotation_slider.value),
            descan_error=descan_error,
            flip_y=bool(flip_y_bool.value),
            scan_shape=tuple(ds.shape.nav),
            det_shape=tuple(ds.shape.sig),
        )

    def run_analysis(*e):
        try:
            run_btn.disabled = True
            udf = ShiftedSumUDF(
                model_parameters=get_model_parameters(),
            )
            for results in ctx.run_udf_iter(ds, udf, progress=False):
                shifted_sum = results.buffers[0]["shifted_sum"].data
                result_fig.update(shifted_sum)
        finally:
            run_btn.disabled = False

    run_btn = pn.widgets.Button(name="Run", button_type="success")
    run_btn.on_click(run_analysis)
    result_fig._toolbar.append(run_btn)

    shifted_sum_window = pn.Row(
        pn.layout.Tabs(
            ("Frame Imaging", frame_window.layout()),
            ("Virtual Imaging", vi_window.layout()),
            ("CoM", com_window.layout()),
            (
                "ShiftedSum",
                pn.Row(
                    pn.Column(
                        semi_conv_slider,
                        defocus_slider,
                        camera_length_slider,
                        scan_rotation_slider,
                        flip_y_bool,
                        scan_step_input,
                        det_px_size_input,
                    ),
                    result_fig.layout,
                )
            ),
        ),  # noqa
    )
    model_parameters = get_model_parameters()
    return shifted_sum_window, model_parameters

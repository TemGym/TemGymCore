import json
from pathlib import Path
import numpy as np
from numba import njit
import jax
import jax.numpy as jnp
import panel as pn
import libertem.api as lt
from libertem.udf import UDF

from jaxgym.stemoverfocus import project_frame_backward
import jaxgym.components as comp

rootdir = Path(__file__).parent
jax.config.update('jax_platform_name', 'cpu')


@njit
def mask_via_for(px_y, px_x, values, buffer):
    ny, nx = buffer.shape
    n = px_y.shape[0]
    for i in range(n):
        py = px_y[i]
        px = px_x[i]
        if 0 <= py < ny and 0 <= px < nx:
            buffer[py, px] += values[i]


class ShiftedSumUDF(UDF):
    def get_task_data(self):
        # Ran once per-partition and re-used
        params_dict = self.params.model_parameters
        crossover_z = jnp.zeros((1))
        PointSource = comp.PointSource(
            z=crossover_z,
            semi_conv=params_dict['semi_conv']
        )
        ScanGrid = comp.ScanGrid(
            z=jnp.array([params_dict['defocus']]),
            scan_step=params_dict['scan_step'],
            scan_shape=self.meta.dataset_shape.nav.to_tuple(),
            scan_rotation=params_dict['scan_rotation'],
        )

        Descanner = comp.Descanner(
            z=jnp.array([params_dict['defocus']]),
            descan_error=params_dict['descan_error'],
            offset_x=0.,
            offset_y=0.,
        )

        Detector = comp.Detector(
            z=jnp.array([params_dict['camera_length']]),
            det_shape=self.meta.dataset_shape.sig.to_tuple(),
            det_pixel_size=params_dict['det_px_size'],
        )

        model = [PointSource, ScanGrid, Descanner, Detector]
        scan_coords = ScanGrid.coords
        detector_coords = Detector.coords
        return {
            'model': model,
            'scan_coords': scan_coords,
            'detector_coords': detector_coords,
        }

    def get_result_buffers(self):
        dtype = np.result_type(
            self.meta.input_dtype,
            np.float32,
        )
        return {
            'shifted_sum': self.buffer(
                kind='single',
                dtype=dtype,
                extra_shape=self.meta.dataset_shape.nav,
            ),
        }

    def process_frame(self, frame: np.ndarray):
        scan_pos_flat = np.ravel_multi_index(
            self.meta.coordinates.ravel(),
            ds.shape.nav,
        )
        det_coords = self.task_data.detector_coords
        scan_pos = self.task_data.scan_coords[scan_pos_flat]
        model = self.task_data.model
        px_y, px_x, values = project_frame_backward(model, det_coords, frame, scan_pos)
        mask_via_for(np.array(px_y), np.array(px_x), np.array(values), self.results.shifted_sum)

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum


def interactive_window(ctx: lt.Context, ds: lt.DataSet, model_params):
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
        step=1.,
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
    descan_error = model_params['descan_error']

    from libertem_ui.figure import ApertureFigure
    result_fig = ApertureFigure.new(
        np.zeros(ds.shape.nav, dtype=np.float32)
    )

    def get_model_parameters():
        return {
            'semi_conv': semi_conv_slider.value / 1000,
            'defocus': defocus_slider.value,  # Distance from the crossover to the sample
            'camera_length': camera_length_slider.value,  # distance from crossover to the detector
            'scan_step': (scan_step_input.value / 1000,) * 2,  # YX!
            'det_px_size': (det_px_size_input.value / 1000,) * 2,  # YX!
            'scan_rotation': scan_rotation_slider.value,
            'descan_error': descan_error,
            'flip_y': flip_y_bool.value,
        }

    def run_analysis(*e):
        try:
            run_btn.disabled = True
            udf = ShiftedSumUDF(model_parameters=get_model_parameters())
            roi = np.random.choice([False] * 1 + [True] * 1, size=ds.shape.nav).astype(bool)
            for results in ctx.run_udf_iter(ds, udf, progress=True, roi=roi):
                shifted_sum = results.buffers[0]["shifted_sum"].data
                result_fig.update(shifted_sum)
        finally:
            run_btn.disabled = False

    run_btn = pn.widgets.Button(name="Run", button_type="success")
    run_btn.on_click(run_analysis)
    result_fig._toolbar.append(run_btn)

    return pn.Row(
        pn.Column(
            semi_conv_slider,
            defocus_slider,
            camera_length_slider,
            pn.Row(
                scan_rotation_slider,
                flip_y_bool,
            ),
            pn.Row(
                scan_step_input,
                det_px_size_input,
            ),
        ),
        pn.Column(
            result_fig.layout,
        ),
    )


if __name__ == "__main__":
    ctx = lt.Context.make_with("inline")  # no parallelisation, good for debugging
    # uses threads, might be efficient on data in memory
    # ctx = lt.Context.make_with("threads", cpus=64)
    # uses Dask+processes, cannot efficiently use data already in memory
    # ctx = lt.Context.make_with(cpus=8)

    params_dict = json.load(open(rootdir / 'params.json'))
    semi_conv = params_dict['semi_conv']
    defocus = params_dict['defocus']
    camera_length = params_dict['camera_length']
    scan_step = params_dict['scan_step']  # YX
    det_px_size = params_dict['det_px_size']  # YX
    scan_rotation = params_dict['scan_rotation']
    descan_error = params_dict['descan_error']
    flip_y = params_dict['flip_y']

    ds_path = rootdir / "fourdstem_array.npy"
    ds = ctx.load("npy", ds_path, num_partitions=4)

    # ds = ctx.load("memory", data=fourdstem_array)  # dataset with in-memory data, not from file
    model_parameters = {
        'semi_conv': semi_conv,
        'defocus': defocus,  # Distance from the crossover to the sample
        'camera_length': camera_length,  # distance from crossover to the detector
        'scan_step': scan_step,  # YX!
        'det_px_size': det_px_size,  # YX!
        'scan_rotation': scan_rotation,
        'descan_error': descan_error,
        'flip_y': False,
    }

    interactive_window(
        ctx, ds, model_parameters
    ).show(
        port=34677, address="grexp1396app", websocket_origin="grexp1396app:34677", open=False
    )

    # udf = ShiftedSumUDF(model_parameters=model_parameters)
    # # roi = np.zeros(ds.shape.nav, dtype=bool)
    # roi = np.random.choice([False] * 2 + [True] * 1, size=ds.shape.nav).astype(bool)
    # results = ctx.run_udf(ds, udf, progress=True, roi=roi)
    # shifted_sum: np.ndarray = results["shifted_sum"].data

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(shifted_sum, cmap='gray')
    # plt.colorbar()
    # plt.title("Shifted Sum")
    # plt.savefig(rootdir / "shifted_sum.png")
    # plt.close()

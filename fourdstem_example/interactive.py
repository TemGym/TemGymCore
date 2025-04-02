
import numpy as np
import libertem.api as lt
from jaxgym.stemoverfocus import project_frame_backward
import jaxgym.components as comp
import jax.numpy as jnp
from libertem.udf import UDF
import json
import jax
jax.config.update('jax_platform_name', 'cpu')

class ShiftedSumUDF(UDF):
    def get_task_data(self):
        # Ran once per-partition and re-used
        params_dict = self.params.model_parameters
        crossover_z = jnp.zeros((1))
        PointSource = comp.PointSource(z=crossover_z, semi_conv=params_dict['semi_conv'])
        ScanGrid = comp.ScanGrid(z=jnp.array([params_dict['defocus']]), 
                                scan_step=params_dict['scan_step'], 
                                scan_shape=params_dict['scan_shape'], 
                                scan_rotation=params_dict['scan_rotation'])

        Descanner = comp.Descanner(z=jnp.array([params_dict['defocus']]), 
                                descan_error=params_dict['descan_error'], 
                                offset_x=0., 
                                offset_y=0.)

        Detector = comp.Detector(z=jnp.array([params_dict['camera_length']]), 
                                det_shape=params_dict['det_shape'], 
                                det_pixel_size=params_dict['det_px_size'])

        model = [PointSource, ScanGrid, Descanner, Detector]
        scan_coords = ScanGrid.coords
        detector_coords = Detector.coords
        return {
            'model': model,
            'scan_coords': scan_coords,
            'detector_coords': detector_coords,
        }
    
    def get_result_buffers(self):
        # Defines the outputs, the true shape will be (1, scan_y, scan_x)
        # for technical reasons so need to be careful when indexing into it
        dtype = np.result_type(
            self.meta.input_dtype,
            np.float32,
        )
        return {
            'shifted_sum': self.buffer(
                kind='single',
                dtype=dtype,
                extra_shape=self.meta.dataset_shape,
                where='device',
            ),
        }
    
    def process_frame(self, frame: np.ndarray):
        scan_pos_flat = self.meta.coordinates[0][0]

        det_coords = self.task_data.detector_coords
        scan_pos = self.task_data.scan_coords[scan_pos_flat]
        model = self.task_data.model
        px_y, px_x, values = project_frame_backward(model, det_coords, frame, scan_pos)

        # Mask out indices that are out of bounds 
        valid_mask = (px_y >= 0) & (px_y < self.results.shifted_sum.shape[1]) & \
                     (px_x >= 0) & (px_x < self.results.shifted_sum.shape[2])
        
        px_y = px_y[valid_mask]
        px_x = px_x[valid_mask]
        values = values[valid_mask]

        # the zero here is because our output buffer is of type "single"
        np.add.at(self.results.shifted_sum[0], (px_y, px_x), values)

    def merge(self, dest, src):
        dest.shifted_sum += src.shifted_sum


if __name__ == "__main__":
    ctx = lt.Context.make_with("inline")  # no parallelisation, good for debugging
    # ctx = lt.Context.make_with("threads", cpus=8)  # uses threads, might be efficient on data in memory
    # ctx = lt.Context.make_with(cpus=8)  # uses Dask+processes, cannot efficiently use data already in memory

    ds_path = "fourdstem_example/fourdstem_array.npy"
    ds = ctx.load("npy", ds_path)
    params_dict = json.load(open('fourdstem_example/params.json'))
    semi_conv = params_dict['semi_conv']
    defocus = params_dict['defocus']
    camera_length = params_dict['camera_length']
    scan_shape = params_dict['scan_shape'] #YX
    det_shape = params_dict['det_shape'] #YX
    scan_step = params_dict['scan_step'] #YX
    det_px_size = params_dict['det_px_size'] #YX
    scan_rotation = params_dict['scan_rotation']
    descan_error = params_dict['descan_error']
    flip_y = params_dict['flip_y']

    # ds = ctx.load("memory", data=fourdstem_array)  # dataset with in-memory data, not from file
    model_parameters = {
        'semi_conv': semi_conv,
        'defocus': defocus, # Distance from the crossover to the sample
        'camera_length': camera_length, # distance from crossover to the detector
        'scan_shape': scan_shape, #YX!
        'det_shape': det_shape, # YX!
        'scan_step': scan_step, # YX!
        'det_px_size': det_px_size, #YX!
        'scan_rotation': scan_rotation,
        'descan_error': descan_error,
        'flip_y': False,
    }
    udf = ShiftedSumUDF(model_parameters=model_parameters)
    # roi = np.zeros(ds.shape.nav, dtype=bool)
    # roi[::2, ::2] = True  # roi for 1/4 of the frames
    results = ctx.run_udf(ds, udf, progress=True) #roi=roi)
    shifted_sum: np.ndarray = results["shifted_sum"].data
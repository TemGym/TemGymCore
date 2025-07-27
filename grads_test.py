import jax
from jaxgym.run import run_to_end
from jaxgym.ray import Ray
from jaxgym import components
from jax.tree_util import SequenceKey
from microscope_calibration.model import DescanErrorParameters
import microscope_calibration.components as comp


def get_key(k):
    try:
        return k.idx
    except AttributeError:
        pass
    try:
        return k.key
    except AttributeError:
        pass
    raise TypeError(f"Unrecognized key type {k}")


def run_with_grads(ray, model, grad_vars):
    model_params_path, tree = jax.tree.flatten_with_path(model)
    param_paths = list(p[0] for p in model_params_path)
    model_params = list(p[1] for p in model_params_path)

    grad_idxs = {}
    for var in grad_vars:
        path = var._build()
        original_path = var._build(original=True)
        com = path[0]
        for comidx, mcom in enumerate(model):
            if mcom is com:
                comidx = SequenceKey(comidx)
                break
        assert mcom is com, f"Component {com} not found in model"
        grad_path = (comidx,) + path[1:]

        for idx, param_path in enumerate(param_paths):
            if param_path[:len(grad_path)] == grad_path:
                grad_idxs[original_path + param_path[len(grad_path):]] = idx

    def run_wrap(*grad_params):
        grad_iter = iter(grad_params)
        grad_model_params = [
            p if ix not in grad_idxs.values()
            else next(grad_iter)
            for ix, p in enumerate(model_params)
        ]
        grad_model = jax.tree.unflatten(tree, grad_model_params)
        out = run_to_end(ray, grad_model)
        return out, out

    jac_fn = jax.jacobian(
        run_wrap,
        argnums=tuple(range(len(grad_idxs))),
        has_aux=True,
    )
    grad_params = list(model_params[idx] for idx in grad_idxs.values())
    grads, value = jac_fn(*grad_params)
    grads = {k: grads[i] for i, k in enumerate(grad_idxs.keys())}
    return value, grads


if __name__ == "__main__":
    model = (
        comp.PointSource(0., 0.01),
        (lens := components.Lens(0.5, 0.1)),
        (desc := comp.Descanner(0.75, 0.2, 0.1, DescanErrorParameters())),
        (det := comp.Detector(1., (0.001,) * 2, (128., 128.))),
    )

    ray = Ray(0.2, 0., 0., 0., 0., 0.)
    val, grads = run_with_grads(
        ray,
        model,
        (lens.params.focal_length, (det.params.det_shape)),
    )
    for k, v in grads.items():
        print(k)
        print(f"\t{v.item()}")

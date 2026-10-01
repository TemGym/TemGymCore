"""Shared helpers of the renderer benchmarks: timing, configuration overrides, result files.

The benchmarks report the *minimum* of several timed runs (a shared GPU only ever adds time)
and same-process ratios; ``benchmarks/compare.py`` fails a run whose timings exceed the stored
minima by more than a factor of three.
"""

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).parents[1]
RESULTS = Path(__file__).parent / "results"
sys.path.insert(0, str(ROOT / "tests"))

jax.config.update("jax_enable_x64", True)


def timeit(fn, repeats=5, warmup=1):
    """Minimum and median wall time in milliseconds of ``fn`` after ``warmup`` calls.

    Parameters
    ----------
    fn : callable
        Zero-argument function returning JAX arrays.
    repeats : int
        Number of timed calls.
    warmup : int
        Untimed calls first (compilation).

    Returns
    -------
    tuple of float
        Minimum and median time in milliseconds.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    return 1e3 * min(times), 1e3 * float(np.median(times))


def parse_config(assignments):
    """Build a ``RenderConfig`` from ``key=value`` strings (sweeps and CI).

    Tile fields take ``PIXELSxBEAMSxWARPS``, ``class_edges`` a comma list of radii in pixels
    (empty for none); everything else is parsed as a number or a plain string.
    """
    from temgym_core.render import RenderConfig, Tiles

    changes = {}
    for item in assignments or ():
        key, value = item.split("=", 1)
        if key in ("forward", "tangent", "adjoint", "rows"):
            changes[key] = Tiles(*(int(v) for v in value.split("x")))
        elif key == "class_edges":
            changes[key] = tuple(float(v) for v in value.split(",") if v)
        elif key in ("coordinates", "impl"):
            changes[key] = value
        else:
            changes[key] = float(value) if any(c in value for c in ".e") else int(value)
    return RenderConfig(**changes)


def gpu_state():
    """``nvidia-smi`` utilisation and memory of the first GPU (a shared GPU inflates timings)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"
    return out


def ruler_ms(repeats=5):
    """Time of a fixed float64 workload independent of the renderer (a GPU-speed yardstick).

    A shared GPU time-slices between processes and inflates every timing by roughly the same
    factor; ``compare.py`` divides that factor out with the ratio of two rulers.
    """
    import jax.numpy as jnp

    x = jnp.linspace(0.0, 1.0, 2**26)
    work = jax.jit(lambda v: jnp.sum(jnp.exp(-v * v) * jnp.cos(1e3 * v) * jnp.sin(7.0 * v)))
    return timeit(lambda: work(x), repeats)[0]


def header(config=None):
    """Metadata stored with every result file."""
    info = dict(date=date.today().isoformat(), jax=jax.__version__, device=str(jax.devices()[0]),
                gpu_at_start=gpu_state(), ruler_ms=ruler_ms())
    if config is not None:
        info["config"] = repr(config)
    return info


def write_results(name, results):
    """Write ``results`` to ``benchmarks/results/<name>.json`` and return the path."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, default=float) + "\n")
    return path


def print_table(title, cases, columns, ratio_to="fwd_ms"):
    """Print timings (ms) per case with the ratio of every column to ``ratio_to``."""
    print(f"\n{title}")
    head = f"{'case':<34s}" + "".join(f"{c:>13s}" for c in columns)
    print(head)
    print("-" * len(head))
    for name, row in cases.items():
        cells = "".join(f"{row[c]:13.2f}" if c in row else f"{'-':>13s}" for c in columns)
        print(f"{name:<34s}{cells}")
        if ratio_to in row:
            ratios = "".join(
                f"{row[c] / row[ratio_to]:12.2f}x" if c in row and c != ratio_to else f"{'':>13s}"
                for c in columns)
            print(f"{'  same-process ratio to ' + ratio_to:<34s}{ratios}")


def transforms(packed, evaluate, n_pix, what=("fwd_ms", "grad_ms", "jvp_ms", "jac7_ms"),
               repeats=5, seed=0):
    """Time the forward field and its derivative transforms for a bundle.

    Parameters
    ----------
    packed : array, shape (N, 12)
        Packed bundle.
    evaluate : callable
        ``evaluate(packed) -> field`` (jittable).
    n_pix : int
        Number of pixels of the field.
    what : sequence of str
        Subset of ``fwd_ms``, ``grad_ms``, ``jvp_ms`` (one tangent), ``jac7_ms`` (value and
        forward-mode Jacobian for seven directions).
    repeats : int
        Timed repetitions; the minimum is stored.
    seed : int
        Seed of the weights and directions.

    Returns
    -------
    dict
        Minimum times in milliseconds.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    out = {}
    if "fwd_ms" in what:
        forward = jax.jit(evaluate)
        out["fwd_ms"] = timeit(lambda: forward(packed), repeats)[0]
    if "grad_ms" in what:
        weights = jnp.asarray(rng.normal(size=n_pix))
        grad = jax.jit(jax.grad(lambda p: jnp.sum(weights * jnp.abs(evaluate(p)) ** 2)))
        out["grad_ms"] = timeit(lambda: grad(packed), repeats)[0]
    if "jvp_ms" in what:
        tangent = jnp.asarray(rng.normal(size=packed.shape))
        jvp = jax.jit(lambda p, t: jax.jvp(evaluate, (p,), (t,)))
        out["jvp_ms"] = timeit(lambda: jvp(packed, tangent), repeats)[0]
    if "jac7_ms" in what:
        basis = jnp.asarray(rng.normal(size=(7,) + packed.shape))

        def render_theta(theta):
            return evaluate(packed + jnp.einsum("c,cnj->nj", theta, basis))

        # jacfwd alone lets XLA drop the forward kernel: time the value with the tangents.
        value_and_jac = jax.jit(lambda th: (render_theta(th), jax.jacfwd(render_theta)(th)))
        out["jac7_ms"] = timeit(lambda: value_and_jac(jnp.zeros(7)), repeats)[0]
    return out

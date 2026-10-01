"""Speed, memory and gradient benchmark of the sampled-wavefront engine ``temgym_core.wave``.

Times one jitted ``apply_lct`` per branch (matrix-product MFT, FFT MFT, near-image, exact) at
1024^2, 2048^2 and 4096^2, a batch of 16 fields at 1024^2, and the gradient of a lens focal length
through ``column_transform -> plan_lct -> apply_lct`` at 1024^2. Every case runs in its own process so
that memory is measured cleanly. Results go to ``benchmarks/results/`` as JSON and are printed as a
table; ``benchmarks/compare_wave.py`` fails when a timing exceeds three times a stored minimum.

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.13 JAX_PLATFORMS=cuda \
        python benchmarks/wave.py --wait-idle --out benchmarks/results/wave_now.json

The GPU may be shared. ``--wait-idle`` polls ``nvidia-smi --query-compute-apps`` and starts a case
only when no other process uses the device (for up to ``--max-wait`` seconds per case); the JSON
records whether another process was present before and after each case, so a contended number is
never mistaken for a clean one. Report medians of at least 30 runs (``--repeats 30``).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BRANCHES = ("mft_matrix", "mft_fft", "near_image", "exact")
TARGET_MS = {1024: 5.0, 2048: None, 4096: 100.0}   # acceptance targets for one branch application
RESULTS = Path(__file__).resolve().parent / "results"


def other_gpu_processes():
    """Return ``[(pid, MiB), ...]`` of compute processes on the GPU, or ``None`` without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [line.split(",") for line in out.strip().splitlines() if line.strip()]
    return [(int(pid), int(mem)) for pid, mem in rows if int(pid) != os.getpid()]


def wait_until_idle(max_wait, poll=3.0):
    """Wait until no other process uses the GPU; return ``(processes at the end, waited seconds)``."""
    start = time.time()
    while True:
        busy = other_gpu_processes()
        if not busy or time.time() - start > max_wait:
            return busy, time.time() - start
        time.sleep(poll)


# ------------------------------------------------------------------------------------------------
# worker processes
# ------------------------------------------------------------------------------------------------
def _median_ms(function, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function().block_until_ready()
        times.append(time.perf_counter() - start)
    return 1e3 * float(np.median(times)), 1e3 * min(times), 1e3 * max(times)


def _branch_case(n, branch, repeats, batch=1):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from temgym_core.wave import (Lattice, adapted_input_lattice, apply_lct, collins_gaussian,
                                  gaussian_field, plan_lct)

    lam = 1.0
    S = np.array([[2.0, 0.3], [0.3, 2.4]])
    T = np.array([[-0.30, 0.02], [0.02, -0.25]])
    drift, lens = np.eye(4), np.eye(4)
    drift[:2, 2:] = S
    lens[2:, :2] = T
    M5 = np.eye(5)
    M5[:4, :4] = lens if branch == "exact" else drift @ lens
    out_lattice = Lattice.regular((n, n), 0.25)
    if branch.startswith("mft"):
        gamma = (0.9 if branch == "mft_matrix" else 1.0) / n
        in_lattice = adapted_input_lattice(M5[:2, 2:4], lam, out_lattice, (gamma, gamma), (n, n))
        method = "mft"
    else:
        in_lattice = Lattice.regular((n, n), 0.15)
        method = "exact" if branch == "exact" else "near_image"
    px = float(np.sqrt(abs(np.linalg.det(np.asarray(in_lattice.pitch)))))
    Q = np.diag([1j * lam / (np.pi * (n * px / 14.0) ** 2), 1j * lam / (np.pi * (n * px / 16.0) ** 2)])
    u = gaussian_field(in_lattice.positions(), 1.0, np.zeros(2), np.zeros(2), Q, lam)
    if batch > 1:
        u = u[None] * jnp.exp(1j * jnp.arange(batch))[:, None, None]
    u = jax.device_put(u)
    u.block_until_ready()

    start = time.perf_counter()
    plan = plan_lct(M5, lam, in_lattice, out_lattice if method == "mft" else None, method=method, check="off")
    jax.block_until_ready(jax.tree.leaves(plan))
    plan_first = time.perf_counter() - start
    start = time.perf_counter()
    plan = plan_lct(M5, lam, in_lattice, out_lattice if method == "mft" else None, method=method, check="off")
    jax.block_until_ready(jax.tree.leaves(plan))
    plan_second = time.perf_counter() - start

    apply = jax.jit(apply_lct)
    start = time.perf_counter()
    apply(plan, u).block_until_ready()
    compile_s = time.perf_counter() - start
    median, best, worst = _median_ms(lambda: apply(plan, u), repeats)
    stats = jax.devices()[0].memory_stats() or {}
    out = apply(plan, u)
    amp, centre, slopes, Qo, path = collins_gaussian(M5, lam, 1.0, np.zeros(2), np.zeros(2), Q)
    reference = gaussian_field(plan.out_lattice.positions(), amp, centre, slopes, Qo, lam, path)
    reference = reference[None] * jnp.exp(1j * jnp.arange(batch))[:, None, None] if batch > 1 else reference
    error = float(jnp.linalg.norm(out - reference) / jnp.linalg.norm(reference))
    return dict(case=f"batch{batch}_{branch}" if batch > 1 else branch, n=n, batch=batch,
                backend=jax.default_backend(), median_ms=median, min_ms=best, max_ms=worst,
                per_field_median_ms=median / batch, repeats=repeats, kind=plan.kind,
                modes=[plan.mode_x, plan.mode_y], compile_s=compile_s, plan_first_ms=1e3 * plan_first,
                plan_second_ms=1e3 * plan_second, rel_err_vs_analytic=error,
                peak_bytes=stats.get("peak_bytes_in_use"))


def _grad_case(n, repeats):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from temgym_core.components import Lens, Plane
    from temgym_core.constants import energy2wavelength
    from temgym_core.wave import (Lattice, adapted_input_lattice, apply_lct, column_transform,
                                  gaussian_field, plan_lct)

    voltage, f0, distance = 200e3, 2.0e-3, 3.0e-3
    lam = float(energy2wavelength(voltage))
    pitch = 1.2e-9 * (256 / n if n > 256 else 1.0)
    out_lattice = Lattice.regular((n, n), pitch)
    reference = column_transform([Lens(z=0.0, focal_length=f0), Plane(z=distance)], 0.0, voltage)
    in_lattice = adapted_input_lattice(reference.matrix[:2, 2:4], lam, out_lattice, (1.0 / n, 1.0 / n), (n, n))
    waist = n * float(np.asarray(in_lattice.pitch)[0, 0]) / 10.0
    u = jax.device_put(gaussian_field(in_lattice.positions(), 1.0, np.zeros(2), np.zeros(2),
                                      np.diag([1j * lam / (np.pi * waist**2)] * 2), lam))
    x, y = out_lattice.coords()
    weights = jax.device_put(jnp.exp(-(x**2 + y**2) / (2 * (8.0 * pitch * (n / 256 if n > 256 else 1.0)) ** 2)))

    def loss(f):
        transform = column_transform([Lens(z=0.0, focal_length=f), Plane(z=distance)], 0.0, voltage)
        plan = plan_lct(transform.matrix, lam, in_lattice, out_lattice, method="mft", check="off")
        intensity = jnp.abs(apply_lct(plan, u, transform.phase0)) ** 2
        return jnp.sum(weights * intensity) / jnp.sum(intensity)

    forward, gradient = jax.jit(loss), jax.jit(jax.grad(loss))
    forward(f0).block_until_ready()
    grad = gradient(f0)
    grad.block_until_ready()
    fwd_median, fwd_min, _ = _median_ms(lambda: forward(f0), repeats)
    grad_median, grad_min, _ = _median_ms(lambda: gradient(f0), repeats)
    h = 1e-5 * f0     # five-point stencil: the plain central difference is limited to ~1e-6
    fd = float((-forward(f0 + 2 * h) + 8 * forward(f0 + h) - 8 * forward(f0 - h) + forward(f0 - 2 * h)) / (12 * h))
    stats = jax.devices()[0].memory_stats() or {}
    return dict(case="grad_focal_length", n=n, batch=1, backend=jax.default_backend(),
                median_ms=grad_median, min_ms=grad_min, forward_median_ms=fwd_median, forward_min_ms=fwd_min,
                grad_over_forward=grad_median / fwd_median, repeats=repeats, grad=float(grad),
                fd_stencil5=fd, rel_diff_vs_fd=abs(float(grad) - fd) / abs(fd),
                peak_bytes=stats.get("peak_bytes_in_use"))


def _worker(spec, repeats):
    kind, n, name, batch = spec.split(":")
    result = _grad_case(int(n), repeats) if kind == "grad" else _branch_case(int(n), name, repeats, int(batch))
    print("RESULT " + json.dumps(result), flush=True)


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------
def _device_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def _table(results):
    header = f"{'case':<26}{'n':>6}{'median ms':>11}{'min ms':>9}{'target ms':>11}{'rel err':>10}  GPU"
    lines = [header, "-" * len(header)]
    for r in results:
        target = TARGET_MS.get(r["n"]) if r["case"] in BRANCHES else None
        shared = ("cpu" if r["backend"] == "cpu" else
                  "shared" if r.get("other_processes_before") or r.get("other_processes_after") else "idle")
        error = r.get("rel_err_vs_analytic", r.get("rel_diff_vs_fd"))
        extra = f"  (grad/forward {r['grad_over_forward']:.2f}x)" if "grad_over_forward" in r else ""
        lines.append(f"{r['case']:<26}{r['n']:>6}{r['median_ms']:>11.3f}{r['min_ms']:>9.3f}"
                     f"{'' if target is None else f'{target:g}':>11}{error:>10.1e}  {shared}{extra}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--branches", nargs="+", default=list(BRANCHES), choices=BRANCHES)
    parser.add_argument("--repeats", type=int, default=30, help="timed runs per case (after warm-up)")
    parser.add_argument("--batch", type=int, default=16, help="batch size of the batched case (0 to skip)")
    parser.add_argument("--batch-size-n", type=int, default=1024)
    parser.add_argument("--grad", action="store_true", default=True)
    parser.add_argument("--no-grad", dest="grad", action="store_false")
    parser.add_argument("--grad-size", type=int, default=1024)
    parser.add_argument("--wait-idle", action="store_true", help="wait for a GPU nobody else uses before each case")
    parser.add_argument("--max-wait", type=float, default=600.0, help="seconds to wait per case")
    parser.add_argument("--out", default=None, help="JSON file (default: benchmarks/results/wave_<timestamp>.json)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker, args.repeats)
        return

    jobs = [f"branch:{n}:{b}:1" for n in args.sizes for b in args.branches]
    if args.batch > 1:
        jobs.append(f"branch:{args.batch_size_n}:mft_fft:{args.batch}")
    if args.grad:
        jobs.append(f"grad:{args.grad_size}:focal_length:1")
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.13")
    results = []
    for job in jobs:
        waited, before = 0.0, None
        if args.wait_idle:
            before, waited = wait_until_idle(args.max_wait)
        else:
            before = other_gpu_processes()
        completed = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", job,
                                    "--repeats", str(args.repeats)], capture_output=True, text=True, env=env)
        lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
        if not lines:
            print("FAILED", job, completed.stderr[-1500:], file=sys.stderr)
            continue
        result = json.loads(lines[0][7:])
        after = other_gpu_processes()
        result.update(other_processes_before=before, other_processes_after=after, waited_s=waited)
        results.append(result)
        print(f"{result['case']:<26}{result['n']:>6}  median {result['median_ms']:.3f} ms  min {result['min_ms']:.3f} ms",
              flush=True)
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(), device=_device_name(),
                python=platform.python_version(), repeats=args.repeats, dtype="complex128")
    try:
        import jax
        meta["jax"] = jax.__version__
    except ImportError:
        pass
    print()
    print(_table(results))
    out = Path(args.out) if args.out else RESULTS / f"wave_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, results=results), indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

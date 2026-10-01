"""The speed-relevant paths of the wave engine on a GPU (``pytest -m gpu`` with ``JAX_PLATFORMS=cuda``).

Each branch is checked against an analytic Gaussian at a size where its speed matters, the batched path and
the focal-length gradient are checked, and a loose timing test guards against a lost fast path (the
strict numbers live in ``benchmarks/wave.py``; a shared GPU makes tight thresholds flaky).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from temgym_core.components import Lens, Plane
from temgym_core.constants import energy2wavelength
from temgym_core.wave import (
    Lattice,
    Wavefront,
    adapted_input_lattice,
    apply_lct,
    collins_gaussian,
    column_transform,
    gaussian_field,
    plan_lct,
    propagate,
)

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("needs a GPU: run with JAX_PLATFORMS=cuda")


def _matrix(branch):
    S = np.array([[2.0, 0.3], [0.3, 2.4]])
    T = np.array([[-0.30, 0.02], [0.02, -0.25]])
    drift, lens = np.eye(4), np.eye(4)
    drift[:2, 2:] = S
    lens[2:, :2] = T
    M5 = np.eye(5)
    M5[:4, :4] = lens if branch == "exact" else drift @ lens
    return M5


def _case(n, branch):
    """Plan and field of one benchmark branch in dimensionless units (wavelength 1)."""
    M5 = _matrix(branch)
    out = Lattice.regular((n, n), 0.25)
    if branch.startswith("mft"):
        gamma = (0.9 if branch == "mft_matrix" else 1.0) / n
        lattice = adapted_input_lattice(M5[:2, 2:4], 1.0, out, (gamma, gamma), (n, n))
        method = "mft"
    else:
        lattice = Lattice.regular((n, n), 0.15)
        method = "exact" if branch == "exact" else "near_image"
    px = float(np.sqrt(abs(np.linalg.det(np.asarray(lattice.pitch)))))
    Q = np.diag([1j / (np.pi * (n * px / 14.0) ** 2), 1j / (np.pi * (n * px / 16.0) ** 2)])
    u = gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2), Q, 1.0)
    plan = plan_lct(M5, 1.0, lattice, out if method == "mft" else None, method=method, check="off")
    amp, centre, slopes, Qo, path = collins_gaussian(M5, 1.0, 1.0, np.zeros(2), np.zeros(2), Q)
    return plan, u, gaussian_field(plan.out_lattice.positions(), amp, centre, slopes, Qo, 1.0, path)


BRANCHES = ["mft_matrix", "mft_fft", "near_image", "exact"]


@pytest.mark.parametrize("branch", BRANCHES)
def test_every_branch_matches_the_analytic_gaussian_at_1024(branch):
    plan, u, reference = _case(1024, branch)
    got = jax.jit(apply_lct)(plan, u)
    assert got.devices() == u.devices() and next(iter(got.devices())).platform == "gpu"
    assert float(jnp.linalg.norm(got - reference) / jnp.linalg.norm(reference)) < 1e-9
    assert (plan.mode_x, plan.mode_y) == ((1, 1) if branch == "mft_fft" else (0, 0))


@pytest.mark.parametrize("branch", BRANCHES)
def test_every_branch_stays_within_an_order_of_magnitude_of_its_speed_target(branch):
    """5 ms at 1024^2 is the target (about 0.1 to 0.4 ms measured on an idle GH200); 50 ms only fails if a
    fast path is lost or the transform runs on the host."""
    plan, u, _ = _case(1024, branch)
    apply = jax.jit(apply_lct)
    apply(plan, u).block_until_ready()
    times = []
    for _ in range(10):
        start = time.perf_counter()
        apply(plan, u).block_until_ready()
        times.append(time.perf_counter() - start)
    assert min(times) < 0.05, times


def test_a_batch_of_fields_shares_one_plan_on_the_gpu():
    plan, u, reference = _case(512, "mft_fft")
    batch = u[None] * jnp.exp(1j * jnp.arange(8))[:, None, None]
    got = jax.jit(apply_lct)(plan, batch)
    expected = reference[None] * jnp.exp(1j * jnp.arange(8))[:, None, None]
    assert got.shape == (8, 512, 512)
    assert float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected)) < 1e-9


def test_focal_length_gradient_on_the_gpu_matches_a_five_point_stencil():
    n, voltage, f0, distance = 512, 200e3, 2e-3, 3e-3
    lam = float(energy2wavelength(voltage))
    out = Lattice.regular((n, n), 1.2e-9 * 256 / n)
    reference = column_transform([Lens(z=0.0, focal_length=f0), Plane(z=distance)], 0.0, voltage)
    source = adapted_input_lattice(reference.matrix[:2, 2:4], lam, out, (1.0 / n, 1.0 / n), (n, n))
    waist = n * float(np.asarray(source.pitch)[0, 0]) / 10.0
    u = gaussian_field(source.positions(), 1.0, np.zeros(2), np.zeros(2), np.diag([1j * lam / (np.pi * waist**2)] * 2), lam)
    x, y = out.coords()
    weights = jnp.exp(-(x**2 + y**2) / (2 * (8.0 * 1.2e-9) ** 2))

    def loss(f):
        transform = column_transform([Lens(z=0.0, focal_length=f), Plane(z=distance)], 0.0, voltage)
        plan = plan_lct(transform.matrix, lam, source, out, method="mft", check="off")
        intensity = jnp.abs(apply_lct(plan, u, transform.phase0)) ** 2
        return jnp.sum(weights * intensity) / jnp.sum(intensity)

    grad = float(jax.jit(jax.grad(loss))(f0))
    h = 1e-5 * f0
    fd = float((-loss(f0 + 2 * h) + 8 * loss(f0 + h) - 8 * loss(f0 - h) + loss(f0 - 2 * h)) / (12 * h))
    assert abs(fd) > 0 and abs(grad - fd) < 1e-6 * abs(fd)


def test_propagate_runs_a_column_with_a_mask_on_the_gpu():
    from temgym_core.components import ConstantAmplitudeShift

    n, p, w0, voltage = 512, 10e-9, 5e-8, 200e3
    lam = float(energy2wavelength(voltage))
    lattice = Lattice.regular((n, n), p)
    Q = 1j * lam / (np.pi * w0**2) * np.eye(2)
    field = Wavefront(gaussian_field(lattice.positions(), 1.0, np.zeros(2), np.zeros(2), Q, lam), lattice, 0.0, voltage)
    column = [Lens(z=3e-3, focal_length=2e-3), Plane(z=4.6e-3)]
    plain = propagate(field, column, tol=1e-14)
    halved = propagate(field, [ConstantAmplitudeShift(z=3e-3, amplitude=0.5)] + column, tol=1e-14)
    assert next(iter(plain.data.devices())).platform == "gpu"
    assert float(jnp.linalg.norm(halved.data - 0.5 * plain.data) / jnp.linalg.norm(plain.data)) < 1e-12

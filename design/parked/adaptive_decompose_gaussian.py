"""Parked 2026-11-22: local refinement of a Gaussian decomposition (the principal set it aside).

It lived in src/temgym_core/source.py, after decompose_gaussian. Its splitting is exact (no split
gives decompose_gaussian to 1e-16; splitting all over 3 levels changes the field by 1.5e-4), but
it saved little: about 1.4x on a cross grating, 2 to 3x on a sparse aperture at 0.5 to 1 %, and
it stalled near 0.47 % where uniform refinement went on converging. See
design/changes/neoarm-first-version.md. To revive: paste back into source.py (it uses jax, jnp,
np and GaussianBeam from there) and put its two tests, in adaptive_decompose_gaussian_tests.py
next to this file, back into tests/test_converged_image.py.
"""

def adaptive_decompose_gaussian(
    beam: GaussianBeam, specimen, *, num_across: int, tolerance: float = 0.003,
    levels: int = 3, extent_rms: float = 6.0, overlap: float = 1.1, reach: float = 1e-5,
) -> GaussianBeam:
    """Decompose one round Gaussian as :func:`decompose_gaussian` does, finer only where needed.

    A thin specimen acts on each beamlet through the quadratic expansion of its complex action
    about the beamlet's centre. Where the action is not quadratic across a beamlet, the beamlet
    is split; elsewhere it stays. The beam must be at the specimen's plane, without ``through``.

    The beamlets sit on nested square lattices: level 0 is :func:`decompose_gaussian` with
    ``num_across``, each further level has half the step and half the waist. A level-``l``
    beamlet is split when ``|k (action - its quadratic expansion)|`` one and two waists from its
    centre, times the Gaussian envelope there (e^-1, e^-4) and times its amplitude times the
    brightest transmission it reaches, relative to the largest, exceeds ``tolerance`` (radians).
    A beamlet on the dark side of an edge counts, since its tail lights the bright side. A split
    beamlet is replaced by the exact decomposition of itself into the next level's beamlets. A child's position, slope, curvature and path depend only on its lattice
    site, so the children of neighbouring beamlets at one site are one beamlet: their amplitudes
    add. A level then costs four beamlets per split beamlet, plus the splitting kernel's reach.

    Parameters
    ----------
    beam : GaussianBeam
        One round Gaussian at the specimen's plane.
    specimen : GaussianActionComponent
        The thin element whose ``complex_action`` decides where to split.
    num_across : int
        Beamlets across level 0 (at least 8). An odd number keeps the lattices nested on the
        beam's centre.
    tolerance : float
        Largest weighted departure from the quadratic expansion, in radians, left unsplit.
    levels : int
        Most levels of splitting after level 0.
    extent_rms, overlap : float
        As in :func:`decompose_gaussian`.
    reach : float
        The splitting kernel is cut where its weight falls below this fraction of its peak, and
        beamlets below this fraction of the largest amplitude are dropped.

    Returns
    -------
    GaussianBeam
        Every beamlet of every level, at the beam's plane, to trace and render as usual.
    """
    num_across = int(num_across)
    if num_across < 8:
        raise ValueError("num_across must be at least 8.")
    if np.shape(beam.x) != () or np.shape(beam.Q_inv) != (2, 2):
        raise ValueError("Supply one Gaussian, not a beamlet bundle.")
    q = np.asarray(beam.Q_inv)
    if not np.allclose(q, q[0, 0] * np.eye(2), rtol=0.0, atol=1e-12 + 1e-9 * np.max(np.abs(q))):
        raise ValueError("The Gaussian must be round.")
    k, curvature = float(beam.k), float(q[0, 0].real)
    width = np.sqrt(2 / (k * q[0, 0].imag))
    step = extent_rms * width / num_across
    waist = overlap * step
    weight_width2 = width**2 - waist**2
    if weight_width2 <= 0:
        raise ValueError("Increase num_across: beamlets must be narrower than the beam.")
    # Level 0, on integer sites i * step about the beam's centre; as decompose_gaussian.
    half = (num_across - 1) / 2
    sites = np.arange(num_across) - half
    ii, jj = np.meshgrid(sites, sites)
    x0, y0 = ii * step, jj * step
    coefficients = (step**2 * width**2 / (np.pi * waist**2 * weight_width2)
                    * np.exp(-(x0 * x0 + y0 * y0) / weight_width2))
    coefficients = complex(beam.amplitude) * coefficients
    # Each level: (step, waist, index origin of the array, coefficient array over sites).
    level_arrays = [(step, waist, -half, coefficients)]

    def departure(xs, ys, level_waist):
        """The weighted departure of the action from its quadratic expansion, per site."""
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        unit = jnp.asarray(np.stack([np.cos(angles), np.sin(angles)], -1))

        def action(p):
            return specimen.complex_action(p, k)

        def one(c):
            a0 = action(c)
            gradient = (jax.grad(lambda p: jnp.real(action(p)))(c)
                        + 1j * jax.grad(lambda p: jnp.imag(action(p)))(c))
            hessian = (jax.hessian(lambda p: jnp.real(action(p)))(c)
                       + 1j * jax.hessian(lambda p: jnp.imag(action(p)))(c))
            worst = 0.0
            brightest = jnp.exp(-k * jnp.imag(a0))
            for radius, envelope in ((1.0, np.exp(-1.0)), (2.0, np.exp(-4.0))):
                d = radius * level_waist * unit
                exact = jax.vmap(action)(c + d)
                quadratic = a0 + d @ gradient + 0.5 * jnp.einsum("ni,ij,nj->n", d, hessian, d)
                worst = jnp.maximum(worst, envelope * jnp.max(jnp.abs(k * (exact - quadratic))))
                brightest = jnp.maximum(brightest, jnp.max(jnp.exp(-k * jnp.imag(exact))))
            # A beamlet on the dark side of an edge still lights the bright side with its tail.
            return worst, brightest

        centres = jnp.asarray(np.stack([xs + float(beam.x), ys + float(beam.y)], -1))
        worst, transmitted = jax.jit(jax.vmap(one))(centres)
        return np.asarray(worst), np.asarray(transmitted)

    largest = None
    for _ in range(levels):
        level_step, level_waist, origin, array = level_arrays[-1]
        active = np.argwhere(np.abs(array) > 0)
        if active.size == 0:
            break
        xs = (active[:, 1] + origin) * level_step
        ys = (active[:, 0] + origin) * level_step
        worst, transmitted = departure(xs, ys, level_waist)
        weight = np.abs(array[active[:, 0], active[:, 1]]) * transmitted
        if largest is None:
            largest = weight.max()
        split = worst * weight / largest > tolerance
        if not split.any():
            break
        # The exact decomposition of one level beamlet into the next level's.
        fine_step, fine_waist = level_step / 2, level_waist / 2
        parent2 = level_waist**2 - fine_waist**2
        radius = int(np.ceil(np.sqrt(parent2 * np.log(1 / reach)) / fine_step))
        offsets = np.arange(-radius, radius + 1)
        ox, oy = np.meshgrid(offsets * fine_step, offsets * fine_step)
        kernel = (fine_step**2 * level_waist**2 / (np.pi * fine_waist**2 * parent2)
                  * np.exp(-(ox * ox + oy * oy) / parent2))
        rows, columns = array.shape
        fine = np.zeros((2 * rows - 1 + 2 * radius, 2 * columns - 1 + 2 * radius), complex)
        for (row, column), amplitude in zip(active[split], array[active[split][:, 0],
                                                                 active[split][:, 1]]):
            r0, c0 = 2 * row, 2 * column
            fine[r0:r0 + 2 * radius + 1, c0:c0 + 2 * radius + 1] += amplitude * kernel
        array = array.copy()
        array[active[split][:, 0], active[split][:, 1]] = 0
        level_arrays[-1] = (level_step, level_waist, origin, array)
        level_arrays.append((fine_step, fine_waist, 2 * origin - radius, fine))

    # Beamlets far out in the kernels' tails carry almost nothing: keep those above ``reach``
    # of the largest amplitude.
    floor = reach * max(np.abs(array).max() for _, _, _, array in level_arrays)
    xs, ys, amplitudes, waists = [], [], [], []
    for level_step, level_waist, origin, array in level_arrays:
        active = np.argwhere(np.abs(array) > floor)
        xs.append((active[:, 1] + origin) * level_step)
        ys.append((active[:, 0] + origin) * level_step)
        amplitudes.append(array[active[:, 0], active[:, 1]])
        waists.append(np.full(len(active), level_waist))
    x, y = jnp.asarray(np.concatenate(xs)), jnp.asarray(np.concatenate(ys))
    waists = jnp.asarray(np.concatenate(waists))
    q_children = (curvature + 2j / (k * waists**2))[:, None, None] * jnp.eye(2)
    return GaussianBeam(
        x=beam.x + x, y=beam.y + y,
        dx=beam.dx + curvature * x, dy=beam.dy + curvature * y,
        z=beam.z, voltage=beam.voltage, amplitude=jnp.asarray(np.concatenate(amplitudes)),
        Q_inv=q_children,
        pathlength=beam.pathlength + beam.dx * x + beam.dy * y + 0.5 * curvature * (x * x + y * y),
    )

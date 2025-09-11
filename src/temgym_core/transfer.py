import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp
import numpy as np


def transfer_rays(ray_coords, transfer_matrices):
    """
    Apply cumulative transfer matrices to a batch of rays.

    This first builds cumulative products of the 5×5 matrices so that the
    m-th matrix equals the product of all up to m, then applies each to
    all input rays.

    Parameters
    ----------
    ray_coords : numpy.ndarray, shape (N, 5), float64
        Rays as rows [x, y, dx, dy, 1].
    transfer_matrices : numpy.ndarray, shape (M, 5, 5), float64
        Sequence of transfer matrices applied in order.

    Returns
    -------
    coords : numpy.ndarray, shape (N, M, 5), float64
        Transformed ray coordinates at each plane.

    Raises
    ------
    IndexError
        If `transfer_matrices` is empty.
    ValueError
        If shapes are incompatible.

    Notes
    -----
    Pure; uses numpy for compatibility with tests. Differentiability is not
    required for this helper.

    Examples
    --------
    >>> N, M = 2, 3
    >>> rays = np.hstack([np.zeros((N,4)), np.ones((N,1))])
    >>> Ts = np.tile(np.eye(5), (M,1,1))
    >>> out = transfer_rays(rays, Ts)
    >>> out.shape
    (2, 3, 5)
    """

    cumulative_matrices = accumulate_matrices_cumulative(transfer_matrices)

    # propagate all N rays through each of the M cumulative matrices
    # result[n, m, i] = sum_j cum_tms[m, i, j] * ray_coords[n, j]
    xy_coords = np.einsum("mij,nj->nmi", cumulative_matrices, ray_coords, optimize=True)

    return xy_coords


def transfer_rays_pt_src(input_pos_xy, input_slopes_xy, transfer_matrix):
    """
    Apply a single 5×5 transfer matrix to rays from a point source.

    Parameters
    ----------
    input_pos_xy : tuple of float
        Source position (x0, y0) in metres.
    input_slopes_xy : tuple of jnp.ndarray
        Slopes (dxs, dys) in radians; arrays of length N.
    transfer_matrix : jnp.ndarray, shape (5, 5)
        Homogeneous transfer matrix.

    Returns
    -------
    coords : jnp.ndarray, shape (4, N)
        Rows [x, y, dx, dy] after transformation.

    Raises
    ------
    ValueError
        If slope arrays have different lengths.

    Notes
    -----
    Pure and JIT-friendly. The input rays are constructed with a homogeneous
    1 row.

    Examples
    --------
    >>> x0, y0 = 0.0, 0.0
    >>> dxs = jnp.array([0.1, 0.0])
    >>> dys = jnp.array([0.0, -0.1])
    >>> T = jnp.eye(5)
    >>> out = transfer_rays_pt_src((x0, y0), (dxs, dys), T)
    >>> out.shape
    (4, 2)
    """

    # Given an input pt_source position and slopes, propagate the rays through the system
    input_pos_x, input_pos_y = input_pos_xy
    input_slopes_x, input_slopes_y = input_slopes_xy

    # Make the input rays we can run through one last time in the
    # model to find positions at sample and detector
    rays_at_source_with_semi_conv = jnp.vstack(
        [
            jnp.full(input_slopes_x.shape[0], input_pos_x),
            jnp.full(input_slopes_y.shape[0], input_pos_y),
            input_slopes_x,
            input_slopes_y,
            jnp.ones_like(input_slopes_x),
        ]
    )

    # Propagate the point source coordinates through the forward ABCD matrices
    coord_list = [rays_at_source_with_semi_conv]
    end_coords = jnp.dot(transfer_matrix, coord_list[-1])

    xs = end_coords[0, :]
    ys = end_coords[1, :]
    dxs = end_coords[2, :]
    dys = end_coords[3, :]

    coords = jnp.array([xs, ys, dxs, dys])

    return coords


def accumulate_matrices(matrices):
    """Multiply a sequence of 5×5 matrices right-to-left.

    Parameters
    ----------
    matrices : sequence of jnp.ndarray, each shape (5, 5)
        Transfer matrices ordered as applied.

    Returns
    -------
    total : jnp.ndarray, shape (5, 5)
        Product matrices[-1] @ ... @ matrices[0].

    Raises
    ------
    IndexError
        If `matrices` is empty.
    """
    total = matrices[-1]
    for tm in reversed(matrices[:-1]):
        total = total @ tm
    return total


def accumulate_matrices_cumulative(matrices):
    """Build cumulative products of 5×5 matrices in forward order.

    Parameters
    ----------
    matrices : numpy.ndarray or jnp.ndarray, shape (M, 5, 5)
        Transfer matrices.

    Returns
    -------
    cumulative : jnp.ndarray, shape (M, 5, 5)
        cumulative[k] = matrices[k] @ ... @ matrices[0].

    Raises
    ------
    IndexError
        If `matrices` is empty.

    Examples
    --------
    >>> Ms = jnp.stack([jnp.eye(5), 2*jnp.eye(5)], 0)
    >>> cum = accumulate_matrices_cumulative(Ms)
    >>> cum.shape
    (2, 5, 5)
    """
    all_matrices = []
    # Begin with the last matrix
    total = matrices[-1]
    all_matrices.append(total)
    # Multiply backwards through the list
    for tm in reversed(matrices[:-1]):
        total = tm @ total
        all_matrices.append(total)
    return jnp.stack(all_matrices, axis=0)

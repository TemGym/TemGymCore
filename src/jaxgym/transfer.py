import jax.numpy as jnp
import numpy as np


def transfer_rays(ray_coords, transfer_matrices):
    """
    Propagate rays through an optical system using the provided transfer matrices.
    This version first builds the cumulative product of the transfer matrices so
    that each plane’s matrix is the product of all preceding ones, then applies
    each to the input rays via einsum.

    Parameters
    ----------
    ray_coords : numpy.ndarray
        A 2D array of shape (N, 5) for N rays: [x, y, dx, dy, 1].
    transfer_matrices : numpy.ndarray
        A 3D array of shape (M, 5, 5) for M sequential transfer matrices.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (N, M, 5): ray coords at each of the M planes.
    """

    cumulative_matrices = accumulate_matrices_cumulative(transfer_matrices)

    # propagate all N rays through each of the M cumulative matrices
    # result[n, m, i] = sum_j cum_tms[m, i, j] * ray_coords[n, j]
    xy_coords = np.einsum("mij,nj->nmi", cumulative_matrices, ray_coords, optimize=True)

    return xy_coords


def transfer_rays_pt_src(input_pos_xy, input_slopes_xy, transfer_matrix):
    """
    Propagate rays through an optical system using the provided transfer matrix.
    This function takes an initial point source position (x, y) and their corresponding
    slopes, constructs a ray vector, and propagates these rays through the system by
    applying the transfer matrix. The output is a set of propagated ray coordinates.
    Parameters
    ----------
    input_pos_xy : tuple
        A tuple (input_pos_x, input_pos_y) representing the x and y coordinates of the
        source position.
    input_slopes_xy : tuple
        A tuple (input_slopes_x, input_slopes_y) representing the slopes of the rays
        in the x and y directions.
    transfer_matrix : numpy.ndarray
        A matrix used to propagate the rays. It should be compatible with the constructed
        ray vector so that the dot product results in a propagated coordinate array.
    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (4, N) where N is the number of rays. The rows correspond to:
            - x positions
            - y positions
            - x slopes (dxs)
            - y slopes (dys)
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
    """
    Compute the total transfer matrix by
    multiplying in right-to-left order
    """
    total = matrices[-1]
    for tm in reversed(matrices[:-1]):
        total = total @ tm
    return total


def accumulate_matrices_cumulative(matrices):
    """
    Compute cumulative products in reverse order:
    start with the last matrix, then multiply by
    the one before it, and so on, return all
    intermediate matrices in a stack
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

import jax.numpy as jnp


def transfer_rays(input_pos_xy, input_slopes_xy, transfer_matrix):
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


def accumulate_transfer_matrices(transfer_matrices, start: int, end: int):
    """Compute the total transfer matrix between component indices [start, end]
    by multiplying in right-to-left order.

    Given that the transfer_matrices list contains both component and
    intermediate propagation matrices, where each component appears at even indices (0, 2, 4, ...),
    this function multiplies the matrices from index 2*start up to index 2*end using reversed order.
    """
    i_start = 2 * start
    i_end = 2 * end
    matrices = transfer_matrices[i_start: i_end + 1]
    total = matrices[-1]
    for tm in reversed(matrices[:-1]):
        total = total @ tm
    return total

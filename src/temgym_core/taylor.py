import itertools
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import factorial
from temgym_core.ray import Ray
import sympy as sp


def order_indices(max_order, n_vars):
    unique_multi_indices = set()

    # Start at 0 to include the constant (zero) multi-index
    for order in range(0, max_order + 1):
        for deriv in itertools.product(range(n_vars), repeat=order):
            multi_index = [0] * n_vars
            for d in deriv:
                multi_index[d] += 1
            unique_multi_indices.add(tuple(multi_index))

    # Sort using graded order: first by total order, then by reversed tuple.
    # This makes the last exponent the most significant within each group.
    return np.array(
        sorted(
            unique_multi_indices, key=lambda x: (sum(x), tuple(-i for i in reversed(x)))
        )
    )


def poly_dict(derivatives, selected_variables, multi_indices):
    polynomial_dict = {}

    # Loop through the selected variables we want to form a polynomial from.
    for var in selected_variables:
        # Loop through the multi-indices of the partial derivatives
        for multi_idx in multi_indices:
            # Get the order of the derivative
            order = sum(multi_idx)

            # Get the partials of this variable and this order of derivative
            # order-1 because the first order partials are stored in the 0th index.
            partials_dataclass = getattr(derivatives[order - 1], var)

            # Get the nonzero indices of this multi-index
            nonzero_indices = jnp.flatnonzero(multi_idx)
            # If there are no nonzero indices, skip this multi-index
            if len(nonzero_indices) == 0:
                continue

            # Loop through nonzero multi-indices
            for idx in nonzero_indices:
                # For the non-zero indices, get the value of the multi-index
                # telling us how many partial derivatives with respect to that variable we want
                num_partials_of_var = multi_idx[idx]

                # Make repeated calls to the partials dataclass
                # to get the value of the partial derivative
                for _ in range(num_partials_of_var):
                    partials = getattr(partials_dataclass, selected_variables[idx])
                    if isinstance(partials, Ray):
                        partials_dataclass = partials
                    else:
                        continue

            # before iterating, make sure `partials` is iterable
            try:
                iter(partials)
            except TypeError:
                partials = [partials]

            for i, partial in enumerate(partials):
                if np.abs(partial) < 1e-15:
                    # If the partial is too small, skip it
                    continue

                # Add the final taylor coeff to the multi-index dictionary
                taylor_coeff_factor = 1 / np.prod(factorial(multi_idx))
                if i not in polynomial_dict:
                    # create a new entry for this output‐index, with sub‐lists for each variable
                    polynomial_dict[i] = {v: [] for v in selected_variables}

                polynomial_dict[i][var].append(
                    multi_idx.tolist() + [float(partial) * taylor_coeff_factor]
                )

    return polynomial_dict


def poly_dict_to_sympy_expr(multi_index_array, var_list, sym_vars=None):
    """
    Converts the multi-index representation for given variable(s) into sympy expression(s).

    Parameters:
      multi_index_array: dict
          Dictionary whose keys are variable names and values are lists of terms.
          Each term is a list where the first n entries are the exponents for each
          independent symbol, and the last entry is the coefficient.
      var_list: str or list of str
          The key or keys in multi_index_array for which to form the polynomial.
      sym_vars: list of sympy.Symbol, optional
          The symbols to be used in the polynomial. If None, defaults to generic symbols
          x0, x1, ..., x{n-1}.

    Returns:
      sympy.Expr or dict: If a single variable is provided, returns the simplified sympy expression.
                          If a list is provided, returns a dictionary mapping each key to its
                          expression.
    """
    if isinstance(var_list, str):
        var_list = [var_list]

    results = {}
    for var in var_list:
        terms = multi_index_array[var]
        n_vars = len(terms[0]) - 1  # number of independent symbols per term
        if sym_vars is None:
            curr_sym_vars = sp.symbols("x0:%d" % n_vars)
        else:
            curr_sym_vars = sym_vars
        expr = sp.Integer(0)
        for term in terms:
            exponents = term[:-1]
            coeff = term[-1]
            monomial = sp.Integer(1)
            for s, exp in zip(curr_sym_vars, exponents):
                monomial *= s ** int(exp)
            expr += coeff * monomial
        results[var] = sp.simplify(expr)

    if len(results) == 1:
        return next(iter(results.values()))
    return results

"""Numerical configuration: temgym-core needs 64-bit JAX arithmetic.

Electron wavelengths are picometres, so the phase ``k * path`` of a column's
excess optical path still reaches 1e9 rad and the phases inside the renderer 1e10 rad.
Single precision silently produces wrong images,
so entry points call :func:`require_x64` instead of downgrading quietly. Nothing
here runs at import time; enabling 64-bit arithmetic is the caller's choice::

    import temgym_core
    temgym_core.enable_x64()
"""

import jax


def x64_enabled() -> bool:
    """Return whether JAX currently uses 64-bit floats and integers."""
    return bool(jax.config.jax_enable_x64)


def enable_x64() -> None:
    """Enable 64-bit JAX arithmetic. Call once, before creating any array."""
    jax.config.update("jax_enable_x64", True)


def require_x64(what: str = "temgym-core") -> None:
    """Raise a clear error unless 64-bit arithmetic is enabled.

    Parameters
    ----------
    what : str
        Name of the calling function or feature, used in the message.

    Raises
    ------
    RuntimeError
        If ``jax_enable_x64`` is off.
    """
    if not x64_enabled():
        raise RuntimeError(
            f"{what} needs 64-bit JAX arithmetic. Call temgym_core.enable_x64() "
            "(or jax.config.update('jax_enable_x64', True)) before creating arrays."
        )

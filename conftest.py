import os

import jax

# Keep ordinary tests on CPU, but allow explicitly requested GPU test runs.
if not (os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME")):
    jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

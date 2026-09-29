import os

# Do not grab 75% of a shared GPU when GPU tests are requested.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import pytest  # noqa: E402

# Keep ordinary tests on CPU, but allow explicitly requested GPU test runs.
if not (os.environ.get("JAX_PLATFORMS") or os.environ.get("JAX_PLATFORM_NAME")):
    jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def pytest_collection_modifyitems(items):
    """Tests that request the ``gpu`` fixture belong to the ``gpu`` mark, so ``-m gpu`` finds them."""
    for item in items:
        if "gpu" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.gpu)

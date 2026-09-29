"""64-bit policy: explicit, never a silent downgrade to float32."""
import jax
import jax.numpy as jnp
import pytest

import temgym_core
from temgym_core import config
from temgym_core.components import Lens
from temgym_core.evaluate import evaluate_gaussians
from temgym_core.ray import Ray
from temgym_core.run import run_to_end


@pytest.fixture
def x64_disabled():
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", True)


def test_x64_is_enabled_by_the_test_configuration():
    assert config.x64_enabled()
    config.require_x64("anything")  # does not raise


def test_enable_x64_turns_the_flag_on():
    jax.config.update("jax_enable_x64", False)
    try:
        temgym_core.enable_x64()
        assert config.x64_enabled()
        assert jnp.zeros(1).dtype == jnp.float64
    finally:
        jax.config.update("jax_enable_x64", True)


def test_require_x64_explains_how_to_fix_it(x64_disabled):
    with pytest.raises(RuntimeError, match=r"enable_x64\(\)") as error:
        config.require_x64("the widget")
    assert "the widget" in str(error.value)


def test_propagation_and_rendering_refuse_float32(x64_disabled):
    ray = Ray(x=0.0, y=0.0, dx=0.0, dy=0.0, z=0.0, pathlength=0.0)
    with pytest.raises(RuntimeError, match="64-bit"):
        run_to_end(ray, (Lens(z=0.5, focal_length=1.0),))
    with pytest.raises(RuntimeError, match="64-bit"):
        evaluate_gaussians(None, None)


def test_importing_the_package_does_not_change_the_flag():
    import subprocess
    import sys

    code = "import jax, temgym_core; print(jax.config.jax_enable_x64)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "JAX_ENABLE_X64": "0",
                              "JAX_PLATFORMS": "cpu"})
    assert out.stdout.strip().splitlines()[-1] == "False"

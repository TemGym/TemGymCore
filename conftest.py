import jax

jax.config.update("jax_enable_x64", True)  # noqa: E702
jax.config.update("jax_platform_name", "cpu")

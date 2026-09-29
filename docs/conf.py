"""Sphinx configuration for temgym-core."""
from importlib.metadata import version as _version

project = "temgym-core"
author = "David Landers, Matthew Bryan, Dieter Weber"
copyright = "2023-2026, the TemGym authors"
release = _version("temgym-core")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
# Small pages execute at build time so the documented code is always correct.
# Heavy GPU notebooks are executed by tools/run_notebooks.py and embedded with
# their stored outputs (list them in nb_execution_excludepatterns).
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_execution_excludepatterns = ["tutorials/*.ipynb"]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]
html_theme = "furo"
html_title = f"temgym-core {version}"

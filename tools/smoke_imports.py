"""Import every module of an installed temgym_core; exit non-zero on any failure.

Run it in an environment that has only the declared dependencies installed, so
an undeclared import (for example a module-level ``import sympy``) fails here
instead of in a user's environment::

    python tools/smoke_imports.py            # core install: extras-only modules may fail helpfully
    python tools/smoke_imports.py --strict   # extras installed: every module must import

A module that needs an optional extra may fail to import, but only with an
``ImportError`` that names the extra (``pip install 'temgym-core[plot]'``).
"""
import importlib
import pkgutil
import sys

import temgym_core

# module -> the extra that provides its optional dependency
NEEDS_EXTRA = {"temgym_core.plotting": "plot"}

strict = "--strict" in sys.argv
failures, skipped = [], []
names = sorted(m.name for m in pkgutil.walk_packages(temgym_core.__path__, "temgym_core."))
for name in ["temgym_core", *names]:
    try:
        importlib.import_module(name)
    except ImportError as error:
        extra = NEEDS_EXTRA.get(name)
        if extra and not strict and f"temgym-core[{extra}]" in str(error):
            skipped.append((name, extra))
        else:
            failures.append((name, f"{type(error).__name__}: {str(error).splitlines()[0]}"))
    except Exception as error:  # report every failure, not just the first
        failures.append((name, f"{type(error).__name__}: {str(error).splitlines()[0]}"))

total = len(names) + 1
print(f"imported {total - len(failures) - len(skipped)}/{total} modules of temgym_core "
      f"{getattr(temgym_core, '__version__', '')}")
for name, extra in skipped:
    print(f"  needs extra [{extra}]: {name}")
for name, message in failures:
    print(f"  FAIL {name}: {message}")
sys.exit(1 if failures else 0)

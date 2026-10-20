"""Minimal setup script to make working with tests easy.

The importable code lives in ``src`` as top-level packages
(``IBM_Emulation``, ``Gene_Drive_Optimisation`` …) and modules (``paths`` …).
We install them via ``package_dir`` so they are importable from anywhere the
package is installed, not just from inside ``src`` or through pytest's
``pythonpath``.

The distribution is named separately from the working repository it was frozen
from, because both install the same top-level package names: install this one
into its own virtual environment.
"""

from setuptools import find_packages, setup

SRC = "src"

setup(
    name="Optimisation_Approximation_Underdominance_Code",
    version="1.0.0",
    package_dir={"": SRC},
    packages=find_packages(where=SRC),
    # Standalone top-level scripts/modules in src/ that are imported by name
    # (e.g. ``import paths``) rather than as part of a package.
    py_modules=[
        "paths",
        "cpu_resources",
    ],
)

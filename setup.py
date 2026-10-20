"""Minimal setup script to make working with tests easy.

The importable code lives in ``Coarse_Graining_IBMS/src`` as top-level packages
(``IBM_Emulation``, ``Gene_Drive_Optimisation`` …) and modules (``paths`` …).
We install them via ``package_dir`` so they are importable everywhere the
package is installed — crucially including Ray worker subprocesses, which do
*not* inherit pytest's ``pythonpath`` and otherwise fail with
``ModuleNotFoundError``.

The distribution is named separately from the working repository it was frozen
from, because both install the same top-level package names: install this one
into its own virtual environment.
"""

from setuptools import find_packages, setup

SRC = "Coarse_Graining_IBMS/src"

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
